# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
VDT training script for occupancy grid map video prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

import wandb
from torchvision.utils import make_grid

from models import VDT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from mask_generator import VideoMaskGenerator
from dataloader import PredOccDataset
from data_preprocessing import preprocess_batch
from autoencoder import SequenceAutoencoderKL
from omegaconf import OmegaConf
from util import instantiate_from_config

def make_video_3(batch_out):
    """Convert preprocessed maps to 3-channel video tensor: (B, 2T, 3, H, W).
    First T frames: past (input), next T frames: future (target to predict).
    """
    input_binary_maps  = batch_out["input_binary_maps"].float()   # (B, T, 1, H, W) - past
    mask_binary_maps   = batch_out["mask_binary_maps"].float()    # (B, T, 1, H, W) - future
    input_occ_grid_map = batch_out["input_occ_grid_map"].float()  # (B, H, W)
    B, T, _, H, W = input_binary_maps.shape
    static_map = input_occ_grid_map.unsqueeze(1).unsqueeze(2).expand(B, T, 1, H, W)  # (B, T, 1, H, W)

    # Past frames (input)
    dynamic_past = input_binary_maps * (1.0 - static_map)
    free_past    = 1.0 - input_binary_maps
    past_3ch     = torch.cat([dynamic_past, static_map, free_past], dim=2)   # (B, T, 3, H, W)

    # Future frames (target)
    dynamic_future = mask_binary_maps * (1.0 - static_map)
    free_future    = 1.0 - mask_binary_maps
    future_3ch     = torch.cat([dynamic_future, static_map, free_future], dim=2)  # (B, T, 3, H, W)

    return torch.cat([past_3ch, future_3ch], dim=1)  # (B, 2T, 3, H, W)


def make_video(batch_out):
    """Convert preprocessed maps to a 1-channel video tensor: (B, 2T, 1, H, W).
    First T frames: past (input), next T frames: future (target to predict).
    """
    input_binary_maps  = batch_out["input_binary_maps"].float()   # (B, T, 1, H, W) - past
    mask_binary_maps   = batch_out["mask_binary_maps"].float()    # (B, T, 1, H, W) - future

    return torch.cat([input_binary_maps, mask_binary_maps], dim=1)  # (B, 2T, 1, H, W)


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def compute_batch_iou(pred, gt, threshold=0.1):
    """Compute mean IoU for occupancy maps over batch and time."""
    pred_occ = pred > threshold
    gt_occ = gt > threshold

    intersection = (pred_occ & gt_occ).sum(dim=(-1, -2, -3)).float()
    union = (pred_occ | gt_occ).sum(dim=(-1, -2, -3)).float()
    return (intersection / (union + 1e-6)).mean()

#################################################################################
#                              Evaluation Loop                                  #
#################################################################################

@torch.no_grad()
def evaluate(ema, ae, diffusion, args, device, rank, train_steps, logger):
    """Run 'predict' task on one test batch, compute per-frame IoU, log GT vs Gen."""

    T_half = args.num_frames // 2  # 10
    latent_size = args.image_size // 4
    eval_diffusion = create_diffusion(str(args.eval_sampling_steps))

    test_dataset = PredOccDataset(data_root=args.eval_data_path, split="val")
    test_loader  = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    batch = next(iter(test_loader))

    batch_out = preprocess_batch(batch, device=device)
    x_video   = make_video(batch_out)           # (B, 2T, 1, H, W) in [0,1]
    B, TT, C, H, W = x_video.shape
    x_flat = x_video.view(-1, C, H, W)          # (B*2T, 1, H, W)

    raw_x = x_flat
    # Encode all frames
    #z_frames = vae.encode(x_flat).latent_dist.sample().mul_(0.18215)
    posterior = ae.encode(x_video)
    z_frames = posterior.mode()
    lat_c = ae.embed_dim
    z_frames = z_frames.view(B, args.num_frames, lat_c, z_frames.shape[-2], z_frames.shape[-1])  # (B, 2T, 4, lat_h, lat_w)
    z_noise = torch.randn(B, args.num_frames, lat_c, latent_size, latent_size, device=device)

    # 'predict' mask: mask out future T frames
    generator = VideoMaskGenerator((z_frames.shape[-4], z_frames.shape[-2], z_frames.shape[-1]))
    mask = generator(B, device, idx=0)  # idx=0 -> predict
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    val_loss = diffusion.training_losses(ema, z_frames, t, mask=mask)["loss"].mean()

    z_perm  = z_noise.permute(0, 2, 1, 3, 4)  # (B, 4, 2T, lat_h, lat_w)

    samples = eval_diffusion.p_sample_loop(
        ema.forward, z_perm.shape, z_perm,
        clip_denoised=False, progress=True, device=device,
        raw_x=z_frames, mask=mask,
    )
    # samples: (4, B, 2T, lat_h, lat_w)
    samples = samples.permute(1, 0, 2, 3, 4) * mask + z_frames.permute(2, 0, 1, 3, 4) * (1 - mask)
    samples = samples.permute(1, 2, 0, 3, 4)  # (B, 2T, 4, lat_h, lat_w)
    samples_flat = samples.reshape(-1, lat_c, latent_size, latent_size)  # (B*2T, 4, ...)
    

    # Decode

    decoded = ae.decode(samples_flat) 
    samples = decoded.reshape(B, args.num_frames, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

    raw_x = raw_x.reshape(-1, args.num_frames, raw_x.shape[-3], raw_x.shape[-2], raw_x.shape[-1])
    mask = F.interpolate(mask.float(), size=(raw_x.shape[-2], raw_x.shape[-1]), mode='nearest').unsqueeze(2)
    raw_x = raw_x * (1 - mask)

    pred_future = samples[:, T_half:]
    gt_future = x_video[:, T_half:]
    val_iou = compute_batch_iou(pred_future, gt_future)

    samples = torch.cat([x_video, raw_x, samples], dim=1)

    vis = samples[0].cpu().clamp(0, 1)
    
    grid = make_grid(vis, nrow=args.num_frames, normalize=False, value_range=(0, 1))
    wandb.log({
        "val/loss": val_loss.item(),
        "val/iou": val_iou.item(),
        "eval/generated": wandb.Image(
            grid,
            caption=f"Top: GT  Middle: input  Bottom: generated  step={train_steps}"
        )
    }, step=train_steps)
    logger.info(
        f"(step={train_steps:07d}) Validation logged to wandb. "
        f"loss={val_loss.item():.6f}, iou={val_iou.item():.4f}"
    )
 
#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    backend = "nccl" if world_size > 1 else "gloo"
    dist.init_process_group(backend)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        wandb.init(
            project="VDT",
            name=f"{experiment_index:03d}-{model_string_name}",
            config=vars(args),
            dir=experiment_dir,
        )
    else:
        logger = create_logger(None)

    # Create VDT model:
    assert args.image_size % 4 == 0, "Image size must be divisible by 4 (for the VAE encoder)."
    latent_size = args.image_size // 4
    additional_kwargs = {'num_frames': args.num_frames,
    'mode': 'video'} 
    model = VDT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=2,
        **additional_kwargs
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion("", training=True)  # training uses all 1000 timesteps
    #vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    cfg = OmegaConf.load("ae_vdt/ae_vdt.yaml")

    ae = instantiate_from_config(cfg.model)   # cfg.model.target == autoencoder.SequenceAutoencoderKL
    ae.init_from_ckpt("ae_vdt/model.ckpt")
    ae = ae.to(device)
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False

    logger.info(f"VDT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # eval_data_path fallback
    if args.eval_data_path is None:
        args.eval_data_path = args.data_path

    # Setup PredOcc dataset:
    dataset = PredOccDataset(data_root=args.data_path, split="train")
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} samples ({args.data_path})")

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    train_steps  = 0
    log_steps    = 0
    running_loss = 0
    start_time   = time()
    num_tasks    = 7  # VideoMaskGenerator task idx 0~6

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            # Preprocess -> 1-channel video (B, T, 1, H, W)
            batch_out = preprocess_batch(batch, device=device)
            x = make_video(batch_out)  # (B, T, 1, H, W) (T=20: past 10 + future 10)
            B, T, C, H, W = x.shape

            if rank == 0:
                log_frames = x[:T].detach().cpu()  # (2T, 1, H, W) - first sample

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                #x = ae.encode(x).latent_dist.sample().mul_(0.18215) # (B*T, 1, H, W)
                posterior = ae.encode(x)
                x = posterior.mode()
            lat_c = x.shape[1]
            lat_h = x.shape[2]
            lat_w = x.shape[3]
            x = x.view(B, args.num_frames, lat_c, lat_h, lat_w)            
            # Generation task mask each step
            choice_idx = 0
            generator = VideoMaskGenerator((x.shape[-4], x.shape[-2], x.shape[-1]))
            mask = generator(B, device, idx=choice_idx)

            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            loss_dict = diffusion.training_losses(model, x, t, mask=mask)
            loss = loss_dict["loss"].mean()

            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # model_kwargs = dict(y=y)
            # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            # loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps    += 1
            train_steps  += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if rank == 0:
                    wandb.log({"train/loss": avg_loss, "train/steps_per_sec": steps_per_sec}, step=train_steps)
                running_loss = 0
                log_steps    = 0
                start_time   = time()

            if train_steps % args.eval_every == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Running evaluation at step {train_steps}...")
                    ema.eval()
                    evaluate(ema, ae, diffusion, args, device, rank, train_steps, logger)
                    ema.eval()  # keep ema in eval mode
                dist.barrier()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema":   ema.state_dict(),
                        "opt":   opt.state_dict(),
                        "args":  args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()
    logger.info("Done!")
    if rank == 0:
        wandb.finish()
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="../data/OGM-datasets/OGM-Turtlebot2/train/", required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=20)  # 2T: past T + future T
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-data-path", type=str, default="../data/OGM-datasets/OGM-Turtlebot2/val/",
                        help="Path to test dataset. Defaults to --data-path if not set.")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-sampling-steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    args = parser.parse_args()
    main(args)
