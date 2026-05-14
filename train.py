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
from time import time, perf_counter
import argparse
import logging
import os

import wandb
from torchvision.utils import make_grid
from tqdm import tqdm

from models.models import VDT_models
from models.diffusion import create_diffusion
from preprocessing.dataloader import PredOccDataset
from preprocessing.data_preprocessing import preprocess_batch
from models.convlstm import ConvLSTMCell
from omegaconf import OmegaConf
from utils.util import instantiate_from_config 

def make_video(batch_out):
    """Convert preprocessed maps to a 1-channel video tensor: (B, 2T, 1, H, W).
    First T frames: past (input), next T frames: future (target to predict).
    """
    input_binary_maps  = batch_out["input_binary_maps"].float()   # (B, T, 1, H, W) - past
    mask_binary_maps   = batch_out["mask_binary_maps"].float()    # (B, T, 1, H, W) - future

    return torch.cat([input_binary_maps, mask_binary_maps], dim=1)  # (B, 2T, 1, H, W)


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=in_channels,
                                                  out_channels=num_hiddens//2,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens//2),
                                        nn.ReLU()
                                    ])
        self._conv_2 = nn.Sequential(*[
                                        nn.Conv2d(in_channels=num_hiddens//2,
                                                  out_channels=num_hiddens,
                                                  kernel_size=4,
                                                  stride=2, 
                                                  padding=1),
                                        nn.BatchNorm2d(num_hiddens)
                                        #nn.ReLU()
                                    ])
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._conv_2(x)
        x = self._residual_stack(x)
        return x 


class CondEncoder(nn.Module):
    """Compress past latent sequence into spatial conditioning map using ConvLSTM."""
    
    def __init__(self, input_dim=1, hidden_dim=32):
        super().__init__()
        self.convlstm = ConvLSTMCell(input_dim, hidden_dim, kernel_size=(3, 3), bias=True)
        
        # Use Encoder class for better feature extraction
        self.encoder = Encoder(
            in_channels=33,
            num_hiddens=128,
            num_residual_layers=2,
            num_residual_hiddens=64
        )
        
        # Final projection: (128, 16, 16) → (32, 16, 16)
        self.cond_proj = nn.Conv2d(128, hidden_dim, kernel_size=1)
    
    def forward(self, past_latents, input_occ_grid=None):
        """
        Args:
            past_latents: (B, T, 1, H, W) - past binary maps (original image size)
            input_occ_grid: (B, 1, H, W) - input occupancy grid map
        Returns:
            cond: (B, 32, 16, 16) - conditioning feature map
        """
        B, T, C, H, W = past_latents.shape
        
        # Initialize ConvLSTM hidden state at original image size
        h_enc, c_enc = self.convlstm.init_hidden(B, (H, W))
        
        # Process past frames through ConvLSTM
        for t in range(T):
            frame = past_latents[:, t]  # (B, 1, H, W)
            h_enc, c_enc = self.convlstm(frame, (h_enc, c_enc))
        
        # h_enc: (B, hidden_dim, H, W) = (B, 32, 64, 64)
        # Combine with input occupancy grid
        if input_occ_grid is not None:
            cond_in = torch.cat([h_enc, input_occ_grid], dim=1)  # (B, 33, H, W)
        else:
            cond_in = h_enc  # (B, 32, H, W)
        
        # Encoder: downsample to latent space size
        cond_feat = self.encoder(cond_in)  # (B, 128, 16, 16)
        
        # Final projection
        cond = self.cond_proj(cond_feat)  # (B, 32, 16, 16)
        
        return cond


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
def evaluate(ema, ae, cond_encoder, diffusion, args, device, rank, epoch, logger):
    """Compute validation loss at end of epoch."""

    test_dataset = PredOccDataset(data_root=args.eval_data_path, split="val")
    test_loader  = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    batch = next(iter(test_loader))

    batch_out = preprocess_batch(batch, device=device)
    past_maps = batch_out["input_binary_maps"].float()      # (B, T, 1, H, W)
    future_maps = batch_out["mask_binary_maps"].float()     # (B, T, 1, H, W)
    B, TT, C, H, W = past_maps.shape

    # Get input occupancy grid
    input_occ_grid = past_maps[:, 0, :, :, :]  # (B, 1, H, W)

    # Encode future
    posterior_future = ae.encode(future_maps)
    future_latents = posterior_future.mode().mul_(args.scale_factor)

    lat_c = future_latents.shape[1]

    # Generate conditioning map
    cond = cond_encoder(past_maps, input_occ_grid)

    # Compute loss on future frames
    t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
    val_loss = diffusion.training_losses(ema, future_latents, t, model_kwargs={"cond": cond})["loss"].mean()

    wandb.log({"val/loss": val_loss.item()}, step=epoch)
    logger.info(f"[Epoch {epoch}] Validation Loss: {val_loss.item():.6f}")


@torch.no_grad()
def log_images(ema, ae, cond_encoder, diffusion, args, device, rank, train_steps, logger):
    """Log IoU + image at training steps."""

    T_half = args.num_frames // 2  # 10
    latent_size = args.image_size // 4
    eval_diffusion = create_diffusion(str(args.eval_sampling_steps))

    test_dataset = PredOccDataset(data_root=args.eval_data_path, split="val")
    test_loader  = DataLoader(test_dataset, batch_size=args.eval_batch_size,
                              shuffle=True, num_workers=2, drop_last=True)
    batch = next(iter(test_loader))

    # Start timing
    t_start = perf_counter()

    batch_out = preprocess_batch(batch, device=device)
    past_maps = batch_out["input_binary_maps"].float()      # (B, T, 1, H, W)
    future_maps = batch_out["mask_binary_maps"].float()     # (B, T, 1, H, W)
    B, TT, C, H, W = past_maps.shape

    # Get input occupancy grid
    input_occ_grid = past_maps[:, 0, :, :, :]  # (B, 1, H, W)

    # Encode future
    posterior_future = ae.encode(future_maps)
    future_latents = posterior_future.mode().mul_(args.scale_factor)  # (B, T, lat_c, lat_h, lat_w)
    
    lat_c = future_latents.shape[1]
    
    # Generate conditioning map from past
    cond = cond_encoder(past_maps, input_occ_grid)  # (B, 32, lat_h, lat_w)
    
    # Generate samples
    z_noise = torch.randn_like(future_latents)
    z_perm = z_noise.permute(0, 2, 1, 3, 4)  # (B, lat_c, T, lat_h, lat_w)
    
    # Repeat cond for all timesteps
    cond_t = cond.unsqueeze(1).repeat(1, T_half, 1, 1, 1)  # (B, T, 32, lat_h, lat_w)
    cond_perm = cond_t.permute(0, 2, 1, 3, 4)  # (B, 32, T, lat_h, lat_w)

    samples = eval_diffusion.p_sample_loop(
        ema.forward, z_perm.shape, z_perm,
        clip_denoised=False, progress=False, device=device,
        model_kwargs={"cond": cond_perm}
    )
    samples = samples.permute(1, 2, 0, 3, 4)  # (B, T, lat_c, lat_h, lat_w)
    samples_flat = samples.reshape(-1, lat_c, latent_size, latent_size)
    samples_flat = 1. / args.scale_factor * samples_flat 

    # Decode
    decoded = ae.decode(samples_flat)
    samples = decoded.reshape(B, args.num_frames, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

    # End timing
    t_end = perf_counter()
    inference_time = t_end - t_start

    # Compute frame-wise IoU
    iou_list = []
    for ti in range(T_half):
        iou_t = compute_batch_iou(samples[:, ti:ti+1], future_maps[:, ti:ti+1], threshold=0.1)
        iou_list.append(iou_t.item())

    # Visualize: past | future | predicted
    x_video = torch.cat([past_maps, future_maps], dim=1)  # (B, 2T, 1, H, W)
    samples_vis = torch.cat([x_video, samples], dim=1)  # (B, 3T, 1, H, W)
    vis = samples_vis[0].cpu().clamp(0, 1)
    grid = make_grid(vis, nrow=args.num_frames, normalize=False, value_range=(0, 1))
    
    iou_text = "  ".join([f"t{ti+1}:{iou_list[ti]:.3f}" for ti in range(len(iou_list))])
    
    wandb.log({
        "train/image": wandb.Image(
            grid,
            caption=f"Frame-wise IoU | {iou_text}\nstep={train_steps}"
        ),
        "train/inference_time_sec": inference_time
    }, step=train_steps)
    logger.info(
        f"(step={train_steps:07d}) Image logged. "
        f"inference_time={inference_time:.2f}s, {iou_text}"
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

    # Initialize CondEncoder
    cond_encoder = CondEncoder(input_dim=1, hidden_dim=32)
    cond_encoder = cond_encoder.to(device)
    
    logger.info(f"VDT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(list(model.parameters()) + list(cond_encoder.parameters()), 
                            lr=1e-4, weight_decay=0)

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

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in tqdm(range(args.epochs), desc="Epoch", disable=(rank != 0)):
        sampler.set_epoch(epoch)
        for batch in tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0), leave=False):
            # Preprocess batch
            batch_out = preprocess_batch(batch, device=device)
            past_maps = batch_out["input_binary_maps"].float()      # (B, T, 1, H, W) - original size
            future_maps = batch_out["mask_binary_maps"].float()     # (B, T, 1, H, W)
            B, T, C, H, W = past_maps.shape

            # Get input occupancy grid (first past frame)
            input_occ_grid = batch_out["input_occ_grid_map"]  # (B, H, W)
            input_occ_grid = input_occ_grid.unsqueeze(1)  # (B, 1, H, W)

            with torch.no_grad():
                # Encode future frames for diffusion
                posterior_future = ae.encode(future_maps)
                future_latents = posterior_future.mode().mul_(args.scale_factor)  # (B, T, lat_c, lat_h, lat_w)

            lat_c = future_latents.shape[1]
            lat_h = future_latents.shape[2]
            lat_w = future_latents.shape[3]

            # Generate conditioning map from past at original image size
            cond = cond_encoder(past_maps, input_occ_grid)  # (B, 32, lat_h, lat_w)

            # Diffusion loss on future frames only (no mask)
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)
            loss_dict = diffusion.training_losses(model, future_latents, t, model_kwargs={"cond": cond})
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            running_loss += loss.item()
            log_steps    += 1
            train_steps  += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                if rank == 0:
                    wandb.log({"train/loss": avg_loss}, step=train_steps)
                running_loss = 0
                log_steps    = 0

            if train_steps % 100 == 0 and train_steps > 0:
                if rank == 0:
                    logger.info(f"Logging images at step {train_steps}...")
                    ema.eval()
                    log_images(ema, ae, cond_encoder, diffusion, args, device, rank, train_steps, logger)
                    ema.train()
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

        # End of epoch: evaluate loss only
        if rank == 0:
            logger.info(f"Running epoch validation at end of epoch {epoch}...")
            ema.eval()
            evaluate(ema, ae, cond_encoder, diffusion, args, device, rank, epoch, logger)
            ema.train()
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
    parser.add_argument("--scale-factor", type=float, default=0.374106)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    args = parser.parse_args()
    main(args)
