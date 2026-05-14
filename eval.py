#!/usr/bin/env python
"""
VDT evaluation script.
Evaluates a trained VDT checkpoint on test data.
Computes per-frame IoU, inference time, and saves GT/predicted visualizations + CSV.
"""

import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_mutual_information as nmi

from models import VDT_models
from models.diffusion import create_diffusion
from models.mask_generator import VideoMaskGenerator
from preprocessing.dataloader import PredOccDataset
from preprocessing.data_preprocessing import preprocess_batch_test
from omegaconf import OmegaConf

from utils.util import instantiate_from_config, reprojection

# Constants
SEQ_LEN = 10
IMG_SIZE = 64
MAP_X_LIMIT = [0.0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
IOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07, 0.08, 0.09)
all_rows_by_thr = {occ_thr: [] for occ_thr in IOU_THRESHOLDS}
all_ssim_rows = []
all_psnr_rows = []
all_nmi_rows = []

def compute_iou(pred, gt, occ_thr=0.8):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    inter = (pred_occ & gt_occ).sum().float()
    union = (pred_occ | gt_occ).sum().float()

    iou = inter / (union + 1e-6)

    return iou

def compute_miou(pred, gt, occ_thr=0.8):
    pred_occ = (pred > occ_thr)
    gt_occ   = (gt > occ_thr)

    pred_free = ~pred_occ
    gt_free   = ~gt_occ

    inter_occ = (pred_occ & gt_occ).sum().float()
    union_occ = (pred_occ | gt_occ).sum().float()
    iou_occ = inter_occ / (union_occ + 1e-6)

    inter_free = (pred_free & gt_free).sum().float()
    union_free = (pred_free | gt_free).sum().float()
    iou_free = inter_free / (union_free + 1e-6)

    miou = (iou_occ + iou_free) / 2.0

    return miou

def compute_ssim_metric(pred, gt):
    """Compute SSIM between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        ssim score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute SSIM (data_range should be 1.0 for normalized values)
    score = ssim(pred_np, gt_np, data_range=1.0, channel_axis=0)
    
    return float(score)


def compute_psnr_metric(pred, gt):
    """Compute PSNR between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        psnr score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute PSNR (data_range should be 1.0 for normalized values)
    score = psnr(gt_np, pred_np, data_range=1.0)
    
    return float(score)


def compute_nmi_metric(pred, gt):
    """Compute NMI between prediction and ground truth.
    
    Args:
        pred: prediction tensor (C, H, W)
        gt: ground truth tensor (C, H, W)
    
    Returns:
        nmi score (float)
    """
    pred_np = pred.detach().cpu().numpy().astype(np.float32)
    gt_np = gt.detach().cpu().numpy().astype(np.float32)
    
    # Compute NMI
    score = nmi(gt_np, pred_np)
    
    return float(score)


def make_video(batch_out):
    """Convert preprocessed maps to a 1-channel video tensor: (B, 2T, 1, H, W)."""
    input_binary_maps  = batch_out["input_binary_maps"].float()   # (B, T, 1, H, W)
    mask_binary_maps   = batch_out["mask_binary_maps"].float()    # (B, T, 1, H, W)
    return torch.cat([input_binary_maps, mask_binary_maps], dim=1)  # (B, 2T, 1, H, W)

def reproject_prediction_maps(prediction, x_rel, y_rel, th_rel):
    """Reproject predicted occupancy maps into the future robot reference frame."""
    prediction_maps = torch.zeros(
        prediction.shape[1], 1, IMG_SIZE, IMG_SIZE,
        device=prediction.device,
        dtype=prediction.dtype,
    )

    for k in range(prediction.shape[1]):
        prediction_t, _ = reprojection(
            prediction[:, k],
            x_rel[:, k],
            y_rel[:, k],
            th_rel[:, k],
            MAP_X_LIMIT,
            MAP_Y_LIMIT,
        )
        prediction_t = prediction_t.reshape(-1, 1, 1, IMG_SIZE, IMG_SIZE)
        predictions = prediction_t.squeeze(1)
        pred_mean = torch.mean(predictions, dim=0, keepdim=True)
        prediction_maps[k, 0] = pred_mean.squeeze()

    return prediction_maps.unsqueeze(0)


def save_prediction_overlay_gif(prediction_maps, gt_binary, output_path, frame_duration_ms=100, scale=8):
    """Save a GIF of predicted maps with GT occupied cells highlighted in red."""
    frames = []

    for frame_idx in range(prediction_maps.shape[0]):
        pred_map = prediction_maps[frame_idx, 0].detach().cpu().clamp(0, 1).numpy()
        gt_map = gt_binary[frame_idx, 0].detach().cpu().numpy() > 0.5

        pred_uint8 = (pred_map * 255).astype(np.uint8)
        rgb_frame = np.stack([pred_uint8, pred_uint8, pred_uint8], axis=-1)
        rgb_frame[gt_map] = np.array([255, 0, 0], dtype=np.uint8)

        pil_frame = Image.fromarray(rgb_frame, mode="RGB")
        if scale != 1:
            pil_frame = pil_frame.resize((IMG_SIZE * scale, IMG_SIZE * scale), Image.Resampling.NEAREST)
        frames.append(pil_frame)

    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration_ms,
            loop=0,
        )

def save_iou_tables(all_rows_by_thr, output_dir):
    for occ_thr, rows in all_rows_by_thr.items():
        csv_path = os.path.join(output_dir, f"eval_table_iou_{occ_thr:.2f}.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_ssim_table(all_ssim_rows, output_dir):
    """Save SSIM metrics to CSV file."""
    if all_ssim_rows:
        csv_path = os.path.join(output_dir, "eval_table_ssim.csv")
        pd.DataFrame(all_ssim_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_psnr_table(all_psnr_rows, output_dir):
    """Save PSNR metrics to CSV file."""
    if all_psnr_rows:
        csv_path = os.path.join(output_dir, "eval_table_psnr.csv")
        pd.DataFrame(all_psnr_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

def save_nmi_table(all_nmi_rows, output_dir):
    """Save NMI metrics to CSV file."""
    if all_nmi_rows:
        csv_path = os.path.join(output_dir, "eval_table_nmi.csv")
        pd.DataFrame(all_nmi_rows).to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")



def load_model(ckpt_path, device, image_size=64, num_frames=20, model_name="VDT-L/2",
               num_classes=1, ae_config_path="ae_vdt/ae_vdt.yaml",
               ae_ckpt_path="ae_vdt/model.ckpt"):
    ae_cfg = OmegaConf.load(ae_config_path)
    latent_channels = ae_cfg.model.params.embed_dim
    latent_size = image_size // 4
    model = VDT_models[model_name](
        input_size=latent_size,
        num_classes=num_classes,
        in_channels=latent_channels,
        num_frames=num_frames,
        mode='video',
    )

    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Load EMA weights (preferred for eval)
    if "ema" in ckpt:
        model.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights.")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights (no EMA found).")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw state_dict.")

    model = model.to(device).eval()

    ae = instantiate_from_config(ae_cfg.model)
    ae.init_from_ckpt(ae_ckpt_path)
    ae = ae.to(device).eval()
    for parameter in ae.parameters():
        parameter.requires_grad = False

    return model, ae


@torch.no_grad()
def evaluate(model, ae, dataloader, device, output_dir, sampling_steps=10,
             occ_thr=0.3, num_frames=20, image_size=64, save_images=True,
             num_batches=None):
    os.makedirs(output_dir, exist_ok=True)

    T_half = num_frames // 2  # 10
    latent_size = image_size // 4
    latent_channels = ae.embed_dim
    eval_diffusion = create_diffusion(str(sampling_steps))

    results = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", total=num_batches)):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        batch_out = preprocess_batch_test(batch, device=device)
        x_video   = make_video(batch_out)            # (B, 2T, 3, H, W) in [0,1]
        gt_binary = batch_out["mask_binary_maps"].float()  # (B, T, 1, H, W) GT future
        x_rel = batch_out["x_rel"]
        y_rel = batch_out["y_rel"]
        th_rel = batch_out["th_rel"]

        B, TT, C, H, W = x_video.shape

        # Encode
        posterior = ae.encode(x_video)
        z_frames = posterior.mode()
        z_frames = z_frames.view(B, num_frames, latent_channels, z_frames.shape[-2], z_frames.shape[-1])
        z_noise = torch.randn(B, num_frames, latent_channels, latent_size, latent_size, device=device)

        # Predict mask
        generator = VideoMaskGenerator((z_frames.shape[-4], z_frames.shape[-2], z_frames.shape[-1]))
        mask = generator(B, device, idx=0)

        z_perm = z_noise.permute(0, 2, 1, 3, 4)

        samples = eval_diffusion.p_sample_loop(
            model.forward, z_perm.shape, z_perm,
            clip_denoised=False, progress=False, device=device,
            raw_x=z_frames, mask=mask,
        )

        # Merge known + generated
        samples = samples.permute(1, 0, 2, 3, 4) * mask + z_frames.permute(2, 0, 1, 3, 4) * (1 - mask)
        samples = samples.permute(1, 2, 0, 3, 4)
        samples_flat = samples.reshape(-1, latent_channels, latent_size, latent_size)

        # Decode
        decoded = ae.decode(samples_flat)
        decoded = decoded.view(B, num_frames, 1, H, W)  # (B, 2T, 1, H, W) 

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        total_time_ms = (t1 - t0) * 1000

        # Extract predicted future frames (last T_half)
        pred_future = decoded[:, T_half:, :, :, :]  # (B, T, 1, H, W)
        prediction_maps = reproject_prediction_maps(pred_future, x_rel, y_rel, th_rel)

        # Compute SSIM metrics
        ssim_row = {
            "i": int(batch_idx),
        }
        for n in range(SEQ_LEN):
                gt_map = gt_binary[0, n]
                pred_map = prediction_maps[n]
                ssim_value = compute_ssim_metric(pred_map, gt_map)
                ssim_row[f"n={n+1}"] = ssim_value
        all_ssim_rows.append(ssim_row)

        psnr_row = {
                "i": int(batch_idx),
            }
        for n in range(SEQ_LEN):
                gt_map = gt_binary[0, n]
                pred_map = prediction_maps[n]
                psnr_value = compute_psnr_metric(pred_map, gt_map)
                psnr_row[f"n={n+1}"] = psnr_value
        all_psnr_rows.append(psnr_row)

        nmi_row = {
                "i": int(batch_idx),
            }
        for n in range(SEQ_LEN):
                gt_map = gt_binary[0, n]
                pred_map = prediction_maps[n]
                nmi_value = compute_nmi_metric(pred_map, gt_map)
                nmi_row[f"n={n+1}"] = nmi_value
        all_nmi_rows.append(nmi_row)

        for occ_thr in IOU_THRESHOLDS:
                row = {
                    "i": int(batch_idx),
                    "Inference_time": float(total_time_ms),
                    "occ_thr": float(occ_thr),
                }
                for n in range(SEQ_LEN):
                    gt_map = gt_binary[0, n]
                    pred_map = prediction_maps[n]
                    iou_value = float(compute_iou(pred_map, gt_map, occ_thr=occ_thr).item())
                    row[f"n={n+1}"] = iou_value
                all_rows_by_thr[occ_thr].append(row)

        if (batch_idx + 1) % 1000 == 0:
            save_iou_tables(all_rows_by_thr, output_dir)
            save_ssim_table(all_ssim_rows, output_dir)
            save_psnr_table(all_psnr_rows, output_dir)
            save_nmi_table(all_nmi_rows, output_dir)


        # Save images
        if save_images and batch_idx < 500:
            # GT future occupancy maps
            fig = plt.figure(figsize=(8, 1))
            for m in range(T_half):
                ax = fig.add_subplot(1, T_half, m + 1)
                gt_map = gt_binary[0, m, 0].detach().cpu().numpy()
                ax.imshow(gt_map, cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"n={m+1}", fontsize=8)
            fig.savefig(os.path.join(output_dir, f"gt_{batch_idx}.png"), dpi=500, bbox_inches='tight')
            plt.close(fig)

            # Predicted future occupancy maps
            fig = plt.figure(figsize=(8, 1))
            for m in range(T_half):
                ax = fig.add_subplot(1, T_half, m + 1)
                pred_map = prediction_maps[0, m, 0].detach().cpu().numpy()
                ax.imshow(pred_map, cmap='gray', vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"n={m+1}", fontsize=8)
            fig.savefig(os.path.join(output_dir, f"pred_{batch_idx}.png"), dpi=500, bbox_inches='tight')
            plt.close(fig)

            save_prediction_overlay_gif(
                prediction_maps[0],
                gt_binary[0],
                os.path.join(output_dir, f"pred_overlay_{batch_idx}.gif"),
                frame_duration_ms=100,
            )

        # Periodic CSV save
        if (batch_idx + 1) % 1000 == 0:
            pd.DataFrame(results).to_csv(os.path.join(output_dir, "eval_table.csv"), index=False)

    # Final save
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "eval_table.csv")
    df.to_csv(csv_path, index=False)

    # Summary
    iou_cols = [c for c in df.columns if c.startswith("n=")]
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(df)}")
    print(f"Avg inference time: {df['Inference_time'].mean():.1f} ms")
    for c in iou_cols:
        print(f"  {c}  IoU: {df[c].mean():.4f}")
    print(f"Results saved to {csv_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VDT Evaluation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to VDT checkpoint (.pt)")
    parser.add_argument("--data-path", type=str, default="../data/OGM-datasets/OGM-Turtlebot2/test/",
                        help="Path to test dataset")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--model", type=str, choices=list(VDT_models.keys()), default="VDT-L/2")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--ae-config", type=str, default="ae_vdt/ae_vdt.yaml")
    parser.add_argument("--ae-ckpt", type=str, default="ae_vdt/model.ckpt")
    parser.add_argument("--sampling-steps", type=int, default=10, help="Diffusion sampling steps")
    parser.add_argument("--occ-thr", type=float, default=0.3, help="Occupancy threshold for IoU")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=None, help="Limit number of batches (None=all)")
    parser.add_argument("--no-images", action="store_true", help="Skip saving visualizations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model, ae = load_model(
        args.ckpt, device,
        image_size=args.image_size,
        num_frames=args.num_frames,
        model_name=args.model,
        num_classes=args.num_classes,
        ae_config_path=args.ae_config,
        ae_ckpt_path=args.ae_ckpt,
    )

    # Load test data
    test_dataset = PredOccDataset(data_root=args.data_path, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, drop_last=True)
    print(f"Test dataset: {len(test_dataset)} samples")

    # Evaluate
    df = evaluate(
        model, ae, test_loader, device, args.output_dir,
        sampling_steps=args.sampling_steps,
        occ_thr=args.occ_thr,
        num_frames=args.num_frames,
        image_size=args.image_size,
        save_images=not args.no_images,
        num_batches=args.num_batches,
    )
