import importlib

import torch
import numpy as np
from collections import abc
from einops import rearrange
from functools import partial

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


import torch
import torch.nn.functional as F
import math

eps = 1e-4

def reprojection(source_map, dx, dy, dtheta, x_lim, y_lim):
    B, C, H, W = source_map.shape
    device, dtype = source_map.device, source_map.dtype

    dx = dx.expand(B)
    dy = dy.expand(B)
    dtheta = dtheta.expand(B)

    x_min, x_max = x_lim
    y_min, y_max = y_lim

    # 1) prob -> logit
    source_logit = torch.logit(source_map.clamp(eps, 1 - eps))

    # 2) target grid centers (t+1 frame) in world coords
    res_x = (x_max - x_min) / H
    res_y = (y_max - y_min) / W

    ix = torch.arange(H, device=device)
    iy = torch.arange(W, device=device)
    ix_grid, iy_grid = torch.meshgrid(ix, iy, indexing='ij')  # (H,W)

    x_tg = x_min + (ix_grid.to(dtype) + 0.5) * res_x
    y_tg = y_min + (iy_grid.to(dtype) + 0.5) * res_y

    x_tg = x_tg.unsqueeze(0).expand(B, -1, -1)  # (B,H,W)
    y_tg = y_tg.unsqueeze(0).expand(B, -1, -1)

    # 3) reprojection : inverse transform to find source sampling locations
    ct = torch.cos(dtheta).view(B, 1, 1).to(dtype)
    st = torch.sin(dtheta).view(B, 1, 1).to(dtype)
    dxv = dx.view(B, 1, 1).to(dtype)
    dyv = dy.view(B, 1, 1).to(dtype)

    x_shift = x_tg + dxv
    y_shift = y_tg + dyv

    x_src =  ct * x_shift - st * y_shift
    y_src =  st * x_shift + ct * y_shift

    # --- 4) world -> normalized coords for grid_sample
    grid_x = 2.0 * (y_src - y_min) / (y_max - y_min) - 1.0  # -> W-axis
    grid_y = 2.0 * (x_src - x_min) / (x_max - x_min) - 1.0  # -> H-axis

    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)

    # valid mask (normalized range)
    valid = (grid_x >= -1.0) & (grid_x <= 1.0) & (grid_y >= -1.0) & (grid_y <= 1.0)
    valid_mask = valid.to(dtype).unsqueeze(1)  # (B,1,H,W)

    # 5) bilinear warp in logit space, then sigmoid back to prob
    warped_logit = F.grid_sample(source_logit, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    warped_prob = torch.sigmoid(warped_logit)

    warped_prob = warped_prob * valid_mask

    return warped_prob, valid_mask



def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res
