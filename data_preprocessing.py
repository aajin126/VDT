import numpy as np
import torch
from torch.utils.data import DataLoader

from local_occ_grid_map import LocalMap
from dataloader import PredOccDataset

POINTS = 1080   # number of lidar points
IMG_SIZE = 64
SEQ_LEN = 10

P_prior = 0.5      # Prior occupancy probability
P_occ = 0.7        # Probability that cell is occupied with total confidence
P_free = 0.3       # Probability that cell is free with total confidence
MAP_X_LIMIT = [0.0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1
TRESHOLD_P_OCC = 0.85


def preprocess_batch(batch, device=None):
    """
    yields:
        batch_out: dict
            'input_binary_maps' : (B, SEQ_LEN, 1, H, W)
            'mask_binary_maps'  : (B, SEQ_LEN, 1, H, W)
            'input_occ_grid_map': (B, H, W)
    """

    scans = batch["scan"]
    positions = batch["position"]

    if device is not None:
        scans = scans.to(device)
        positions = positions.to(device)

    B = scans.size(0)

    # future maps (target)
    mask_gridMap = LocalMap(X_lim=MAP_X_LIMIT,
                            Y_lim=MAP_Y_LIMIT,
                            resolution=RESOLUTION,
                            p=P_prior,
                            size=[B, SEQ_LEN],
                            device=device)

    obs_pos_N = positions[:, SEQ_LEN - 1] # (B,3)
    future_poses = positions[:, SEQ_LEN:] # (B,SEQ_LEN,3)

    x_future_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)
    y_future_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)
    theta_future_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)

    x_future_odom, y_future_odom, theta_future_odom = mask_gridMap.robot_coordinate_transform(
        future_poses, obs_pos_N
    )

    future_distances = scans[:, SEQ_LEN:] # get future 10 frames
    future_angles = torch.linspace(-(135 * np.pi / 180), 135 * np.pi / 180, future_distances.shape[-1], device=scans.device)

    future_distances_x, future_distances_y = mask_gridMap.lidar_scan_xy(
        future_distances, future_angles, x_future_odom, y_future_odom, theta_future_odom
    )

    mask_binary_maps = mask_gridMap.discretize(future_distances_x, future_distances_y)  # (B,SEQ_LEN,H,W)

    # history maps (input)
    input_gridMap = LocalMap(X_lim=MAP_X_LIMIT,
                            Y_lim=MAP_Y_LIMIT,
                            resolution=RESOLUTION,
                            p=P_prior,
                            size=[B, SEQ_LEN],
                            device=device)

    x_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)
    y_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)
    theta_odom = torch.zeros(B, SEQ_LEN, device=device or scans.device)

    pos = positions[:, :SEQ_LEN]
    x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, obs_pos_N)

    distances = scans[:, :SEQ_LEN]
    angles = torch.linspace(
        -(135 * np.pi / 180), 135 * np.pi / 180, distances.shape[-1], device=scans.device
    )

    distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        
    # discretize to binary maps:
    input_binary_maps = input_gridMap.discretize(distances_x, distances_y)  # (B,SEQ_LEN,H,W)
    input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
        
    #static map
    input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)     # (B,H,W)

    # add channel dimension:
    input_binary_maps = input_binary_maps.unsqueeze(2)
    mask_binary_maps = mask_binary_maps.unsqueeze(2)

    batch_out = {
        "input_binary_maps": input_binary_maps,   # (B,SEQ_LEN,1,H,W)
        "mask_binary_maps": mask_binary_maps,     # (B,SEQ_LEN,1,H,W)
        "input_occ_grid_map": input_occ_grid_map  # (B,H,W)
    }

    return batch_out

def preprocess_batch_test(batch, device=None):
    """
    yields:
        batch_out: dict
            'input_binary_maps' : (B, SEQ_LEN, 1, H, W)
            'mask_binary_maps'  : (B, SEQ_LEN, 1, H, W)
            'input_occ_grid_map': (B, H, W)
    """

    scans = batch["scan"]
    positions = batch["position"]

    if device is not None:
        scans = scans.to(device)
        positions = positions.to(device)

    B = scans.size(0)

    mask_gridMap = LocalMap(X_lim = MAP_X_LIMIT, 
                    Y_lim = MAP_Y_LIMIT, 
                    resolution = RESOLUTION, 
                    p = P_prior,
                    size=[B, SEQ_LEN],
                    device = device)
    # robot positions:
    x_odom = torch.zeros(B, SEQ_LEN).to(device)
    y_odom = torch.zeros(B, SEQ_LEN).to(device)
    theta_odom = torch.zeros(B, SEQ_LEN).to(device)
    # Lidar measurements:
    distances = scans[:,SEQ_LEN:]
    # the angles of lidar scan: -135 ~ 135 degree
    angles = torch.linspace(-(135*np.pi/180), 135*np.pi/180, distances.shape[-1]).to(device)
    # Lidar measurements in X-Y plane: transform to the predicted robot reference frame
    distances_x, distances_y = mask_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
    # discretize to binary maps:
    mask_binary_maps = mask_gridMap.discretize(distances_x, distances_y)
    mask_binary_maps = mask_binary_maps.unsqueeze(2)

    # current position:
    obs_pos_N = positions[:, SEQ_LEN-1]
    # calculate relative future positions to current position:
    future_poses = positions[:, SEQ_LEN:] 
    x_rel, y_rel, th_rel = mask_gridMap.robot_coordinate_transform(future_poses, obs_pos_N)

    # history maps (input)
    input_gridMap = LocalMap(X_lim=MAP_X_LIMIT,
                            Y_lim=MAP_Y_LIMIT,
                            resolution=RESOLUTION,
                            p=P_prior,
                            size=[B, SEQ_LEN],
                            device=device)

    pos = positions[:, :SEQ_LEN]

    x_odom, y_odom, theta_odom = input_gridMap.robot_coordinate_transform(pos, obs_pos_N)

    distances = scans[:, :SEQ_LEN]
    angles = torch.linspace(
        -(135 * np.pi / 180), 135 * np.pi / 180, distances.shape[-1], device=scans.device
    )

    distances_x, distances_y = input_gridMap.lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)
        
    # discretize to binary maps:
    input_binary_maps = input_gridMap.discretize(distances_x, distances_y)  # (B,SEQ_LEN,H,W)
    input_gridMap.update(x_odom, y_odom, distances_x, distances_y, P_free, P_occ)
        
    #static map
    input_occ_grid_map = input_gridMap.to_prob_occ_map(TRESHOLD_P_OCC)     # (B,H,W)

    # add channel dimension:
    input_binary_maps = input_binary_maps.unsqueeze(2)


    batch_out = {
        "input_binary_maps": input_binary_maps,   # (B,SEQ_LEN,1,H,W)
        "mask_binary_maps": mask_binary_maps,     # (B,SEQ_LEN,1,H,W)
        "input_occ_grid_map": input_occ_grid_map,  # (B,H,W)
        "x_rel": x_rel,                           # (B,SEQ_LEN)
        "y_rel": y_rel,                           # (B,SEQ_LEN)
        "th_rel": th_rel                          # (B,SEQ_LEN)
    }
    return batch_out