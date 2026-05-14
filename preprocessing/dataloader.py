import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.local_occ_grid_map import LocalMap

POINTS = 1080   # number of lidar points
IMG_SIZE = 64
SEQ_LEN = 10

P_prior = 0.5      # Prior occupancy probability
P_occ = 0.7        # Probability that cell is occupied with total confidence
P_free = 0.3       # Probability that cell is free with total confidence
MAP_X_LIMIT = [0.0, 6.4]
MAP_Y_LIMIT = [-3.2, 3.2]
RESOLUTION = 0.1
TRESHOLD_P_OCC = 0.8

NEW_LINE = "\n"


class PredOccDataset(Dataset):
    """
      data_root/
        scans/train.txt, val.txt, ... 
        positions/train.txt, val.txt, ...
    """

    def __init__(self, data_root, split="train"):
        super().__init__()
        self.data_root = data_root
        self.split = split

        self.scan_file_names = []
        self.pos_file_names = []
        self.vel_file_names = []

        fp_scan = open(os.path.join(data_root, "scans", f"{split}.txt"), "r")
        fp_pos = open(os.path.join(data_root, "positions", f"{split}.txt"), "r")
        fp_vel = open(os.path.join(data_root, "velocities", f"{split}.txt"), "r")

        for line in fp_scan.read().split(NEW_LINE):
            if ".npy" in line:
                self.scan_file_names.append(os.path.join(data_root, "scans", line))
        for line in fp_pos.read().split(NEW_LINE):
            if ".npy" in line:
                self.pos_file_names.append(os.path.join(data_root, "positions", line))
        for line in fp_vel.read().split(NEW_LINE):
            if ".npy" in line:
                self.vel_file_names.append(os.path.join(data_root, "velocities", line))


        fp_scan.close()
        fp_pos.close()
        fp_vel.close()

        assert len(self.scan_file_names) == len(self.pos_file_names) == len(self.vel_file_names)
        self.length = len(self.scan_file_names)

        print(f"PredOccDataset({split}) length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        scans = np.zeros((SEQ_LEN + SEQ_LEN, POINTS), dtype=np.float32)
        positions = np.zeros((SEQ_LEN + SEQ_LEN, 3), dtype=np.float32)
        velocities = np.zeros((SEQ_LEN + SEQ_LEN, 2), dtype=np.float32)

        if idx + (SEQ_LEN + SEQ_LEN) < self.length:
            idx_s = idx
        else:
            idx_s = idx - (SEQ_LEN + SEQ_LEN)

        for i in range(SEQ_LEN + SEQ_LEN):
            scan_name = self.scan_file_names[idx_s + i]
            pos_name = self.pos_file_names[idx_s + i]
            vel_name = self.vel_file_names[idx_s + i]

            scans[i] = np.load(scan_name)
            positions[i] = np.load(pos_name)
            velocities[i] = np.load(vel_name)

        scans[np.isnan(scans)] = 20.0
        scans[np.isinf(scans)] = 20.0
        scans[scans == 30] = 20.0

        positions[np.isnan(positions)] = 0.0
        positions[np.isinf(positions)] = 0.0

        velocities[np.isnan(velocities)] = 0.0
        velocities[np.isinf(velocities)] = 0.0

        scan_tensor = torch.from_numpy(scans).float()
        pose_tensor = torch.from_numpy(positions).float()
        vel_tensor = torch.from_numpy(velocities).float()

        data = {
                'scan': scan_tensor,
                'position': pose_tensor,
                'velocity': vel_tensor
                }

        return data