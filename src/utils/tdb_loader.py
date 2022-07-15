import tiledb as tdb

import torch
import time
from torch.utils.data import Dataset, DataLoader, RandomSampler

class TDBDset(Dataset):
    def __init__(self, tdb_path, roi):
        n = self.db.size // (6 * 32)
        self.db = self.db.reshape(n, 6 * 32)
        self.start_idx = roi[0][0] * roi[1][0] * roi[2][0]
        self.end_idx = roi[0][1] * roi[1][1] * roi[2][1]

        # self.cubes = list(product(*[list(range(start, stop, tsize[idx])) for idx, (start, stop) in enumerate(roi)]))
        print("start random permuting")
        now = time.time()
        self.idxs = torch.randperm(self.end_idx - self.start_idx, device='cuda').cpu() + self.start_idx
        print(f"permuting time taken {time.time() - now}")

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
