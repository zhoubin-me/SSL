import tiledb as tdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from itertools import product
import time
from tqdm import tqdm
import ctypes

class NPDataset(Dataset):
    def __init__(self, fname,
                 roi=((590, 1190), (2060, 3450), (1060, 1400)),
                 tsize=(16, 16, 16),
                 shape=(4775, 6514, 1876, 64),
                 transform=None):
        self.data = np.memmap(fname, dtype='float16', mode='r', shape=shape)
        madvise = ctypes.CDLL("libc.so.6").madvise
        madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        madvise.restype = ctypes.c_int
        assert madvise(self.data.ctypes.data, self.data.size * self.data.dtype.itemsize, 1) == 0, "MADVISE FAILED"
        # 1 means MADV_RANDOM

        self.tsize = tsize
        self.roi = roi
        self.transform = transform
        now = time.time()
        print("Permutating ...")
        self.cubes = list(product(*[list(range(start, stop, tsize[idx])) for idx, (start, stop) in enumerate(roi)]))
        end = time.time()
        print("Permutation Time taken", end-now)

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, idx):
        start_x, start_y, start_z = self.cubes[idx]
        end_x, end_y, end_z = (x + y for x, y in zip(self.cubes[idx], self.tsize))
        cube_data = self.data[start_x:end_x, start_y:end_y, start_z:end_z, :].copy()
        if self.transform is not None:
            cube_data = self.transform(cube_data)
        return cube_data

def tdb2np():
    fname = "/ssd1/data/namibia/angles/4.00-55.00-64-lin-angles.tdb"
    # roi = ((590, 1190), (2060, 3450), (1060, 1400))
    roi = ((1190, 1790), (2060, 3450), (1000, 1340))
    # roi = ((1790, 2090), (2060, 3450), (950, 1300))
    # roi = ((2090, 2690), (2060, 3450), (900, 1220))
    tsize = (16, 16, 16)
    cubes = list(product(*[list(range(start, stop, tsize[idx])) for idx, (start, stop) in enumerate(roi)]))
    db = tdb.open(fname, 'r')
    shape = db.shape
    print(shape)
    fp = np.memmap("/hdd1/users/bzhou/namibia1.dat", dtype='float16', mode='w+', shape=shape)
    for start in tqdm(cubes):
        stop = tuple(x + y for x, y in zip(start, tsize))
        data = db[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2], :]['a']
        fp[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2], :] = data.astype('float16')
        fp.flush()


if __name__ == '__main__':
    tdb2np()
    # batch_size = 128
    # roi = ((590, 1190), (2060, 3450), (1060, 1400))
    # dset = NPDataset("/hdd1/users/bzhou/namibia1.dat", roi=roi, tsize=(15, 15, 10))
    # loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=16)
    #
    # print(len(loader))
    # now = time.time()
    # print("Start sampling")
    # N = 1000
    # for idx, x in enumerate(loader):
    #     if idx > N:
    #         break
    # print(f"Average time taken for each batch of {N} batches with batch size {batch_size}:")
    # print((time.time() - now) / N)
