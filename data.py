import os
import netCDF4
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, default_collate
from tools import progress

"""
Contains Methods for Data Preprocessing and Loading.
Droplet distribution simulation data provided by Andrea Jenny, available on PSC's Bridges-2.

drop_distr_subset.nc - [bins(33), time(2), X(512), Y(512), Z(150)]
    - bin counts?, netCDF4, ~10GB
atex/drop_distr_{%2d}d_{%2d}h_{%2d}m_{%2d}s.nc - [bins(33), time(1), X(640), Y(640), Z(75)]
    - mixing ratio, netCDF4, ~196GB (49 * ~4B) (* 6+ metrological cases, e.g. 'atex', 'dycoms')

-> drop_distr_{%2d}.data - [bins(33), position(?<640*640*75)]
    - non-zero mixing ratio, HDF5, ~17.2GB (48 * ~300MB)
"""


DATA_PATH = 'data/drop_distr_subset.nc'
DATA_FOLDER_RAW = 'data/atex/raw'
DATA_FOLDER = 'data/atex/nz'
DATA_ATEX_ALL = ['data/atex', 'data/atex_0.5x', 'data/atex_2.0x']
DATA_DYCOMS_ALL = ['data/dycoms', 'data/dycoms_0.5x', 'data/dycoms_2.0x']


def min_max_normalized(t, l=0, u=1, mn=None, mx=None):
    """
    Normalize data to be in interval [l, u]
    """
    mn = torch.min(t) if mn is None else mn
    mx = torch.max(t) if mx is None else mx
    return l + u * (np.clip(t, mn, mx) - mn) / (mx - mn)


def nc2json(source_path, target_path):
    """
    Read netCDF4 and write to JSON (e.g. for use with STAN)

    If netCDF4 is 4.95 GB per time step (binary), JSON is 200-250 GB per time step (UTF-8, non-binary).
    The latter is 40-50 times bigger and thus prohibitively slow.
    """
    with open(target_path, "w") as tgt:
        with netCDF4.Dataset(source_path, "r") as src:
            data = src.variables["ff1i1"]
            data.set_auto_mask(False)
            # bins(33), time(2), X(512), Y(512), Z(150)
            B, T, X, Y, Z = data.shape
            # subset
            T, Z = 1, 1
            X, Y, B = 30, 30, 3
            json.dump({
                "K": 2,
                "B": B,
                "T": T,
                "N_X": X,
                "N_Y": Y,
                "N_Z": Z,
                "y": data[:B, :T, :X, :Y, :Z].tolist()
            }, tgt)


def nc2hdf5(source_path, target_path, process=True, eps=0.00001, plot=False):
    """
    Read netCDF4 and write to HDF5 (same functionality, better Python integration)

    :param process - (optionally) discard locations near-zero water mass higher than the threshold <eps>,
                     This flattens data, so we include location information.

    0.14% > 0.0001
    0.23% > 0.00001
    1.7% > 0

    """
    with h5py.File(target_path, "w") as tgt:
        with netCDF4.Dataset(source_path, "r") as src:
            # data = src.variables["ff1i1"]
            data = src.variables["drop_distr"]
            data.set_auto_mask(False)
            B, T, X, Y, Z = data.shape
            d = data[:]

            if plot:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ds = d.sum(axis=0)
                ds = ds[ds > 0]
                hist, bins, _ = ax[0].hist(ds, bins=20)
                logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
                ax[1].hist(ds, bins=logbins)
                ax[1].set_xscale('log')
                lm = ax[1].get_ylim()
                ax[1].vlines(0.00001, *lm, color='red')
                ax[1].set_ylim(lm)
                plt.show()

            if process:
                x_ = np.arange(0, X)
                y_ = np.arange(0, Y)
                z_ = np.arange(0, Z)
                x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
                mask = d.sum(axis=0) > eps

                d = d[:, mask]
                if d.size == 0:
                    tgt.close()
                    os.remove(target_path)
                    return

                mask = mask.squeeze(0)
                x, y, z = x[mask], y[mask], z[mask]
                tgt.create_dataset("x", data=x, dtype=np.float32)
                tgt.create_dataset("y", data=y, dtype=np.float32)
                tgt.create_dataset("z", data=z, dtype=np.float32)
            else:
                d = d.reshape((B, -1))
            tgt.create_dataset("data", data=d, dtype=np.float32)


class FastSampler(Sampler):
    """
    Faster Batch Sampling for HDF5.
    Allows for weak shuffling.
    """
    def __init__(self, dataset_length, batch_size, weak_shuffle=False):
        self.batch_size = batch_size
        self.dataset_length = dataset_length
        self.n_batches = int(np.ceil(self.dataset_length / self.batch_size))
        self.batch_id_perm = torch.randperm(self.n_batches) if weak_shuffle else torch.arange(self.n_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for ths_i in range(self.n_batches):
            i = int(self.batch_id_perm[ths_i])
            yield np.s_[i * self.batch_size: min((i+1) * self.batch_size, self.dataset_length)]


class PlotSampler(Sampler):
    """
    Fast Time-Sequential Batch Sampling for HDF5.
    """
    def __init__(self, dataset_lengths, batch_size, t_plot=None):
        self.batch_size = batch_size
        self.dataset_lengths = dataset_lengths
        self.anchors = np.cumsum([0] + self.dataset_lengths)[:-1]

        # Skip all but one time step
        if t_plot is not None:
            if not hasattr(t_plot, '__iter__'):
                t_plot = [t_plot]
            self.dataset_lengths = [l if i in t_plot else 25000 for i, l in enumerate(self.dataset_lengths)]

        self.n_batches = [int(np.ceil(dl / self.batch_size)) for dl in self.dataset_lengths]

    def __len__(self):
        return sum(self.n_batches)

    def __iter__(self):
        for t in range(len(self.n_batches)):
            for i in range(self.n_batches[t]):
                yield np.s_[self.anchors[t] + i * self.batch_size:
                            self.anchors[t] + min((i+1) * self.batch_size, self.dataset_lengths[t])]


def collate_skip(data):
    """
    Skip collate for already batched data
    """
    if type(data) == list:
        # Merge two batches. Can this happen for non-split-batches?
        assert len(data[0]['x'].shape) == 2
        out_batch = dict()
        for k, v in data[0].items():
            if type(v) == dict:
                for k2, v2 in data[0][k].items():
                    out_batch[k][k2] = torch.cat([d[k][k2] for d in data])
            elif type(v) == list:
                # loc
                out_batch[k] = []
                for i in range(len(data[0][k])):
                    out_batch[k] += [torch.cat([d[k][i] for d in data])]
            elif isinstance(v, (int, np.integer)):
                # time
                out_batch[k] = [d[k] for d in data]
            else:
                # x
                out_batch[k] = torch.cat([d[k] for d in data])
        return out_batch
    else:
        return data


def get_all_files(file_path):
    """
    get all files in any subfolders
    """
    if os.path.isfile(file_path):
        return [file_path]
    else:
        # Recurse into all subfiles/subfolders
        return [f_ for f in os.listdir(file_path) for f_ in get_all_files(os.path.join(file_path, f))]


class Data(Dataset):
    """
    PyTorch dataset for HDF5 data
    """
    def __init__(self, file_path, transform=None):
        if not type(file_path) == list:
            file_path = [file_path]
        self.file_paths = sum([get_all_files(fp) for fp in file_path], [])
        self.transform = transform
        self._files = [h5py.File(f, "r") for f in self.file_paths]
        self.lengths = [f['data'].shape[1] for f in self._files]
        self.anchors = np.cumsum([0] + self.lengths)[:-1]
        self._files = None

    @property
    def files(self):
        if self._files is None:
            self._files = [h5py.File(f, "r") for f in self.file_paths]
        return self._files

    def __getitem__(self, idx, fid=None):
        """
        :param idx: int (or slice)
        :param fid: time idx, if None compute time and local idx from idx
        """
        if fid is None:
            # compute time and local idx and
            fid = np.argmax(idx < self.anchors) - 1
            idx = idx - self.anchors[fid]
        slice = torch.Tensor(self.files[fid]['data'][:, idx].T)
        locs = [torch.Tensor(self.files[fid][p][idx]) for p in ['x', 'y', 'z']]
        if self.transform is not None:
            slice = self.transform(slice)
        return {'x': slice, 'loc': locs, 't': fid}

    def __getitems__(self, idx):
        """
        :param idx: slice
        """
        fids = [np.argmax(idx_ < self.anchors) - 1 for idx_ in [idx.start, idx.stop-1]]
        ancs = [self.anchors[fid] for fid in fids]
        if fids[0] != fids[1]:
            # Combine batch from two files
            return [self.__getitem__(np.s_[int(idx.start - ancs[0]):], fids[0]),
                    self.__getitem__(np.s_[:int(idx.stop - ancs[1])], fids[1])]
        else:
            return self.__getitem__(np.s_[int(idx.start - ancs[0]): int(idx.stop - ancs[0])], fids[0])

    def __len__(self):
        return sum(self.lengths)

    def size(self, dim):
        if self.transform is not None and dim == 0:
            return self[0].shape[1]
        return self.files[0]['data'].shape[dim]

    def dataloader(self, batch_size=1, shuffle=None, by_time=False, t_plot=None, **kwargs):
        if shuffle is None:
            if by_time:
                sampler = PlotSampler(self.lengths, batch_size, t_plot)
            else:
                sampler = FastSampler(len(self), batch_size, weak_shuffle=kwargs.pop('weak_shuffle', False))
            return DataLoader(self, **kwargs, batch_sampler=sampler, collate_fn=collate_skip)
        else:
            kwargs['shuffle'] = shuffle
            kwargs['batch_size'] = batch_size
            return DataLoader(self, **kwargs)


class TData(Dataset):
    """
    One-batch tensor dataset
    """
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        return self.x

    def __len__(self):
        return 1

    def size(self, dim):
        return self.x.shape[dim]

    def dataloader(self):
        return DataLoader(self, batch_size=1, collate_fn=torch.cat)


if __name__ == "__main__":
    # Preprocess
    # nc2json(DATA_PATH, os.path.splitext(DATA_PATH)[0] + '.json')
    # nc2hdf5(DATA_PATH, os.path.splitext(DATA_PATH)[0] + '.data', process=False)
    # nc2hdf5(DATA_PATH, os.path.splitext(DATA_PATH)[0] + '_nz.data')
    for f in progress(os.listdir(DATA_FOLDER_RAW), 'Preprocessing'):
        nc2hdf5(os.path.join(DATA_FOLDER_RAW, f), os.path.join(DATA_FOLDER, os.path.splitext(f)[0] + '_nz.data'))

    # Load
    data = Data(DATA_FOLDER).dataloader(batch_size=256, weak_shuffle=True)
    batch = next(iter(data))
    print(batch['x'].shape)
