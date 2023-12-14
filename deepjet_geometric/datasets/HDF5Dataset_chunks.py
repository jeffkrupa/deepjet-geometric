import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class HDF5Dataset_chunks(Dataset):
    def __init__(self, root_dir, Nevents=None, chunk_size=1000000, batch_size=32):
        super().__init__()
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.cache = {}
        self.file_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.h5')]
        self.lengths = []
        self.total_size = 0

        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                file_length = file['features'].shape[0]
                if Nevents is not None and self.total_size + file_length > Nevents:
                    remaining = Nevents - self.total_size
                    self.lengths.append(remaining)
                    self.total_size += remaining
                    break
                else:
                    self.lengths.append(file_length)
                    self.total_size += file_length

        # Adjust total_size to reflect the number of batches, not events
        self.total_size = (self.total_size + self.batch_size - 1) // self.batch_size

    def __len__(self):
        return self.total_size

    def load_chunk(self, chunk_idx):
        # Check if the chunk is already loaded
        if chunk_idx in self.cache:
            return self.cache[chunk_idx]

        # Calculate global start and end indices for the chunk
        global_start_idx = chunk_idx * self.chunk_size
        global_end_idx = min(global_start_idx + self.chunk_size, self.total_size * self.batch_size)

        # Initialize a list to hold data for this chunk
        chunk_data = []

        # Loop over files and indices to load data
        while global_start_idx < global_end_idx:
            file_idx = next(i for i, v in enumerate(np.cumsum(self.lengths)) if v > global_start_idx)
            idx_in_file = global_start_idx - (np.cumsum(self.lengths)[file_idx - 1] if file_idx > 0 else 0)
            items_to_read = min(global_end_idx - global_start_idx, self.lengths[file_idx] - idx_in_file)

            with h5py.File(self.file_paths[file_idx], 'r') as file:
                features = file['features'][idx_in_file:idx_in_file + items_to_read, :, :]
                chunk_data.extend([torch.from_numpy(f).float() for f in features])

            global_start_idx += items_to_read

        # Store the loaded data in the cache
        self.cache[chunk_idx] = chunk_data

    def __getitem__(self, idx):
        chunk_idx = idx // (self.chunk_size // self.batch_size)
        if chunk_idx not in self.cache:
            self.load_chunk(chunk_idx)

        batch_start = (idx % (self.chunk_size // self.batch_size)) * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.cache[chunk_idx]))
        return {"x_pf": torch.stack(self.cache[chunk_idx][batch_start:batch_end])}

# Example usage
# dataset = HDF5Dataset_chunks(root_dir='/path/to/hdf5/files', Nevents=10000, batch_size=32)
# loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

