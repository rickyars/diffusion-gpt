"""
Generic dataset loader for character-level discrete diffusion.
Reads .txt files and builds vocabulary dynamically.
"""

import os
import pickle
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.utils.data as data


class CharacterDataset(data.Dataset):
    """
    Character-level dataset for discrete diffusion.

    Each item is a 1D tensor of character indices of length `context_len`.
    Reads from a .txt file where each line is a document or paragraph.
    """

    def __init__(
        self,
        data_path: str,
        context_len: int = 256,
        split: str = "train",
        val_split: float = 0.1,
        vocab_path: Optional[str] = None,
    ):
        """
        Args:
            data_path: Path to .txt file (one document per line)
            context_len: Length of each training sequence
            split: 'train' or 'val'
            val_split: Fraction of data to use for validation
            vocab_path: Optional path to load existing vocabulary pickle
        """
        self.context_len = context_len
        self.split = split

        # Load or build vocabulary
        if vocab_path and os.path.exists(vocab_path):
            print(f"Loading vocabulary from {vocab_path}")
            with open(vocab_path, 'rb') as f:
                # Note: weights_only not available for pickle.load, this is safe for our vocab files
                meta = pickle.load(f)
                self.stoi = meta['stoi']
                self.itos = meta['itos']
                self.vocab_size = meta['vocab_size']
        else:
            print(f"Building vocabulary from {data_path}")
            self.stoi, self.itos, self.vocab_size = self._build_vocab(data_path)

        # Load and encode text
        print(f"Loading and encoding text from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Encode the entire text
        self.data = self._encode(text)
        print(f"Loaded {len(self.data):,} characters")

        # Split into train/val
        split_idx = int(len(self.data) * (1 - val_split))
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        print(f"{split} split has {len(self.data):,} characters")
        self._n = max(0, len(self.data) - self.context_len)

    def _build_vocab(self, data_path: str) -> Tuple[Dict[str, int], Dict[int, str], int]:
        """Build character-level vocabulary from text file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Get unique characters
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # Create mappings
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        print(f"Vocabulary: {repr(''.join(chars))}")
        print(f"Vocabulary size: {vocab_size}")

        return stoi, itos, vocab_size

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to integer indices."""
        indices = [self.stoi[ch] for ch in text if ch in self.stoi]
        return np.array(indices, dtype=np.int64)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self._n:
            raise IndexError(f"Index {index} out of range for dataset of length {self._n}.")
        # Slice a contiguous window
        x = self.data[index: index + self.context_len]
        return torch.from_numpy(x.copy())

    def save_vocab(self, save_path: str):
        """Save vocabulary to pickle file for reuse."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        meta = {
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size
        }
        with open(save_path, 'wb') as f:
            pickle.dump(meta, f)
        print(f"Saved vocabulary to {save_path}")


def get_data_loader(
    data_path: str,
    batch_size: int,
    context_len: int = 256,
    split: str = "train",
    val_split: float = 0.1,
    vocab_path: Optional[str] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[data.DataLoader, int, Dict[int, str], Dict[str, int]]:
    """
    Create a DataLoader for character-level text data.

    Args:
        data_path: Path to .txt file
        batch_size: Batch size
        context_len: Sequence length
        split: 'train' or 'val'
        val_split: Validation split fraction
        vocab_path: Optional path to vocabulary pickle
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        (dataloader, vocab_size, itos, stoi)
    """
    dataset = CharacterDataset(
        data_path=data_path,
        context_len=context_len,
        split=split,
        val_split=val_split,
        vocab_path=vocab_path,
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, dataset.vocab_size, dataset.itos, dataset.stoi
