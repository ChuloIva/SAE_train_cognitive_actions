"""Memory-mapped dataset for efficient activation loading."""

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """Dataset that loads activations from a memory-mapped file.

    This avoids loading the entire dataset into RAM, which is useful
    for large datasets that exceed available memory.
    """

    def __init__(self, mmap_path: str, shape: tuple, dtype='float32'):
        """Initialize memory-mapped dataset.

        Args:
            mmap_path: Path to the memory-mapped file
            shape: Shape of the data (n_samples, n_features)
            dtype: Data type of the array
        """
        self.data = np.memmap(mmap_path, dtype=dtype, mode='r', shape=shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Copy to avoid keeping the memmap file open
        return torch.from_numpy(self.data[idx].copy()).float()


def train_sae_memmap(
    mmap_path: str,
    shape: tuple,
    M: int,
    K: int,
    **kwargs
):
    """Train SAE using memory-mapped activations.

    This is a wrapper around hypothesaes.train_sae that uses
    memory-mapped files instead of loading all data into RAM.

    Args:
        mmap_path: Path to memory-mapped activation file
        shape: Shape of the activations (n_samples, n_features)
        M: Number of SAE features
        K: Active features per example
        **kwargs: Additional arguments passed to train_sae

    Returns:
        Trained SparseAutoencoder
    """
    import sys
    sys.path.insert(0, "HypotheSAEs")

    from hypothesaes.sae import SparseAutoencoder, get_sae_checkpoint_name, load_model
    from torch.utils.data import DataLoader
    import os
    from sae_extensions import fit_with_loader

    # Extract parameters
    val_mmap_path = kwargs.pop('val_mmap_path', None)
    val_shape = kwargs.pop('val_shape', None)
    checkpoint_dir = kwargs.pop('checkpoint_dir', None)
    overwrite_checkpoint = kwargs.pop('overwrite_checkpoint', False)
    matryoshka_prefix_lengths = kwargs.pop('matryoshka_prefix_lengths', None)
    batch_topk = kwargs.pop('batch_topk', False)
    aux_k = kwargs.pop('aux_k', None)
    multi_k = kwargs.pop('multi_k', None)
    dead_neuron_threshold_steps = kwargs.pop('dead_neuron_threshold_steps', 256)
    batch_size = kwargs.pop('batch_size', 512)
    learning_rate = kwargs.pop('learning_rate', 5e-4)
    n_epochs = kwargs.pop('n_epochs', 100)
    aux_coef = kwargs.pop('aux_coef', 1/32)
    multi_coef = kwargs.pop('multi_coef', 0.0)
    patience = kwargs.pop('patience', 3)
    clip_grad = kwargs.pop('clip_grad', 1.0)
    show_progress = kwargs.pop('show_progress', True)

    input_dim = shape[1]

    # Check for existing checkpoint
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_name = get_sae_checkpoint_name(M, K, matryoshka_prefix_lengths)
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        if os.path.exists(checkpoint_path) and not overwrite_checkpoint:
            return load_model(checkpoint_path)

    # Create datasets
    train_dataset = MemmapDataset(mmap_path, shape)
    val_dataset = MemmapDataset(val_mmap_path, val_shape) if val_mmap_path else None

    # Create SAE
    sae = SparseAutoencoder(
        input_dim=input_dim,
        m_total_neurons=M,
        k_active_neurons=K,
        aux_k=aux_k,
        multi_k=multi_k,
        dead_neuron_threshold_steps=dead_neuron_threshold_steps,
        prefix_lengths=matryoshka_prefix_lengths,
        use_batch_topk=batch_topk,
    )

    # Monkey-patch the fit_with_loader method
    import types
    sae.fit_with_loader = types.MethodType(fit_with_loader, sae)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0) if val_dataset else None

    # Train using custom fit method
    sae.fit_with_loader(
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=checkpoint_dir,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        aux_coef=aux_coef,
        multi_coef=multi_coef,
        patience=patience,
        clip_grad=clip_grad,
        show_progress=show_progress,
    )

    return sae
