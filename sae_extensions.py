"""Extensions to HypotheSAEs SparseAutoencoder for memory-efficient training."""

import torch
import numpy as np
from tqdm.auto import tqdm
import os


def fit_with_loader(
    self,
    train_loader,
    val_loader=None,
    save_dir=None,
    learning_rate: float = 5e-4,
    n_epochs: int = 100,
    aux_coef: float = 1/32,
    multi_coef: float = 0.0,
    patience: int = 5,
    show_progress: bool = True,
    clip_grad: float = 1.0
):
    """Train the sparse autoencoder using DataLoader objects.

    This method accepts PyTorch DataLoaders instead of full tensors,
    allowing for memory-efficient training with memory-mapped files.

    Args:
        train_loader: PyTorch DataLoader for training data
        val_loader: Optional PyTorch DataLoader for validation data
        save_dir: Directory to save checkpoints
        learning_rate: Learning rate for optimizer
        n_epochs: Maximum number of training epochs
        aux_coef: Coefficient for auxiliary loss
        multi_coef: Coefficient for multi-k loss
        patience: Early stopping patience
        show_progress: Whether to show progress bar
        clip_grad: Gradient clipping value
    """
    from hypothesaes.sae import get_sae_checkpoint_name

    # Initialize weights from first batch
    first_batch = next(iter(train_loader))[0]
    self.initialize_weights_(first_batch.to(self.device))

    optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    # Training loop setup
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'dead_neuron_ratio': []}

    # Training loop
    iterator = tqdm(range(n_epochs)) if show_progress else range(n_epochs)
    for epoch in iterator:
        self.train()
        train_losses = []

        for batch_x, in train_loader:
            batch_x = batch_x.to(self.device)
            recon, info = self(batch_x)
            loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)

            optimizer.zero_grad()
            loss.backward()
            self.adjust_decoder_gradient_()

            # Apply gradient clipping
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad)

            optimizer.step()
            self.normalize_decoder_()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Track dead neurons
        dead_ratio = (self.steps_since_activation > self.dead_neuron_threshold_steps).float().mean().item()
        history['dead_neuron_ratio'].append(dead_ratio)

        # Validation
        avg_val_loss = None
        if val_loader is not None:
            self.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, in val_loader:
                    batch_x = batch_x.to(self.device)
                    recon, info = self(batch_x)
                    val_loss = self.compute_loss(batch_x, recon, info, aux_coef, multi_coef)
                    val_losses.append(val_loss.item())

            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if show_progress:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Update progress bar
        if show_progress:
            postfix = {
                'train_loss': f'{avg_train_loss:.4f}',
                'val_loss': f'{avg_val_loss:.4f}' if val_loader else 'N/A',
                'dead_ratio': f'{dead_ratio:.3f}'
            }
            if self.use_batch_topk:
                postfix['threshold'] = f'{self.threshold.item():.2e}'
            iterator.set_postfix(postfix)

    # Save final model
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        filename = get_sae_checkpoint_name(self.m_total_neurons, self.k_active_neurons, self.prefix_lengths)
        self.save(os.path.join(save_dir, filename))

    return history
