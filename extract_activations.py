"""
Extract LLM activations sequentially from dataset (FAST methodology).

This script implements the FAST approach: processing each data instance independently
to preserve semantic integrity, rather than concatenating multiple instances.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def extract_activations_sequential(
    texts: List[str],
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    layer_idx: int = 12,
    max_length: int = 512,
    device: str = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    batch_size: int = 1,  # Process one at a time for true FAST approach
    cache_dir: Optional[str] = None,
) -> tuple[np.ndarray, List[int]]:
    """
    Extract activations from LLM sequentially (FAST approach).

    Args:
        texts: List of text examples
        model_name: HuggingFace model name
        layer_idx: Which layer to extract activations from
        max_length: Maximum sequence length (for truncation)
        device: Device to run model on
        batch_size: Batch size (default 1 for sequential processing)
        cache_dir: HuggingFace cache directory

    Returns:
        activations: numpy array of shape (n_examples, seq_len, hidden_dim)
        seq_lengths: list of actual sequence lengths before padding
    """
    print(f"Loading model: {model_name}")
    print(f"Extracting from layer {layer_idx}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Get hidden dimension
    hidden_dim = model.config.hidden_size
    print(f"Hidden dimension: {hidden_dim}")

    all_activations = []
    seq_lengths = []

    # Process each example sequentially (FAST approach)
    print(f"Processing {len(texts)} examples sequentially...")
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting activations"):
            # Tokenize individual example
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=False,  # No padding yet - keep original length
            ).to(device)

            actual_length = inputs['input_ids'].shape[1]
            seq_lengths.append(actual_length)

            # Forward pass to get hidden states
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            # Extract activations from specified layer
            # outputs.hidden_states is tuple of (layer0, layer1, ..., layerN)
            layer_activations = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

            # Move to CPU and convert to numpy
            layer_activations = layer_activations.squeeze(0).cpu().float().numpy()  # (seq_len, hidden_dim)

            all_activations.append(layer_activations)

            # Clear cache
            del inputs, outputs, layer_activations
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"Extracted activations from {len(all_activations)} examples")
    print(f"Sequence lengths: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")

    return all_activations, seq_lengths


def pad_activations(
    activations_list: List[np.ndarray],
    max_length: Optional[int] = None,
    pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad variable-length activations to same length.

    Args:
        activations_list: List of activations of shape (seq_len_i, hidden_dim)
        max_length: Maximum length to pad to (None = use longest sequence)
        pad_value: Value to use for padding

    Returns:
        padded_activations: array of shape (n_examples, max_seq_len, hidden_dim)
        padding_mask: boolean array of shape (n_examples, max_seq_len)
                     (True = actual token, False = padding)
    """
    n_examples = len(activations_list)
    hidden_dim = activations_list[0].shape[1]

    if max_length is None:
        max_length = max(a.shape[0] for a in activations_list)

    padded = np.full((n_examples, max_length, hidden_dim), pad_value, dtype=np.float32)
    mask = np.zeros((n_examples, max_length), dtype=bool)

    for i, acts in enumerate(activations_list):
        seq_len = min(acts.shape[0], max_length)
        padded[i, :seq_len, :] = acts[:seq_len, :]
        mask[i, :seq_len] = True

    return padded, mask


def flatten_activations(
    padded_activations: np.ndarray,
    padding_mask: np.ndarray,
    exclude_padding: bool = True
) -> np.ndarray:
    """
    Flatten sequence dimension for SAE training.

    Args:
        padded_activations: (n_examples, seq_len, hidden_dim)
        padding_mask: (n_examples, seq_len) - True for real tokens
        exclude_padding: If True, exclude padding positions

    Returns:
        flattened: (n_tokens, hidden_dim) where n_tokens depends on exclude_padding
    """
    if exclude_padding:
        # Only keep non-padded positions
        mask_expanded = padding_mask[:, :, np.newaxis]  # (n_examples, seq_len, 1)
        flattened = padded_activations[mask_expanded.squeeze(-1)]  # (n_real_tokens, hidden_dim)
    else:
        # Keep all positions including padding
        n_examples, seq_len, hidden_dim = padded_activations.shape
        flattened = padded_activations.reshape(n_examples * seq_len, hidden_dim)

    return flattened


def main():
    """Main extraction pipeline."""
    # Configuration
    DATASET_PATH = "cognitive_actions_7k_final_1759233061.jsonl"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    LAYER_IDX = 12  # Middle layer for Llama-3.1-8B (32 layers total)
    MAX_LENGTH = 512  # Reasonable max length for cognitive action examples
    CACHE_DIR = None  # Will use default HF cache
    OUTPUT_DIR = Path("activations")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}")
    data = load_dataset(DATASET_PATH)
    texts = [item['text'] for item in data]
    print(f"Loaded {len(texts)} examples")

    # Extract activations sequentially (FAST approach)
    activations_list, seq_lengths = extract_activations_sequential(
        texts=texts,
        model_name=MODEL_NAME,
        layer_idx=LAYER_IDX,
        max_length=MAX_LENGTH,
        cache_dir=CACHE_DIR,
    )

    # Pad activations to same length
    print("\nPadding activations...")
    padded_activations, padding_mask = pad_activations(activations_list)
    print(f"Padded shape: {padded_activations.shape}")

    # Flatten for SAE training (excluding padding)
    print("\nFlattening activations (excluding padding positions)...")
    flattened_activations = flatten_activations(
        padded_activations,
        padding_mask,
        exclude_padding=True
    )
    print(f"Flattened shape: {flattened_activations.shape}")

    # Save activations
    output_file = OUTPUT_DIR / f"activations_layer{LAYER_IDX}.npz"
    print(f"\nSaving to {output_file}")
    np.savez_compressed(
        output_file,
        activations=flattened_activations,
        padded_activations=padded_activations,
        padding_mask=padding_mask,
        seq_lengths=seq_lengths,
        metadata={
            'model_name': MODEL_NAME,
            'layer_idx': LAYER_IDX,
            'max_length': MAX_LENGTH,
            'n_examples': len(texts),
            'n_tokens': flattened_activations.shape[0],
        }
    )
    print("Done!")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Layer: {LAYER_IDX}")
    print(f"Examples: {len(texts)}")
    print(f"Total tokens (excl. padding): {flattened_activations.shape[0]:,}")
    print(f"Hidden dimension: {flattened_activations.shape[1]}")
    print(f"Avg tokens per example: {flattened_activations.shape[0] / len(texts):.1f}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
