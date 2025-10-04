"""
Interpret trained SAE features.

This script uses HypotheSAEs' interpretation functionality to generate
natural language descriptions of what each SAE neuron represents.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add HypotheSAEs to path
sys.path.insert(0, str(Path(__file__).parent / "HypotheSAEs"))

from hypothesaes.quickstart import interpret_sae
from hypothesaes.sae import load_model


def load_texts(dataset_path: str):
    """Load texts from JSONL dataset."""
    texts = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
    return texts


def main():
    """Main interpretation pipeline."""
    # Configuration
    DATASET_PATH = "cognitive_actions_7k_final_1759233061.jsonl"
    ACTIVATIONS_PATH = "activations/activations_layer12.npz"
    SAE_CHECKPOINT = "checkpoints/cognitive_actions/SAE_M=256_K=8.pt"

    # Interpretation parameters
    N_TOP_NEURONS = 20  # Number of neurons to interpret
    INTERPRETER_MODEL = "gpt-4.1"  # Or use local model
    N_EXAMPLES = 20  # Examples to show LLM per neuron
    PRINT_TOP_EXAMPLES = 5  # Top activating examples to print

    print("="*60)
    print("SAE Feature Interpretation")
    print("="*60)

    # Load data
    print("\nLoading data...")
    texts = load_texts(DATASET_PATH)
    print(f"Loaded {len(texts)} texts")

    # Load activations
    data = np.load(ACTIVATIONS_PATH)
    activations = data['activations']
    print(f"Loaded activations: {activations.shape}")

    # Load SAE
    print(f"\nLoading SAE from {SAE_CHECKPOINT}")
    sae = load_model(SAE_CHECKPOINT)

    # Get SAE activations on the data
    print("\nComputing SAE activations...")
    sae_activations = sae.get_activations(activations, show_progress=True)
    print(f"SAE activations shape: {sae_activations.shape}")

    # Note: We have flattened activations (one per token)
    # But we need to map back to examples for interpretation
    # For simplicity, we'll use mean pooling over tokens per example

    # Load metadata to get sequence lengths
    seq_lengths = data['seq_lengths']
    n_examples = len(seq_lengths)

    print(f"\nPooling SAE activations per example...")
    # Compute example-level activations by mean pooling
    example_activations = []
    token_idx = 0
    for length in seq_lengths:
        # Get activations for this example's tokens
        example_acts = sae_activations[token_idx:token_idx+length]
        # Mean pool (you could also use max pooling)
        pooled = example_acts.mean(axis=0)
        example_activations.append(pooled)
        token_idx += length

    example_activations = np.array(example_activations)
    print(f"Example activations shape: {example_activations.shape}")

    # Interpret neurons
    print("\n" + "="*60)
    print("Interpreting SAE Features")
    print("="*60)
    print(f"Using {INTERPRETER_MODEL} to interpret top {N_TOP_NEURONS} neurons")
    print(f"Showing {N_EXAMPLES} examples per neuron to the LLM")
    print("\n⚠️  Note: This requires OPENAI_KEY_SAE environment variable")

    # Task-specific instructions for cognitive actions
    TASK_INSTRUCTIONS = """All texts are examples of cognitive actions and metacognitive processes.
Features should describe specific types of cognitive actions, mental processes, or patterns of thinking.
Examples:
- "describes reconsidering a previous belief or decision"
- "mentions awareness of one's own bias or assumptions"
- "expresses uncertainty about a course of action"
"""

    try:
        results = interpret_sae(
            texts=texts,
            embeddings=example_activations,
            sae=sae,
            n_top_neurons=N_TOP_NEURONS,
            interpreter_model=INTERPRETER_MODEL,
            n_examples_for_interpretation=N_EXAMPLES,
            print_examples_n=PRINT_TOP_EXAMPLES,
            task_specific_instructions=TASK_INSTRUCTIONS,
        )

        # Save results
        output_file = "interpretations.csv"
        results.to_csv(output_file, index=False)
        print(f"\n✅ Saved interpretations to {output_file}")

    except Exception as e:
        print(f"\n❌ Error during interpretation: {e}")
        print("\nPossible issues:")
        print("  1. OPENAI_KEY_SAE environment variable not set")
        print("  2. OpenAI API rate limits")
        print("  3. Model name incorrect")
        print("\nTo use local models instead, edit this script and change INTERPRETER_MODEL")

    print("\n" + "="*60)
    print("Interpretation Complete")
    print("="*60)


if __name__ == "__main__":
    main()
