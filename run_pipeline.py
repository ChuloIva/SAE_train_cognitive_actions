"""
Complete FAST-style SAE training pipeline.

Runs both activation extraction and SAE training in sequence.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n❌ Error running {description}")
        sys.exit(1)

    print(f"\n✅ {description} completed successfully")


def main():
    """Run complete pipeline."""
    # Check that required files exist
    if not Path("cognitive_actions_7k_final_1759233061.jsonl").exists():
        print("❌ Error: Dataset file 'cognitive_actions_7k_final_1759233061.jsonl' not found")
        sys.exit(1)

    if not Path("HypotheSAEs").exists():
        print("❌ Error: HypotheSAEs directory not found")
        print("Please ensure HypotheSAEs is in the current directory")
        sys.exit(1)

    print("="*70)
    print("FAST-style SAE Training Pipeline")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Extract LLM activations sequentially (FAST approach)")
    print("  2. Train SAE on extracted activations")
    print("\nEstimated time: 30-60 minutes (depending on hardware)")

    # Ask for confirmation
    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Aborted.")
        sys.exit(0)

    # Step 1: Extract activations
    activations_file = Path("activations/activations_layer12.npz")
    if activations_file.exists():
        print(f"\n⚠️  Activations file already exists: {activations_file}")
        response = input("Skip activation extraction? (y/n): ").lower().strip()
        if response == 'y':
            print("Skipping Step 1: Activation extraction")
        else:
            run_command(
                [sys.executable, "extract_activations.py"],
                "Step 1: Extracting LLM activations"
            )
    else:
        run_command(
            [sys.executable, "extract_activations.py"],
            "Step 1: Extracting LLM activations"
        )

    # Step 2: Train SAE
    run_command(
        [sys.executable, "train_sae.py"],
        "Step 2: Training SAE"
    )

    # Done
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print(f"  - Activations: {activations_file}")
    print(f"  - SAE checkpoint: checkpoints/cognitive_actions/")
    print("\nNext steps:")
    print("  - Use the trained SAE for feature interpretation")
    print("  - See README_FAST.md for interpretation examples")


if __name__ == "__main__":
    main()
