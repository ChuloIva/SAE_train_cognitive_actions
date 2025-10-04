# Files Overview

## 📁 Project Structure

```
SAE_train/
│
├── 🔵 Core Scripts (Run These)
│   ├── extract_activations.py       ⭐ Step 1: Extract LLM activations
│   ├── train_sae.py                 ⭐ Step 2: Train SAE
│   ├── interpret_sae.py             ⭐ Step 3: Interpret features (optional)
│   └── run_pipeline.py              ⭐ All-in-one runner (recommended)
│
├── 📖 Documentation (Read These)
│   ├── QUICKSTART.md                🚀 START HERE - Quick reference
│   ├── IMPLEMENTATION_SUMMARY.md    📋 High-level overview
│   ├── README_FAST.md               📚 Complete methodology
│   ├── FAST_vs_HypotheSAEs.md       🔬 Technical comparison
│   └── FILES_OVERVIEW.md            📁 This file
│
├── 📂 Input Data (Already Exists)
│   ├── cognitive_actions_7k_final_1759233061.jsonl  (Your dataset)
│   ├── HypotheSAEs/                 (SAE library)
│   ├── padding_seq.md               (Research notes)
│   ├── paper.md                     (FAST paper)
│   └── 2506.07691v1.pdf            (FAST paper PDF)
│
└── 📂 Output (Created by Scripts)
    ├── activations/
    │   └── activations_layer12.npz  (Created by Step 1)
    ├── checkpoints/
    │   └── cognitive_actions/
    │       └── SAE_M=256_K=8.pt     (Created by Step 2)
    └── interpretations.csv          (Created by Step 3)
```

## 📋 File Descriptions

### Core Scripts

#### 1. `extract_activations.py` ⭐
**What it does**: Extracts LLM activations using FAST methodology
- Loads Llama-3.1-8B-Instruct
- Processes each example sequentially (no concatenation)
- Extracts layer 12 activations
- Handles variable-length sequences
- Outputs: `activations/activations_layer12.npz`

**When to run**: First step, or when changing:
- Model (e.g., switch to different Llama model)
- Layer (e.g., extract from layer 18 instead of 12)
- Dataset (e.g., new JSONL file)

**Time**: ~30 min on GPU, ~2 hours on CPU

#### 2. `train_sae.py` ⭐
**What it does**: Trains Sparse Autoencoder on activations
- Loads activations from Step 1
- Trains Top-K SAE using HypotheSAEs
- M=256, K=8 (configurable)
- Optional Matryoshka prefixes [64, 256]
- Outputs: `checkpoints/cognitive_actions/SAE_M=256_K=8.pt`

**When to run**: After Step 1, or when changing:
- SAE hyperparameters (M, K)
- Training settings (epochs, learning rate)
- Matryoshka prefixes

**Time**: ~15 min on GPU, ~45 min on CPU

#### 3. `interpret_sae.py` ⭐
**What it does**: Generates natural language feature descriptions
- Loads trained SAE from Step 2
- Interprets top 20 neurons
- Uses GPT-4.1 for interpretation
- Outputs: `interpretations.csv`

**When to run**: After Step 2, optional
**Requires**: OPENAI_KEY_SAE environment variable
**Time**: ~10 min (depends on API speed)

#### 4. `run_pipeline.py` ⭐ RECOMMENDED
**What it does**: Runs Steps 1+2 automatically
- Checks prerequisites
- Asks for confirmation
- Runs extraction then training
- Shows progress and summaries

**When to run**: First time, or full re-run
**Time**: ~45 min total on GPU

### Documentation

#### `QUICKSTART.md` 🚀 START HERE
- TL;DR commands
- Configuration guide
- Hyperparameter selection
- Troubleshooting
- Quick reference tables

**Read this first if**: You want to run the pipeline quickly

#### `IMPLEMENTATION_SUMMARY.md` 📋
- High-level overview
- What we built and why
- Key achievements
- Usage examples
- Expected results

**Read this if**: You want to understand what the implementation does

#### `README_FAST.md` 📚
- Complete methodology explanation
- FAST approach details
- Pipeline breakdown
- Hardware requirements
- Detailed parameter guide

**Read this if**: You want in-depth understanding of FAST methodology

#### `FAST_vs_HypotheSAEs.md` 🔬
- FAST paper methodology
- HypotheSAEs library design
- Technical comparison
- Implementation adaptations
- Why our approach works

**Read this if**: You want to understand the technical details

## 🎯 Which File Should I Use?

### I want to...

**...run the pipeline quickly**
→ Run: `python run_pipeline.py`
→ Read: `QUICKSTART.md`

**...understand what this does**
→ Read: `IMPLEMENTATION_SUMMARY.md`

**...change hyperparameters**
→ Edit: `train_sae.py` (M, K, epochs, etc.)
→ Reference: `QUICKSTART.md` section "Choosing Hyperparameters"

**...use a different layer**
→ Edit: `extract_activations.py` (LAYER_IDX)
→ Reference: `QUICKSTART.md` section "Which Layer?"

**...understand FAST methodology**
→ Read: `README_FAST.md`
→ Read: `FAST_vs_HypotheSAEs.md`

**...troubleshoot errors**
→ Reference: `QUICKSTART.md` section "Troubleshooting"

**...interpret SAE features**
→ Run: `python interpret_sae.py`
→ Check: `interpretations.csv`

**...understand technical details**
→ Read: `FAST_vs_HypotheSAEs.md`

## 📊 File Sizes

| File | Size | Notes |
|------|------|-------|
| `extract_activations.py` | ~8 KB | Script |
| `train_sae.py` | ~6 KB | Script |
| `interpret_sae.py` | ~5 KB | Script |
| `run_pipeline.py` | ~3 KB | Script |
| `activations_layer12.npz` | **~500 MB** | Large output file |
| `SAE_M=256_K=8.pt` | **~30 MB** | Model checkpoint |
| `interpretations.csv` | ~50 KB | Text file |

## 🔄 Workflow

### First Time Setup
```
1. Read QUICKSTART.md (5 min)
2. Run: python run_pipeline.py (45 min)
3. Check outputs in activations/ and checkpoints/
4. (Optional) Run: python interpret_sae.py (10 min)
```

### Adjusting Hyperparameters
```
1. Edit train_sae.py (change M, K, etc.)
2. Run: python train_sae.py (15 min)
3. Compare results
```

### Using Different Layer
```
1. Edit extract_activations.py (change LAYER_IDX)
2. Run: python extract_activations.py (30 min)
3. Run: python train_sae.py (15 min)
```

### Understanding Implementation
```
1. Read IMPLEMENTATION_SUMMARY.md
2. Read FAST_vs_HypotheSAEs.md
3. Reference README_FAST.md as needed
```

## 🎓 Reading Order (for learning)

**If you're new to SAEs and FAST**:
1. `QUICKSTART.md` - Get overview and run pipeline
2. `IMPLEMENTATION_SUMMARY.md` - Understand what we built
3. `README_FAST.md` - Learn FAST methodology
4. `FAST_vs_HypotheSAEs.md` - Deep dive into technical details

**If you just want to run it**:
1. `QUICKSTART.md` - Quick reference
2. Run: `python run_pipeline.py`
3. Done!

**If you want to understand the theory**:
1. `padding_seq.md` - Original research question
2. `README_FAST.md` - FAST methodology
3. `FAST_vs_HypotheSAEs.md` - Implementation details
4. `paper.md` / `2506.07691v1.pdf` - Full FAST paper

## 🔍 Quick Find

| I need to... | File | Section |
|--------------|------|---------|
| Run the pipeline | `run_pipeline.py` | - |
| Change M or K | `train_sae.py` | Line 18-19 |
| Change layer | `extract_activations.py` | Line 79 |
| Change model | `extract_activations.py` | Line 78 |
| See examples | `QUICKSTART.md` | "Expected Output" |
| Fix OOM error | `QUICKSTART.md` | "Troubleshooting" |
| Understand FAST | `README_FAST.md` | "Overview" |
| Compare methods | `FAST_vs_HypotheSAEs.md` | "Comparison Summary" |

## ✅ Checklist

Before running the pipeline, make sure you have:

- [ ] `cognitive_actions_7k_final_1759233061.jsonl` in current directory
- [ ] `HypotheSAEs/` directory in current directory
- [ ] PyTorch, transformers, numpy, tqdm installed
- [ ] GPU available (or prepared to wait for CPU)
- [ ] Llama-3.1-8B-Instruct in HuggingFace cache (or it will download)

To run interpretation (optional):
- [ ] `OPENAI_KEY_SAE` environment variable set

## 🚀 Quick Start Command

```bash
# Everything you need in one command
python run_pipeline.py
```

That's it! See `QUICKSTART.md` for more details.
