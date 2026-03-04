## Unit Testor
Testing each of the 15 Modulated signal generator


# Dataset
## SMALL DATASET



## LARGE DATASET

### Dual
❯ ls ./signal_dataset/dual/train/signals -1 | wc -l
8_25_000
❯ ls ./signal_dataset/dual/test/signals -1 | wc -l
3_78_000
❯ ls ./signal_dataset/dual/val/signals -1 | wc -l
1_32_000

### Single
❯ ls ./signal_dataset/single/train/signals -1 | wc -l
1_32_000
❯ ls ./signal_dataset/single/test/signals -1 | wc -l
84_000
❯ ls ./signal_dataset/single/val/signals -1 | wc -l
33_000



# TensorBoard Quick Setup & Usage

## Installation

```bash
# Install with pip
pip install tensorboard

# Or with conda
conda install -c conda-forge tensorboard
```

## Basic Usage

### 1. **Start TensorBoard Server**
```bash
# Point to your logs directory (as defined in CONFIG['tensorboard_dir'])
tensorboard --logdir ./runs/convnext

# Common options:
tensorboard --logdir ./runs/convnext --port 6006 --bind_all
# --port: change port (default 6006)
# --bind_all: allow external access (for remote servers)
```

### 2. **Access the Dashboard**
- Open browser to: `http://localhost:6006`
- For remote server: `http://<server-ip>:6006`

## What You'll See

| Tab | What's Monitored |
|-----|------------------|
| **SCALARS** | Loss curves, accuracy, learning rate, gradient stats, GPU memory |
| **IMAGES** | Sample spectrograms with predictions |
| **GRAPHS** | Model architecture visualization |
| **DISTRIBUTIONS** | Weight/gradient distributions over time |
| **HISTOGRAMS** | Activation histograms |

## Key Metrics in Our Implementation

```
train/
  ├── loss_batch     # Per-batch training loss
  ├── loss_epoch     # Epoch-averaged training loss
  ├── lr             # Learning rate
  ├── grad_norm      # Gradient norm (health check)
  └── grad_mean/std  # Gradient statistics

val/
  ├── loss           # Validation loss
  └── accuracy       # Validation accuracy

snr/
  └── acc_XdB        # Accuracy per SNR level

gpu/
  ├── memory_allocated  # GPU memory usage (GB)
  ├── gpu_util          # GPU utilization %
  └── temperature       # GPU temperature
```

## Quick Tips

```bash
# Compare multiple runs
tensorboard --logdir run1=./runs/convnext,run2=./runs/convnext_v2

# Reload every 5 seconds (for long training)
tensorboard --logdir ./runs/convnext --reload_interval 5

# Run in background (Linux/Mac)
nohup tensorboard --logdir ./runs/convnext --port 6006 &

# SSH tunnel for remote server
ssh -L 6006:localhost:6006 user@server-ip
# Then open http://localhost:6006 locally
```

## Expected Output in Browser

You'll see live updating charts of:
- **Training loss** decreasing
- **Validation accuracy** increasing
- **Gradient norms** staying stable (not vanishing/exploding)
- **Per-SNR accuracy** improving over time
- **GPU memory** usage (should be near capacity for optimal utilization)