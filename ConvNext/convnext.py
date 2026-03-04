"""
Complete ConvNeXt Implementation for Signal Modulation Classification
GPU-Optimized with Comprehensive Monitoring and Checkpointing
"""

import os
import sys
import time
import json
import math
import pickle
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, OrderedDict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import torchvision.transforms as transforms

import torchvision.transforms as transforms
from scipy.signal import stft
from scipy.ndimage import zoom

# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose your GPU target
# GPU Option-1: RTX 4060 Laptop (3072 cores, 8GB VRAM)
# GPU Option-2: RTX 4090 (16384 cores, 24GB VRAM)

DATASET='../signal_dataset_small'

CONFIG = {
    # Dataset paths
    'data_root': f'./{DATASET}/',
    'single_csv': f'./{DATASET}/single/train/metadata/global_metadata.csv',
    'dual_csv': f'./{DATASET}/dual/train/metadata/global_metadata.csv',
    
    # Training parameters
    'batch_size': 256,           # 256 for 4060 (8GB), 512 for 4090 (24GB)
    'epochs': 150,
    'lr': 1e-3,
    'weight_decay': 0.05,
    'num_classes': 16,
    
    # Model variant
    'variant': 'ronto',           # 'ronto' for faster, 'queto' for better accuracy
    
    # Optimizer
    'optimizer': 'adamw',
    'betas': (0.9, 0.995),
    
    # Hardware
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,              # Adjust based on CPU cores
    'prefetch_factor': 2,
    'pin_memory': True,
    
    # Mixed precision (crucial for throughput)
    # 'use_amp': True,
    'use_amp': False, # try later
    
    
    # Checkpointing
    'resume_path': './checkpoints/convnext_latest.pth',
    'best_model_path': './checkpoints/convnext_best.pth',
    'save_interval': 5,             # Save every N epochs
    
    # Logging
    'log_interval': 20,             # Batches between logging
    'viz_interval': 100,             # Batches between TFI visualization
    'tensorboard_dir': './runs/convnext',
    
    # Gradient monitoring
    'grad_clip': 1.0,                # Max gradient norm
    'grad_check_interval': 50,        # Check gradient stats every N batches
    
    # Data augmentation (as per paper)
    'augmentation': {
        'hflip_prob': 0.5,
        'vflip_prob': 0.5,
        'random_crop_scale': (0.08, 1.0),
        'random_crop_ratio': (0.75, 1.25),
        'random_erase_prob': 0.33,
        'random_erase_scale': (0.02, 0.33),
    },


    # Signal parameters (as per Table I in paper)
    "modulation_params": {
        "NM": {"f0_range": (0.1, 0.4)},
        "LFM": {"f0_range": (0.01, 0.45), "delta_f_range": (0.05, 0.4)},
        "DLFM": {"f0_range": (0.01, 0.4), "delta_f_range": (0.05, 0.35)},
        "MLFM": {"f0_range": (0.15, 0.5), "delta_f_range": (0.1, 0.35), "r_range": (0.3, 0.7)},
        "EQFM": {"fmin_range": (0.01, 0.4), "delta_f_range": (0.05, 0.3)},
        "SFM": {"fmin_range": (0.01, 0.15), "delta_f_range": (0.05, 0.35), 
                "fSFM_range": (0.75, 10), "phiSFM_range": (0, 2*np.pi)},
        "BFSK": {"f1_range": (0.05, 0.45), "f2_range": (0.05, 0.45), "N_values": [5, 7, 11, 13]},
        "QFSK": {"f_range": (0.05, 0.45), "N_value": 4},
        "BPSK": {"f0_range": (0.05, 0.45), "N_values": [5, 7, 11, 13]},
        "Frank": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
        "P1": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
        "P2": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
        "P3": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
        "P4": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
        "LFM_BPSK": {"fmin_range": (0.05, 0.45), "delta_f_range": (0.05, 0.4), 
                    "N_values": [5, 7, 11, 13]}
    },

    "modulation_types": [
        "NM", "LFM", "DLFM", "MLFM", "EQFM", "SFM", 
        "BFSK", "QFSK", "BPSK", "Frank", "P1", "P2", "P3", "P4", "LFM_BPSK"
    ],
    
    # SNR levels for evaluation
    'snr_levels': list(range(-21, 19, 3)),
    
    
    # STFT parameters
    'stft': {
        'n_fft': 512,
        'win_length': 101,
        'hop_length': 8,
        'window': 'hann',
        'tf_shape': (256, 256),
    }
}

# Override batch size based on GPU
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_memory = props.total_memory / 1024**3  # GB
    
    if total_memory >= 20:  # RTX 4090
        CONFIG['batch_size'] = 512
        CONFIG['num_workers'] = 8
    elif total_memory >= 8:  # RTX 4060
        CONFIG['batch_size'] = 256
        CONFIG['num_workers'] = 4
    else:
        CONFIG['batch_size'] = 128
        CONFIG['num_workers'] = 2

# Create directories
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./logs', exist_ok=True)
os.makedirs('./viz', exist_ok=True)

# ============================================================================
# GPU UTILIZATION MONITORING
# ============================================================================

class GPUMonitor:
    """Monitor GPU utilization, memory, and temperature"""
    
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.history = defaultdict(list)
        
    def log_stats(self):
        if not torch.cuda.is_available():
            return {}
        
        stats = {
            'memory_allocated': torch.cuda.memory_allocated(self.device_id) / 1024**3,
            'memory_reserved': torch.cuda.memory_reserved(self.device_id) / 1024**3,
            'max_memory_allocated': torch.cuda.max_memory_allocated(self.device_id) / 1024**3,
        }
        
        # Try to get utilization (not always available)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats['gpu_util'] = util.gpu
            stats['memory_util'] = util.memory
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats['temperature'] = temp
        except:
            pass
        
        for k, v in stats.items():
            self.history[k].append(v)
        
        return stats
    
    def print_summary(self):
        if not self.history:
            return
        
        print("\n" + "="*60)
        print("GPU UTILIZATION SUMMARY")
        print("="*60)
        for k, v in self.history.items():
            if v:
                print(f"{k:20s}: Mean={np.mean(v):.2f}, Max={np.max(v):.2f}, Min={np.min(v):.2f}")
        print("="*60 + "\n")

# ============================================================================
# MODEL ARCHITECTURE (GPU-OPTIMIZED)
# ============================================================================

class LayerNorm2d(nn.Module):
    """
    Channel-wise LayerNorm for 2D feature maps
    More GPU-friendly than batch norm for small batches
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        
    def forward(self, x):
        # Compute mean and variance across spatial dimensions
        # Shape: (B, C, H, W) -> (B, C, 1, 1)
        u = x.mean(dim=[2, 3], keepdim=True)
        s = (x - u).pow(2).mean(dim=[2, 3], keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with optimized operations for GPU parallelism
    """
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        
        # Depthwise conv: groups=dim for maximum parallelism
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # Layer norm
        self.norm = LayerNorm2d(dim)
        
        # Pointwise/1x1 convs - optimized for tensor cores
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        # Stochastic depth
        self.drop_path = nn.Identity() if drop_path == 0 else DropPath(drop_path)
        
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = shortcut + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtSignalClassifier(nn.Module):
    """
    ConvNeXt architecture optimized for signal spectrograms
    Channel progression designed for GPU tensor cores:
    - Powers of 2 for optimal memory alignment
    - Gradual increase to maintain parallel operations
    """
    
    # Variant configurations from paper
    VARIANTS = {
        'ronto': {
            'dims': [16, 32, 64, 128],
            'depths': [1, 1, 1, 1],           # 1 block per stage
            'drop_path_rate': 0.2,
        },
        'queto': {
            'dims': [32, 48, 64, 96],
            'depths': [1, 1, 1, 1],
            'drop_path_rate': 0.2,
        },
        # For RTX 4090 - larger variant
        '4090': {
            'dims': [32, 64, 128, 256],
            'depths': [2, 2, 2, 2],           # 2 blocks per stage
            'drop_path_rate': 0.3,
        }
    }
    
    def __init__(self, variant='ronto', num_classes=16, in_chans=1):
        super().__init__()
        
        if variant not in self.VARIANTS:
            variant = 'ronto'
        
        config = self.VARIANTS[variant]
        dims = config['dims']
        depths = config['depths']
        drop_path_rate = config['drop_path_rate']
        
        self.variant = variant
        self.dims = dims
        
        # Build stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Downsample layers (stem)
        self.downsample_layers = nn.ModuleList()
        
        # Stem: initial convolution
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # Downsampling between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Stages (ConvNeXt blocks)
        self.stages = nn.ModuleList()
        block_idx = 0
        
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=dims[i],
                    drop_path=dpr[block_idx + j]
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            block_idx += depths[i]
        
        # Final normalization and head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, LayerNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        
        # Global average pooling
        x = x.mean([-2, -1])  # (B, C)
        
        # Final norm and head
        x = self.norm(x)
        x = self.head(x)
        
        return x
    
    def get_flops(self, input_size=(1, 256, 256)):
        """Calculate FLOPs for the model"""
        from fvcore.nn import FlopCountAnalysis
        
        dummy = torch.randn(1, 1, *input_size)
        flops = FlopCountAnalysis(self, dummy)
        return flops.total() / 1e9  # GFLOPs


# ============================================================================
# SIGNAL TO SPECTROGRAM CONVERTER
# ============================================================================
class SignalToSpectrogram(nn.Module): 
    """
    Convert raw signal to spectrogram using STFT
    GPU-accelerated implementation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        stft_params = config['stft']
        
        self.n_fft = stft_params['n_fft']
        self.win_length = stft_params['win_length']
        self.hop_length = stft_params['hop_length']
        self.tf_shape = stft_params['tf_shape']
        
        # Pre-compute window on GPU for efficiency
        self.register_buffer('window', torch.hann_window(self.win_length))
        
    def to_device(self, device):
        self.window = self.window.to(device)
        return self
    
    @torch.no_grad()
    def __call__(self, signal):
        """
        Args:
            signal: (B, N) or (N,) tensor of raw signal
        Returns:
            spectrogram: (B, 1, H, W) tensor
        """
        was_batch = True
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            was_batch = False
        
        # STFT - uses cuFFT on GPU
        stft_result = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(signal.device),
            return_complex=True,
            center=True
        )  # (B, F, T)
        
        # Convert to magnitude spectrogram
        spectrogram = torch.abs(stft_result)  # (B, F, T)
        
        # Remove negative frequencies (keep first half)
        spectrogram = spectrogram[:, :self.n_fft // 2, :]  # (B, F//2, T)
        
        # Resize to target shape using interpolation
        # More GPU-friendly than scipy zoom
        if spectrogram.shape[1:] != self.tf_shape:
            spectrogram = F.interpolate(
                spectrogram.unsqueeze(1),  # (B, 1, F, T)
                size=self.tf_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(1)  # (B, H, W)
        
        # Normalize
        max_vals = spectrogram.amax(dim=[1, 2], keepdim=True)
        spectrogram = spectrogram / (max_vals + 1e-8)
        
        # Add channel dimension
        spectrogram = spectrogram.unsqueeze(1)  # (B, 1, H, W)
        
        if not was_batch:
            spectrogram = spectrogram.squeeze(0)
        
        return spectrogram


# ============================================================================
# DATASET WITH ON-THE-FLY TFI GENERATION
# ============================================================================

class SignalDataset(Dataset):
    """
    Dataset that loads raw signals and generates TFI on-the-fly
    Includes extensive error checking and monitoring
    """
    def __init__(self, 
                 single_csv: str,
                 dual_csv: str,
                 data_root: str,
                 config: Dict,
                 split: str = 'train',
                 transform=None,
                 max_samples: Optional[int] = None):
        
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Statistics tracking
        self.load_stats = {
            'total_attempts': 0,
            'successes': 0,
            'failures': 0,
            'corrupted': 0,
            'snr_distribution': defaultdict(int),
            'modulation_distribution': defaultdict(int)
        }
        
        # Load metadata
        print(f"\n[Dataset] Loading {split} metadata...")
        self.metadata = self._load_metadata(single_csv, dual_csv)
        
        if max_samples is not None:
            self.metadata = self.metadata.iloc[:max_samples]
        
        print(f"[Dataset] Loaded {len(self.metadata)} samples")
        
        # Initialize TFI generator
        self.tfi_generator = SignalToSpectrogram(config)
        
    def _load_metadata(self, single_csv, dual_csv):
        """Load and merge metadata from single and dual component CSVs"""
        dfs = []
        
        # Load single component
        if os.path.exists(single_csv):
            df_single = pd.read_csv(single_csv)
            df_single['component_type'] = 'single'
            dfs.append(df_single)
            print(f"  - Single component: {len(df_single)} samples")
        
        # Load dual component
        if os.path.exists(dual_csv):
            df_dual = pd.read_csv(dual_csv)
            df_dual['component_type'] = 'dual'
            dfs.append(df_dual)
            print(f"  - Dual component: {len(df_dual)} samples")
        
        if not dfs:
            raise FileNotFoundError(f"No metadata found at {single_csv} or {dual_csv}")
        
        # Concatenate and shuffle
        df = pd.concat(dfs, ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        self.load_stats['total_attempts'] += 1
        
        row = self.metadata.iloc[idx]
        signal_path = self.data_root / row['signal_path']
        
        # Load signal
        try:
            signal = np.load(signal_path).flatten().astype(np.float32)
            self.load_stats['successes'] += 1
        except Exception as e:
            self.load_stats['failures'] += 1
            print(f"\n[WARNING] Failed to load {signal_path}: {e}")
            # Return zero signal as fallback
            signal = np.zeros(2048, dtype=np.float32)
        
        # Validate signal
        if len(signal) != 2048:
            self.load_stats['corrupted'] += 1
            if len(signal) > 2048:
                signal = signal[:2048]
            else:
                signal = np.pad(signal, (0, 2048 - len(signal)))
        
        # Convert to tensor
        signal = torch.from_numpy(signal).float()
        
        # Generate TFI on-the-fly
        try:
            spectrogram = self.tfi_generator(signal)  # (1, H, W)
        except Exception as e:
            print(f"\n[ERROR] TFI generation failed: {e}")
            spectrogram = torch.zeros(1, 256, 256)
        
        # Apply transforms (data augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        # Create target (16-dim one-hot)
        target = torch.zeros(16, dtype=torch.float32)
        
        # Fill modulation classes
        mod_classes = self.config['modulation_types']
        
        if pd.notna(row.get('modulation')):
            mod = row['modulation']
            if mod in mod_classes:
                target[mod_classes.index(mod)] = 1.0
        
        if pd.notna(row.get('modulation1')):
            mod1 = row['modulation1']
            if mod1 in mod_classes:
                target[mod_classes.index(mod1)] = 1.0
        
        if pd.notna(row.get('modulation2')):
            mod2 = row['modulation2']
            if mod2 in mod_classes:
                target[mod_classes.index(mod2)] = 1.0
        
        # Dual component flag (index 15)
        target[15] = 1.0 if row['component_type'] == 'dual' else 0.0
        
        # Track statistics
        snr = row.get('snr_db', 0)
        self.load_stats['snr_distribution'][snr] += 1
        
        # Track modulations for this sample
        if pd.notna(row.get('modulation')):
            self.load_stats['modulation_distribution'][row['modulation']] += 1
        if pd.notna(row.get('modulation1')):
            self.load_stats['modulation_distribution'][row['modulation1']] += 1
        if pd.notna(row.get('modulation2')):
            self.load_stats['modulation_distribution'][row['modulation2']] += 1
        
        return spectrogram, target, torch.tensor(snr, dtype=torch.float32)


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class SpectrogramTransform:
    """Composable transforms for spectrograms"""
    
    @staticmethod
    def random_horizontal_flip(img, p=0.5):
        if torch.rand(1) < p:
            return torch.flip(img, dims=[2])
        return img
    
    @staticmethod
    def random_vertical_flip(img, p=0.5):
        if torch.rand(1) < p:
            return torch.flip(img, dims=[1])
        return img
    
    @staticmethod
    def random_resized_crop(img, scale=(0.08, 1.0), ratio=(0.75, 1.25)):
        _, h, w = img.shape
        area = h * w
        
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()
            
            crop_h = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if crop_h <= h and crop_w <= w:
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()
                
                img = img[:, top:top+crop_h, left:left+crop_w]
                img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
                return img
        
        # Fallback: center crop
        crop_h = min(h, int(h * scale[0]))
        crop_w = min(w, int(w * scale[0]))
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        img = img[:, top:top+crop_h, left:left+crop_w]
        img = F.interpolate(img.unsqueeze(0), size=(h, w), mode='bilinear').squeeze(0)
        return img
    
    @staticmethod
    def random_erasing(img, p=0.33, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        if torch.rand(1) < p:
            _, h, w = img.shape
            area = h * w
            
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()
            
            erase_h = int(round(math.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if erase_h <= h and erase_w <= w:
                top = torch.randint(0, h - erase_h + 1, (1,)).item()
                left = torch.randint(0, w - erase_w + 1, (1,)).item()
                
                # Fill with random values from 0 to 0.5
                img[:, top:top+erase_h, left:left+erase_w] = torch.rand(1).item() * 0.5
        
        return img
    
    @staticmethod
    def time_stretch(img, max_stretch=0.1):
        """Time stretching augmentation (not in paper but useful)"""
        if torch.rand(1) < 0.3:
            stretch = 1.0 + (torch.rand(1).item() - 0.5) * 2 * max_stretch
            _, h, w = img.shape
            new_w = int(w * stretch)
            img = F.interpolate(img.unsqueeze(0), size=(h, new_w), mode='bilinear').squeeze(0)
            if new_w < w:
                pad = w - new_w
                img = F.pad(img, (0, pad))
            else:
                img = img[:, :, :w]
        return img



def get_train_transform(config):
    """Get training data augmentation pipeline using torchvision transforms"""
    aug_config = config['augmentation']
    
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=aug_config['hflip_prob']),
        transforms.RandomVerticalFlip(p=aug_config['vflip_prob']),
        transforms.RandomResizedCrop(
            size=(256, 256),
            scale=aug_config['random_crop_scale'],
            ratio=aug_config['random_crop_ratio']
        ),
        transforms.RandomErasing(
            p=aug_config['random_erase_prob'],
            scale=aug_config['random_erase_scale']
        ),
    ])

def get_val_transform(config):
    """Validation transform (no augmentation)"""
    return None


# ============================================================================
# TRAINING MONITOR
# ============================================================================

class TrainingMonitor:
    """
    Comprehensive training monitor that tracks:
    - Loss curves
    - Gradient statistics
    - Learning rate
    - GPU utilization
    - Sample TFI visualization
    - Per-SNR accuracy
    """
    
    def __init__(self, config, model, log_dir='./logs'):
        self.config = config
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # History storage
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'grad_norm': [],
            'grad_mean': [],
            'grad_std': [],
            'epoch_times': [],
            'snr_accuracy': {},
        }
        
        # GPU monitor
        self.gpu_monitor = GPUMonitor()
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_epoch = -1
        
        # Visualization counter
        self.viz_counter = 0
        
        print(f"\n[Monitor] Initialized - Logging to {log_dir}")
    
    def log_training_step(self, epoch, batch_idx, loss, lr, grad_norm=None, grad_stats=None):
        """Log per-batch metrics"""
        step = epoch * 1000 + batch_idx  # Approximate global step
        
        self.writer.add_scalar('train/loss_batch', loss, step)
        self.writer.add_scalar('train/lr', lr, step)
        
        if grad_norm is not None:
            self.writer.add_scalar('train/grad_norm', grad_norm, step)
        
        if grad_stats:
            self.writer.add_scalar('train/grad_mean', grad_stats['mean'], step)
            self.writer.add_scalar('train/grad_std', grad_stats['std'], step)
            self.writer.add_scalar('train/grad_max', grad_stats['max'], step)
            self.writer.add_scalar('train/grad_min', grad_stats['min'], step)
    
    def log_epoch(self, epoch, train_loss, val_loss, val_acc, lr, epoch_time):
        """Log epoch-level metrics"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        self.history['epoch_times'].append(epoch_time)
        
        self.writer.add_scalar('train/loss_epoch', train_loss, epoch)
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/accuracy', val_acc, epoch)
        self.writer.add_scalar('train/lr_epoch', lr, epoch)
        self.writer.add_scalar('train/epoch_time', epoch_time, epoch)
        
        # Log GPU stats
        gpu_stats = self.gpu_monitor.log_stats()
        for k, v in gpu_stats.items():
            self.writer.add_scalar(f'gpu/{k}', v, epoch)
        
        # Check if best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.writer.add_scalar('val/best_accuracy', val_acc, epoch)
    
    def log_snr_accuracy(self, snr_stats, epoch):
        """Log per-SNR accuracy"""
        for snr, acc in snr_stats.items():
            # Log to TensorBoard
            self.writer.add_scalar(f'snr/acc_{snr}dB', acc, epoch)
            
            # Store in history - need to initialize if not exists
            if 'snr_accuracy' not in self.history:
                self.history['snr_accuracy'] = {}
            
            if snr not in self.history['snr_accuracy']:
                self.history['snr_accuracy'][snr] = []
            
            self.history['snr_accuracy'][snr].append(acc)   

    def visualize_sample(self, spectrogram, prediction, target, snr, epoch, batch_idx):
        """Save sample TFI for visual inspection"""
        if self.viz_counter % self.config['viz_interval'] != 0:
            self.viz_counter += 1
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Spectrogram
        im = axes[0].imshow(spectrogram[0].cpu().numpy(), aspect='auto', cmap='hot')
        axes[0].set_title(f'Spectrogram (SNR: {snr:.1f} dB)')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Frequency')
        plt.colorbar(im, ax=axes[0])
        
        # Prediction heatmap
        pred_probs = torch.sigmoid(prediction).cpu().numpy()
        axes[1].bar(range(16), pred_probs)
        axes[1].set_title('Prediction')
        axes[1].set_xlabel('Class Index')
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1)
        
        # Target
        target_probs = target.cpu().numpy()
        axes[2].bar(range(16), target_probs)
        axes[2].set_title('Target')
        axes[2].set_xlabel('Class Index')
        axes[2].set_ylabel('Probability')
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save
        save_path = self.log_dir / f'viz_epoch{epoch}_batch{batch_idx}.png'
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        # Also log to TensorBoard
        self.writer.add_figure('sample/spectrogram', fig, epoch)
        
        self.viz_counter += 1
    
    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        if not self.history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r--', label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.history['val_acc'], 'g-')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[0, 2].plot(epochs, self.history['lr'], 'm-')
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('LR')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True)
        
        # Gradient Norm
        if self.history['grad_norm']:
            grad_epochs = range(1, len(self.history['grad_norm']) + 1)
            axes[1, 0].plot(grad_epochs, self.history['grad_norm'], 'orange')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Norm')
            axes[1, 0].grid(True)
        
        # SNR Accuracy
        if self.history['snr_accuracy']:
            snrs = sorted(self.history['snr_accuracy'].keys())
            for snr in snrs:
                acc_curve = self.history['snr_accuracy'][snr]
                if len(acc_curve) == len(epochs):
                    axes[1, 1].plot(epochs, acc_curve, label=f'{snr} dB')
            axes[1, 1].set_title('Per-SNR Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend(loc='best', fontsize=8)
            axes[1, 1].grid(True)
        
        # GPU Memory
        gpu_mem = self.gpu_monitor.history.get('memory_allocated', [])
        if gpu_mem:
            mem_epochs = range(1, len(gpu_mem) + 1)
            axes[1, 2].plot(mem_epochs, gpu_mem, 'c-')
            axes[1, 2].set_title('GPU Memory Usage')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Memory (GB)')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\n[Monitor] Training history saved to {save_path}")
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Best Validation Accuracy: {self.best_val_acc*100:.2f}% at epoch {self.best_epoch}")
        print(f"Total Epochs Trained: {len(self.history['train_loss'])}")
        print(f"Average Epoch Time: {np.mean(self.history['epoch_times']):.2f}s")
        print(f"Final Train Loss: {self.history['train_loss'][-1]:.5f}")
        print(f"Final Val Loss: {self.history['val_loss'][-1]:.5f}")
        print(f"Final Val Acc: {self.history['val_acc'][-1]*100:.2f}%")
        
        self.gpu_monitor.print_summary()
        print("="*70 + "\n")


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

class CheckpointManager:
    """Manages model checkpointing with resume capability"""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_path = Path(config['resume_path'])
        self.best_path = Path(config['best_model_path'])
        
        # Create directories
        self.checkpoint_path.parent.mkdir(exist_ok=True)
        self.best_path.parent.mkdir(exist_ok=True)
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, monitor, 
                        is_best=False, filename=None):
        """Save checkpoint with full state"""
        if filename is None:
            filename = self.checkpoint_path
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'history': monitor.history,
            'best_val_acc': monitor.best_val_acc,
            'best_epoch': monitor.best_epoch,
            'config': self.config,
            'variant': model.variant,
            'dims': model.dims,
        }
        
        # Save
        torch.save(checkpoint, filename)
        print(f"\n[Checkpoint] Saved to {filename}")
        
        if is_best:
            torch.save(checkpoint, self.best_path)
            print(f"[Checkpoint] Saved best model to {self.best_path}")
        
        # Also save lightweight version for quick resume
        light_path = filename.with_suffix('.light.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
        }, light_path)
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, device='cuda'):
        """Load checkpoint and return resume info"""
        if not os.path.exists(self.checkpoint_path):
            print(f"\n[Checkpoint] No checkpoint found at {self.checkpoint_path}")
            return {
                'start_epoch': 0,
                'history': {'train_loss': [], 'val_loss': [], 'val_acc': [], 
                           'lr': [], 'grad_norm': [], 'epoch_times': []},
                'best_val_acc': 0.0,
                'best_epoch': -1,
            }
        
        print(f"\n[Checkpoint] Loading from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=device)
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  - Optimizer state loaded")
        
        # Load scheduler if provided
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  - Scheduler state loaded")
        
        # Get resume info
        resume_info = {
            'start_epoch': checkpoint.get('epoch', 0),
            'history': checkpoint.get('history', {
                'train_loss': [], 'val_loss': [], 'val_acc': [], 
                'lr': [], 'grad_norm': [], 'epoch_times': []
            }),
            'best_val_acc': checkpoint.get('best_val_acc', 0.0),
            'best_epoch': checkpoint.get('best_epoch', -1),
        }
        
        print(f"  - Resuming from epoch {resume_info['start_epoch']}")
        print(f"  - Best val acc so far: {resume_info['best_val_acc']*100:.2f}%")
        
        return resume_info


# ============================================================================
# GRADIENT MONITOR
# ============================================================================

def compute_gradient_stats(model):
    """Compute statistics of gradients"""
    grad_norms = []
    grad_values = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            grad_values.extend(param.grad.abs().flatten().tolist())
    
    if not grad_values:
        return None
    
    grad_values = np.array(grad_values)
    
    stats = {
        'norm': np.mean(grad_norms) if grad_norms else 0,
        'mean': grad_values.mean(),
        'std': grad_values.std(),
        'max': grad_values.max(),
        'min': grad_values.min(),
        'p90': np.percentile(grad_values, 90),
        'p10': np.percentile(grad_values, 10),
    }
    
    return stats


def check_gradient_health(grad_stats):
    """Check if gradients are healthy"""
    if grad_stats is None:
        return "No gradients"
    
    issues = []
    
    if grad_stats['mean'] < 1e-7:
        issues.append("Gradients too small - vanishing?")
    
    if grad_stats['max'] > 10:
        issues.append("Gradients exploding!")
    
    if grad_stats['p90'] / (grad_stats['p10'] + 1e-8) > 100:
        issues.append("Gradient distribution highly skewed")
    
    return issues if issues else "Healthy"


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, scheduler, 
                monitor, epoch, config, scaler=None):
    """
    Train for one epoch with comprehensive monitoring
    """
    model.train()
    running_loss = 0.0
    num_batches = len(dataloader)
    
    # For gradient monitoring
    grad_check_interval = config['grad_check_interval']
    grad_norms = []
    
    print(f"\n[Epoch {epoch+1}/{config['epochs']}] Training...")
    
    for batch_idx, (inputs, targets, snrs) in enumerate(dataloader):
        # Move to device
        inputs = inputs.to(config['device'])
        targets = targets.to(config['device'])
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision if enabled
        if scaler is not None:
            # with autocast(): # deprecated 
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Monitor gradients periodically
        if batch_idx % grad_check_interval == 0:
            grad_stats = compute_gradient_stats(model)
            if grad_stats:
                grad_norms.append(grad_stats['norm'])
                health = check_gradient_health(grad_stats)
                
                # Log to monitor
                monitor.log_training_step(
                    epoch, batch_idx, loss.item(), 
                    scheduler.get_last_lr()[0] if scheduler else config['lr'],
                    grad_stats['norm'], grad_stats
                )
                
                if batch_idx % (grad_check_interval * 5) == 0:
                    print(f"  Batch {batch_idx}/{num_batches} | Loss: {loss.item():.4f} | Grad: {health}")
        
        # Visualize sample
        if batch_idx % config['viz_interval'] == 0 and batch_idx > 0:
            with torch.no_grad():
                monitor.visualize_sample(
                    inputs[0], outputs[0], targets[0], snrs[0].item(),
                    epoch, batch_idx
                )
    
    # Step scheduler if it's epoch-based
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        scheduler.step()
    
    # Compute epoch loss
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Log average gradient norm
    if grad_norms:
        monitor.history['grad_norm'].append(np.mean(grad_norms))
    
    return epoch_loss


def validate(model, dataloader, criterion, config, monitor, epoch):
    """
    Validate model with per-SNR breakdown
    """
    model.eval()
    val_loss = 0.0
    correct_exact = 0
    total_samples = 0
    
    # Per-SNR statistics
    snr_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    print(f"[Epoch {epoch+1}/{config['epochs']}] Validating...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets, snrs) in enumerate(dataloader):
            inputs = inputs.to(config['device'])
            targets = targets.to(config['device'])
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item() * inputs.size(0)
            
            # Compute exact match accuracy
            preds = (torch.sigmoid(outputs) > 0.5).float()
            matches = (preds == targets).all(dim=1)
            correct_exact += matches.sum().item()
            total_samples += inputs.size(0)
            
            # Update SNR stats
            for i, snr in enumerate(snrs):
                snr_val = int(snr.item())
                snr_stats[snr_val]['total'] += 1
                if matches[i].item():
                    snr_stats[snr_val]['correct'] += 1
    
    # Compute metrics
    avg_loss = val_loss / len(dataloader.dataset)
    accuracy = correct_exact / total_samples
    
    # Compute per-SNR accuracy
    snr_accuracy = {}
    for snr, stats in snr_stats.items():
        if stats['total'] > 0:
            snr_accuracy[snr] = stats['correct'] / stats['total']
    
    # Log to monitor
    monitor.log_snr_accuracy(snr_accuracy, epoch)
    
    # Print SNR breakdown
    print(f"\n  SNR Breakdown (Accuracy):")
    for snr in sorted(snr_stats.keys()):
        acc = (snr_stats[snr]['correct'] / snr_stats[snr]['total']) * 100
        print(f"    SNR {snr:>3} dB : {acc:6.2f}%  (Count: {snr_stats[snr]['total']})")
    
    return avg_loss, accuracy


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*80)
    print("CONVNEXT FOR SIGNAL MODULATION CLASSIFICATION")
    print("="*80)
    
    # Print hardware info
    print(f"\nHardware Configuration:")
    print(f"  Device: {CONFIG['device']}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"  Batch Size: {CONFIG['batch_size']}")
        print(f"  Mixed Precision: {CONFIG['use_amp']}")
    print(f"  Num Workers: {CONFIG['num_workers']}")
    
    # ==========================================
    # 1. DATASET PREPARATION
    # ==========================================
    print("\n" + "-"*40)
    print("STEP 1: Preparing Dataset")
    print("-"*40)
    
    # Create datasets
    train_dataset = SignalDataset(
        single_csv=CONFIG['single_csv'],
        dual_csv=CONFIG['dual_csv'],
        data_root=CONFIG['data_root'],
        config=CONFIG,
        split='train',
        transform=get_train_transform(CONFIG)
    )
    
    val_dataset = SignalDataset(
        single_csv=CONFIG['single_csv'],
        dual_csv=CONFIG['dual_csv'],
        data_root=CONFIG['data_root'],
        config=CONFIG,
        split='val',
        transform=get_val_transform(CONFIG)
    )
    
    # Create dataloaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'] * 2,  # Larger batches for validation
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'],
        persistent_workers=True if CONFIG['num_workers'] > 0 else False
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Training batches/epoch: {len(train_loader)}")
    print(f"  Validation batches/epoch: {len(val_loader)}")
    
    # Check dataset health
    if train_dataset.load_stats['failures'] > 0:
        fail_rate = train_dataset.load_stats['failures'] / train_dataset.load_stats['total_attempts']
        if fail_rate > 0.05:
            print(f"\n[WARNING] High data load failure rate: {fail_rate*100:.1f}%")
    
    # ==========================================
    # 2. MODEL INITIALIZATION
    # ==========================================
    print("\n" + "-"*40)
    print("STEP 2: Initializing Model")
    print("-"*40)
    
    model = ConvNeXtSignalClassifier(
        variant=CONFIG['variant'],
        num_classes=CONFIG['num_classes']
    ).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: ConvNeXt-{CONFIG['variant']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Calculate FLOPs
    try:
        flops = model.get_flops()
        print(f"  FLOPs: {flops:.2f} GFLOPs")
    except:
        pass
    
    # ==========================================
    # 3. OPTIMIZER & SCHEDULER
    # ==========================================
    print("\n" + "-"*40)
    print("STEP 3: Setting up Optimizer & Scheduler")
    print("-"*40)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        betas=CONFIG['betas'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Cosine annealing with warmup
    warmup_epochs = 5
    total_epochs = CONFIG['epochs']
    
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=CONFIG['use_amp']) if CONFIG['use_amp'] else None
    
    print(f"  Optimizer: AdamW (lr={CONFIG['lr']}, weight_decay={CONFIG['weight_decay']})")
    print(f"  Scheduler: Cosine annealing with {warmup_epochs}-epoch warmup")
    print(f"  Mixed Precision: {CONFIG['use_amp']}")
    
    # ==========================================
    # 4. CHECKPOINT LOADING
    # ==========================================
    print("\n" + "-"*40)
    print("STEP 4: Loading Checkpoint (if exists)")
    print("-"*40)
    
    checkpoint_manager = CheckpointManager(CONFIG)
    resume_info = checkpoint_manager.load_checkpoint(
        model, optimizer, scheduler, CONFIG['device']
    )
    
    start_epoch = resume_info['start_epoch']
    monitor = TrainingMonitor(CONFIG, model)
    monitor.history = resume_info['history']
    monitor.best_val_acc = resume_info['best_val_acc']
    monitor.best_epoch = resume_info['best_epoch']
    
    # ==========================================
    # 5. TRAINING LOOP
    # ==========================================
    print("\n" + "-"*40)
    print("STEP 5: Starting Training")
    print("-"*40)
    
    total_start_time = time.time()
    
    for epoch in range(start_epoch, CONFIG['epochs']):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            monitor, epoch, CONFIG, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG, monitor, epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log epoch
        epoch_time = time.time() - epoch_start_time
        monitor.log_epoch(
            epoch, train_loss, val_loss, val_acc,
            scheduler.get_last_lr()[0], epoch_time
        )
        
        # Print epoch summary
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} Summary ---")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.5f}")
        print(f"  Val Loss: {val_loss:.5f}")
        print(f"  Val Accuracy: {val_acc*100:.2f}%")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        if (epoch + 1) % CONFIG['save_interval'] == 0 or (epoch + 1) == CONFIG['epochs']:
            is_best = val_acc > monitor.best_val_acc
            if is_best:
                monitor.best_val_acc = val_acc
                monitor.best_epoch = epoch
            
            checkpoint_manager.save_checkpoint(
                epoch + 1, model, optimizer, scheduler, monitor,
                is_best=is_best
            )
    
    # ==========================================
    # 6. FINAL SUMMARY
    # ==========================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal Training Time: {total_time/3600:.2f} hours")
    
    monitor.print_summary()
    monitor.plot_history('final_training_history.png')
    
    # Save final model
    torch.save(model.state_dict(), './checkpoints/final_model.pth')
    print(f"\nFinal model saved to ./checkpoints/final_model.pth")


if __name__ == "__main__":
    main()