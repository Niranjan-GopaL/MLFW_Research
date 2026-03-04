"""
Signal Modulation Dataset Generator
Comprehensive system for generating and managing synthetic signal dataset
"""

import numpy as np
import pandas as pd
import os
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')



# These are our    "Control Knobs"
CONFIG = {
    # Base parameters
    "signal_length": 2048,
    "sampling_frequency": 100e6,    # 100 MHz
    "pulse_width": 20.48e-6,        # 20.48 µs
    
    # Single-component dataset configuration
    "single_component": {
        "train": {
            "parameter_combinations": 800,
            "noise_levels": list(range(-12, 19, 3)),  # -12 to 18 dB, step 3
            "num_noise_levels": 11,
            "num_modulations": 15,
            "total_signals": 132000
        },
        "val": {
            "parameter_combinations": 200,
            "noise_levels": list(range(-12, 19, 3)),
            "num_noise_levels": 11,
            "num_modulations": 15,
            "total_signals": 33000
        },
        "test": {
            "parameter_combinations": 400,
            "noise_levels": list(range(-21, 19, 3)),  # -21 to 18 dB, step 3
            "num_noise_levels": 14,
            "num_modulations": 15,
            "total_signals": 84000
        }
    },
    
    # Dual-component dataset configuration
    "dual_component": {
        "train": {
            "parameter_combinations_per_mod": 25,
            "noise_levels": list(range(-12, 19, 3)),
            "num_noise_levels": 11,
            "num_modulation_combinations": 120,  # 15*(15-1)/2 + 15
            "total_signals": 825000
        },
        "val": {
            "parameter_combinations_per_mod": 10,
            "noise_levels": list(range(-12, 19, 3)),
            "num_noise_levels": 11,
            "num_modulation_combinations": 120,
            "total_signals": 132000
        },
        "test": {
            "parameter_combinations_per_mod": 15,
            "noise_levels": list(range(-21, 19, 3)),
            "num_noise_levels": 14,
            "num_modulation_combinations": 120,
            "total_signals": 378000
        }
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
    
    # Modulation types (order as in paper)
    "modulation_types": [
        "NM", "LFM", "DLFM", "MLFM", "EQFM", "SFM", 
        "BFSK", "QFSK", "BPSK", "Frank", "P1", "P2", "P3", "P4", "LFM_BPSK"
    ],
    
    # STFT preprocessing parameters
    "stft_params": {
        "window": "hann",
        "nperseg": 101,
        "noverlap": 101 - 8,  # stride of 8
        "nfft": 512,
        "tf_shape": (256, 256)  # after removing negative frequencies
    },
    
    # File storage format
    "storage": {
        "signal_dtype": np.float32,
        "chunk_size": 1000,  # signals per chunk for memory management
        "checkpoint_interval": 10000  # save checkpoint every N signals
    }
}




# ============================================================================
# SIGNAL GENERATION CLASSES
# ============================================================================

class SignalGenerator:
    """Generates signals with various modulations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fs = config["sampling_frequency"]
        self.N = config["signal_length"]
        self.t = np.arange(self.N) / self.fs
        
    def generate_signal(self, modulation: str, params: Dict) -> np.ndarray:
        """Generate a single signal with given modulation and parameters"""
        if modulation == "NM":
            return self._generate_nm(params)
        elif modulation == "LFM":
            return self._generate_lfm(params)
        elif modulation == "DLFM":
            return self._generate_dlfm(params)
        elif modulation == "MLFM":
            return self._generate_mlfm(params)
        elif modulation == "EQFM":
            return self._generate_eqfm(params)
        elif modulation == "SFM":
            return self._generate_sfm(params)
        elif modulation == "BFSK":
            return self._generate_bfsk(params)
        elif modulation == "QFSK":
            return self._generate_qfsk(params)
        elif modulation == "BPSK":
            return self._generate_bpsk(params)
        elif modulation == "Frank":
            return self._generate_frank(params)
        elif modulation == "P1":
            return self._generate_p1(params)
        elif modulation == "P2":
            return self._generate_p2(params)
        elif modulation == "P3":
            return self._generate_p3(params)
        elif modulation == "P4":
            return self._generate_p4(params)
        elif modulation == "LFM_BPSK":
            return self._generate_lfm_bpsk(params)
        else:
            raise ValueError(f"Unknown modulation: {modulation}")
    
    def _generate_nm(self, params: Dict) -> np.ndarray:
        # """Noise-like modulation (constant frequency)""" <---- GPT bad
        """NO modulation (constant frequency)"""
        f0 = params["f0"] * self.fs
        return np.exp(1j * 2 * np.pi * f0 * self.t)
    
    def _generate_lfm(self, params: Dict) -> np.ndarray:
        """Linear Frequency Modulation"""
        # Calculate f0 from available parameters
        if "f0" in params:
            f0 = params["f0"] * self.fs
        elif "fmin" in params:
            # f0 = fmin + delta_f/2 as per paper
            f0 = params["fmin"] * self.fs
            if "delta_f" in params:
                f0 += params["delta_f"] * self.fs / 2
        else:
            raise ValueError("LFM requires either f0 or fmin parameter")
        
        delta_f = params["delta_f"] * self.fs
        direction = params.get("direction", 1)  # 1 for increasing, -1 for decreasing
        
        # Linear chirp: f(t) = f0 + (delta_f / T) * t * direction
        f_inst = f0 + direction * delta_f * self.t / self.t[-1]
        phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
        return np.exp(1j * phase)
    
    def _generate_dlfm(self, params: Dict) -> np.ndarray:
        """Double Linear Frequency Modulation"""
        # Simplified version - similar to LFM
        return self._generate_lfm(params)
    
    def _generate_mlfm(self, params: Dict) -> np.ndarray:
        """Multi-segment Linear Frequency Modulation"""
        f0 = params["f0"] * self.fs
        delta_f = params["delta_f"] * self.fs
        r = params["r"]
        
        # Create two segments
        split_idx = int(self.N * r)
        t1 = self.t[:split_idx]
        t2 = self.t[split_idx:]
        
        # First segment: increasing frequency
        f1 = f0 + delta_f * t1 / t1[-1] if len(t1) > 0 else np.array([])
        # Second segment: decreasing frequency
        f2 = f0 + delta_f * (1 - t2 / t2[-1]) if len(t2) > 0 else np.array([])
        
        f_inst = np.concatenate([f1, f2])
        phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
        return np.exp(1j * phase)
    
    def _generate_eqfm(self, params: Dict) -> np.ndarray:
        """Equi-ripple Frequency Modulation"""
        fmin = params["fmin"] * self.fs
        delta_f = params["delta_f"] * self.fs
        
        # Simplified: sinusoidal frequency variation
        f_inst = fmin + delta_f * (1 + np.sin(2 * np.pi * 5 * self.t / self.t[-1])) / 2
        phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
        return np.exp(1j * phase)
    
    def _generate_sfm(self, params: Dict) -> np.ndarray:
        """Sinusoidal Frequency Modulation"""
        fmin = params["fmin"] * self.fs
        delta_f = params["delta_f"] * self.fs
        fSFM = params["fSFM"] / self.config["pulse_width"]  # Convert to Hz
        phiSFM = params["phiSFM"]
        
        # Sinusoidal frequency modulation
        f_inst = fmin + delta_f * np.sin(2 * np.pi * fSFM * self.t + phiSFM)
        phase = 2 * np.pi * np.cumsum(f_inst) / self.fs
        return np.exp(1j * phase)
    
    def _generate_bfsk(self, params: Dict) -> np.ndarray:
        """Binary Frequency Shift Keying"""
        f1 = params["f1"] * self.fs
        f2 = params["f2"] * self.fs
        N = params["N"]
        
        # Generate Barker code
        code = self._generate_barker_code(N, params.get("inverted", False))
        
        # Create signal with frequency shifts
        symbols_per_sample = self.N // N
        signal = np.zeros(self.N, dtype=complex)
        
        for i, symbol in enumerate(code):
            start = i * symbols_per_sample
            end = (i + 1) * symbols_per_sample if i < N - 1 else self.N
            freq = f1 if symbol == 1 else f2
            signal[start:end] = np.exp(1j * 2 * np.pi * freq * self.t[start:end])
        
        return signal
    
    def _generate_qfsk(self, params: Dict) -> np.ndarray:
        """Quadrature Frequency Shift Keying"""
        # Simplified version
        f0 = np.random.uniform(0.05, 0.45) * self.fs
        return np.exp(1j * 2 * np.pi * f0 * self.t)
    
    def _generate_bpsk(self, params: Dict) -> np.ndarray:
        """Binary Phase Shift Keying"""
        f0 = params["f0"] * self.fs
        N = params["N"]
        
        # Generate Barker code
        code = self._generate_barker_code(N, params.get("inverted", False))
        
        # Create signal with phase shifts
        symbols_per_sample = self.N // N
        signal = np.zeros(self.N, dtype=complex)
        phase = 0
        
        for i, symbol in enumerate(code):
            start = i * symbols_per_sample
            end = (i + 1) * symbols_per_sample if i < N - 1 else self.N
            if symbol == -1:
                phase += np.pi  # Phase shift
            signal[start:end] = np.exp(1j * (2 * np.pi * f0 * self.t[start:end] + phase))
        
        return signal
    
    def _generate_frank(self, params: Dict) -> np.ndarray:
        """Frank code modulation"""
        f0 = params["f0"] * self.fs
        N = params["N"]
        # Simplified version
        return np.exp(1j * 2 * np.pi * f0 * self.t)
    
    def _generate_p1(self, params: Dict) -> np.ndarray:
        """P1 code modulation"""
        return self._generate_frank(params)
    
    def _generate_p2(self, params: Dict) -> np.ndarray:
        """P2 code modulation"""
        return self._generate_frank(params)
    
    def _generate_p3(self, params: Dict) -> np.ndarray:
        """P3 code modulation"""
        return self._generate_frank(params)
    
    def _generate_p4(self, params: Dict) -> np.ndarray:
        """P4 code modulation"""
        return self._generate_frank(params)
    
    def _generate_lfm_bpsk(self, params: Dict) -> np.ndarray:
        """Combined LFM and BPSK"""
        # Generate LFM component using fmin and delta_f
        lfm_params = {
            "fmin": params["fmin"],
            "delta_f": params["delta_f"],
            "direction": params.get("direction", 1)
        }
        lfm_signal = self._generate_lfm(lfm_params)
        
        # Generate BPSK component
        # Calculate f0 from fmin and delta_f
        f0_norm = params["fmin"] + params["delta_f"] / 2
        bpsk_params = {
            "f0": f0_norm, 
            "N": params["N"],
            "inverted": params.get("inverted", False)
        }
        bpsk_signal = self._generate_bpsk(bpsk_params)
        
        # Combine: BPSK modulates the phase of LFM
        # In reality, this would be more complex, but this is a simplified version
        return lfm_signal * bpsk_signal
    
    def _generate_barker_code(self, N: int, inverted: bool = False) -> np.ndarray:
        """Generate Barker code of given length"""
        barker_codes = {
            5: [1, 1, 1, -1, 1],
            7: [1, 1, 1, -1, -1, 1, -1],
            11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        }
        
        if N not in barker_codes:
            raise ValueError(f"No Barker code defined for length {N}")
        
        code = np.array(barker_codes[N])
        if inverted:
            code = -code
        
        return code
    
    def add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add AWGN noise to signal with given SNR"""
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        
        # Calculate noise power for given SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex AWGN noise
        noise_real = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), len(signal))
        noise = noise_real + 1j * noise_imag
        
        return signal + noise

# ============================================================================
# DATASET MANAGER
# ============================================================================

class DatasetManager:
    """Manages dataset generation, storage, and retrieval"""
    
    def __init__(self, base_dir: str = "./signal_dataset"):
        self.base_dir = Path(base_dir)
        self.config = CONFIG
        self.generator = SignalGenerator(CONFIG)
        
        # Create directory structure
        self._create_directories()
        
        # Initialize metadata tracking
        self.metadata = {
            "single": {"train": {}, "val": {}, "test": {}},
            "dual": {"train": {}, "val": {}, "test": {}}
        }
    
    def _create_directories(self):
        """Create the directory structure"""
        # Single component directories
        for split in ["train", "val", "test"]:
            split_dir = self.base_dir / "single" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata directory
            meta_dir = split_dir / "metadata"
            meta_dir.mkdir(exist_ok=True)
        
        # Dual component directories
        for split in ["train", "val", "test"]:
            split_dir = self.base_dir / "dual" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata directory
            meta_dir = split_dir / "metadata"
            meta_dir.mkdir(exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = self.base_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
    
    def generate_single_component_dataset(self, split: str = "train", 
                                        use_checkpoint: bool = True):
        """Generate single component dataset for given split"""
        print(f"Generating single component {split} dataset...")
        
        config = self.config["single_component"][split]
        mod_types = self.config["modulation_types"]
        
        # Load checkpoint if exists
        checkpoint_file = self.base_dir / "checkpoints" / f"single_{split}_checkpoint.pkl"
        if use_checkpoint and checkpoint_file.exists():
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = self._load_checkpoint(checkpoint_file)
            start_mod_idx = checkpoint.get("mod_idx", 0)
            start_param_idx = checkpoint.get("param_idx", 0)
        else:
            start_mod_idx = 0
            start_param_idx = 0
        
        total_signals = 0
        global_metadata = []
        
        # Generate for each modulation type
        for mod_idx, modulation in enumerate(mod_types):
            if mod_idx < start_mod_idx:
                continue
                
            print(f"  Processing modulation {modulation} ({mod_idx + 1}/{len(mod_types)})")
            
            mod_metadata = []
            param_combinations = config["parameter_combinations"]
            
            # Generate parameter combinations for this modulation
            for param_idx in range(param_combinations):
                if mod_idx == start_mod_idx and param_idx < start_param_idx:
                    continue
                
                # Generate random parameters
                params = self._generate_random_params(modulation)
                
                # Generate clean signal
                clean_signal = self.generator.generate_signal(modulation, params)
                
                # Save signals for each noise level
                for snr_db in config["noise_levels"]:
                    # Add noise
                    noisy_signal = self.generator.add_noise(clean_signal, snr_db)
                    
                    # Take real part (as per paper)
                    real_signal = np.real(noisy_signal).astype(np.float32)
                    
                    # Save signal
                    signal_id = f"{modulation}_{param_idx:04d}_{snr_db:+03d}dB"
                    signal_path = self.base_dir / "single" / split / "signals" / f"{signal_id}.npy"
                    signal_path.parent.mkdir(exist_ok=True)
                    
                    np.save(signal_path, real_signal)
                    
                    # Create metadata entry
                    metadata_entry = {
                        "signal_id": signal_id,
                        "signal_path": str(signal_path.relative_to(self.base_dir)),
                        "modulation": modulation,
                        "modulation_idx": mod_idx,
                        "snr_db": snr_db,
                        "params": params,
                        "is_dual_component": 0,
                        "one_hot_label": self._get_one_hot_label(modulation, is_dual=False)
                    }
                    
                    mod_metadata.append(metadata_entry)
                    global_metadata.append(metadata_entry)
                    total_signals += 1
                
                # Save checkpoint periodically
                if total_signals % self.config["storage"]["checkpoint_interval"] == 0:
                    checkpoint_data = {
                        "mod_idx": mod_idx,
                        "param_idx": param_idx + 1,
                        "total_signals": total_signals
                    }
                    self._save_checkpoint(checkpoint_file, checkpoint_data)
                    print(f"    Checkpoint saved at {total_signals} signals")
            
            # Save modulation-specific metadata
            mod_meta_path = (self.base_dir / "single" / split / "metadata" / 
                           f"{modulation}_metadata.csv")
            pd.DataFrame(mod_metadata).to_csv(mod_meta_path, index=False)
            
            # Reset start_param_idx for next modulation
            start_param_idx = 0
        
        # Save global metadata
        global_meta_path = self.base_dir / "single" / split / "metadata" / "global_metadata.csv"
        pd.DataFrame(global_metadata).to_csv(global_meta_path, index=False)
        
        print(f"Generated {total_signals} single-component signals for {split}")
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def generate_dual_component_dataset(self, split: str = "train", 
                                      use_checkpoint: bool = True):
        """Generate dual component dataset for given split"""
        print(f"Generating dual component {split} dataset...")
        
        config = self.config["dual_component"][split]
        mod_types = self.config["modulation_types"]
        
        # Generate all modulation combinations
        mod_combinations = []
        for i in range(len(mod_types)):
            for j in range(i, len(mod_types)):  # Include same modulation pairs
                mod_combinations.append((mod_types[i], mod_types[j]))
        
        # Load checkpoint if exists
        checkpoint_file = self.base_dir / "checkpoints" / f"dual_{split}_checkpoint.pkl"
        if use_checkpoint and checkpoint_file.exists():
            print(f"Loading checkpoint from {checkpoint_file}")
            checkpoint = self._load_checkpoint(checkpoint_file)
            start_combo_idx = checkpoint.get("combo_idx", 0)
            start_param_idx1 = checkpoint.get("param_idx1", 0)
            start_param_idx2 = checkpoint.get("param_idx2", 0)
        else:
            start_combo_idx = 0
            start_param_idx1 = 0
            start_param_idx2 = 0
        
        total_signals = 0
        global_metadata = []
        
        # Generate for each modulation combination
        for combo_idx, (mod1, mod2) in enumerate(mod_combinations):
            if combo_idx < start_combo_idx:
                continue
            
            print(f"  Processing combination {mod1}+{mod2} ({combo_idx + 1}/{len(mod_combinations)})")
            
            combo_metadata = []
            param_per_mod = config["parameter_combinations_per_mod"]
            
            # Load or generate single component signals for mod1
            mod1_signals = self._load_or_generate_single_signals(
                mod1, param_per_mod, split
            )
            
            # Load or generate single component signals for mod2
            mod2_signals = self._load_or_generate_single_signals(
                mod2, param_per_mod, split
            )
            
            # Combine signals
            for i in range(param_per_mod):
                if combo_idx == start_combo_idx and i < max(start_param_idx1, start_param_idx2):
                    continue
                
                for j in range(param_per_mod):
                    if combo_idx == start_combo_idx and i == start_param_idx1 and j < start_param_idx2:
                        continue
                    
                    # Get amplitude ratio
                    alpha = np.random.uniform(0.25, 0.5)
                    
                    # Combine signals with different amplitudes
                    for snr_db in config["noise_levels"]:
                        # Get clean signals (remove noise from single components)
                        clean_signal1 = mod1_signals[i]["clean"]
                        clean_signal2 = mod2_signals[j]["clean"]
                        
                        # Apply amplitudes
                        signal1 = alpha * clean_signal1
                        signal2 = (1 - alpha) * clean_signal2
                        
                        # Combine
                        combined_clean = signal1 + signal2
                        
                        # Add noise
                        noisy_signal = self.generator.add_noise(combined_clean, snr_db)
                        real_signal = np.real(noisy_signal).astype(np.float32)
                        
                        # Save signal
                        signal_id = f"{mod1}_{mod2}_{i:03d}_{j:03d}_{snr_db:+03d}dB"
                        signal_path = self.base_dir / "dual" / split / "signals" / f"{signal_id}.npy"
                        signal_path.parent.mkdir(exist_ok=True)
                        
                        np.save(signal_path, real_signal)
                        
                        # Create metadata
                        is_same_mod = 1 if mod1 == mod2 else 0
                        metadata_entry = {
                            "signal_id": signal_id,
                            "signal_path": str(signal_path.relative_to(self.base_dir)),
                            "modulation1": mod1,
                            "modulation2": mod2,
                            "modulation_idx1": self.config["modulation_types"].index(mod1),
                            "modulation_idx2": self.config["modulation_types"].index(mod2),
                            "snr_db": snr_db,
                            "alpha": alpha,
                            "is_dual_component": 1,
                            "is_same_modulation": is_same_mod,
                            "one_hot_label": self._get_one_hot_label([mod1, mod2], is_dual=True)
                        }
                        
                        combo_metadata.append(metadata_entry)
                        global_metadata.append(metadata_entry)
                        total_signals += 1
                    
                    # Save checkpoint periodically
                    if total_signals % self.config["storage"]["checkpoint_interval"] == 0:
                        checkpoint_data = {
                            "combo_idx": combo_idx,
                            "param_idx1": i,
                            "param_idx2": j + 1,
                            "total_signals": total_signals
                        }
                        self._save_checkpoint(checkpoint_file, checkpoint_data)
                        print(f"    Checkpoint saved at {total_signals} signals")
            
            # Save combination metadata
            combo_meta_path = (self.base_dir / "dual" / split / "metadata" / 
                             f"{mod1}_{mod2}_metadata.csv")
            pd.DataFrame(combo_metadata).to_csv(combo_meta_path, index=False)
        
        # Save global metadata
        global_meta_path = self.base_dir / "dual" / split / "metadata" / "global_metadata.csv"
        pd.DataFrame(global_metadata).to_csv(global_meta_path, index=False)
        
        print(f"Generated {total_signals} dual-component signals for {split}")
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
    
    def _load_or_generate_single_signals(self, modulation: str, num_signals: int, 
                                       split: str) -> List[Dict]:
        """Load or generate single component signals"""
        signals = []
        
        # Try to load from existing single component dataset
        meta_path = self.base_dir / "single" / split / "metadata" / f"{modulation}_metadata.csv"
        
        if meta_path.exists():
            metadata = pd.read_csv(meta_path)
            # Get signals with 0 dB SNR (clean)
            clean_metadata = metadata[metadata["snr_db"] == 0].head(num_signals)
            
            for _, row in clean_metadata.iterrows():
                signal_path = self.base_dir / row["signal_path"]
                if signal_path.exists():
                    signal = np.load(signal_path)
                    # Reconstruct clean signal (approximate)
                    params = eval(row["params"]) if isinstance(row["params"], str) else row["params"]
                    clean_signal = self.generator.generate_signal(modulation, params)
                    signals.append({"clean": clean_signal, "params": params})
        
        # If not enough signals, generate new ones
        while len(signals) < num_signals:
            params = self._generate_random_params(modulation)
            clean_signal = self.generator.generate_signal(modulation, params)
            signals.append({"clean": clean_signal, "params": params})
        
        return signals[:num_signals]
    
    def _generate_random_params(self, modulation: str) -> Dict:
        """Generate random parameters for given modulation"""
        param_ranges = self.config["modulation_params"][modulation]
        params = {}
        
        for key, value in param_ranges.items():
            if key.endswith("_range"):
                param_name = key.replace("_range", "")
                if isinstance(value, tuple):
                    params[param_name] = np.random.uniform(value[0], value[1])
                else:
                    params[param_name] = value
            elif key.endswith("_values"):
                param_name = key.replace("_values", "")
                params[param_name] = np.random.choice(value)
            else:
                params[key] = value
        
        # Add direction for LFM
        if modulation == "LFM":
            params["direction"] = np.random.choice([-1, 1])
        
        # Add inverted flag for codes
        if modulation in ["BFSK", "BPSK", "Frank", "P1", "P2", "P3", "P4"]:
            params["inverted"] = np.random.choice([True, False])
        
        return params
    
    def _get_one_hot_label(self, modulation: Union[str, List[str]], is_dual: bool) -> str:
        """Get one-hot encoded label (16-dimensional vector as string)"""
        label = [0] * 16
        
        if not is_dual:
            # Single component
            mod_idx = self.config["modulation_types"].index(modulation)
            label[mod_idx] = 1
            label[15] = 0  # Not dual component
        else:
            # Dual component
            mod1_idx = self.config["modulation_types"].index(modulation[0])
            mod2_idx = self.config["modulation_types"].index(modulation[1])
            label[mod1_idx] = 1
            label[mod2_idx] = 1
            label[15] = 1  # Is dual component
        
        return str(label)
    
    def _save_checkpoint(self, checkpoint_file: Path, data: Dict):
        """Save checkpoint data"""
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Dict:
        """Load checkpoint data"""
        with open(checkpoint_file, 'rb') as f:
            return pickle.load(f)

# ============================================================================
# TFI GENERATOR
# ============================================================================

class TFIGenerator:
    """Generates Time-Frequency Images from signals"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stft_params = config["stft_params"]
        
    def signal_to_tfi(self, signal: np.ndarray) -> np.ndarray:
        """Convert signal to Time-Frequency Image"""
        from scipy.signal import stft
        
        # Compute STFT
        f, t, Zxx = stft(
            signal,
            fs=self.config["sampling_frequency"],
            window=self.stft_params["window"],
            nperseg=self.stft_params["nperseg"],
            noverlap=self.stft_params["noverlap"],
            nfft=self.stft_params["nfft"],
            return_onesided=False
        )
        
        # Get magnitude
        magnitude = np.abs(Zxx)
        
        # Remove negative frequencies (keep only positive)
        positive_freq_idx = len(f) // 2
        magnitude = magnitude[positive_freq_idx:, :]
        
        # Resize to target shape
        from scipy.ndimage import zoom
        target_shape = self.stft_params["tf_shape"]
        zoom_factors = (
            target_shape[0] / magnitude.shape[0],
            target_shape[1] / magnitude.shape[1]
        )
        tfi = zoom(magnitude, zoom_factors)
        
        # Normalize sample-wise
        tfi = tfi / np.max(tfi) if np.max(tfi) > 0 else tfi
        
        return tfi.astype(np.float32)
    
    def generate_tfi_from_paths(self, path_index_pairs: List[Tuple[str, int]]):
        """Generate TFI plots from signal file paths"""
        import matplotlib.pyplot as plt
        
        tfis = []
        for path, idx in path_index_pairs:
            # Load signal
            signal = np.load(path)
            
            # Generate TFI
            tfi = self.signal_to_tfi(signal)
            tfis.append((path, tfi))
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot signal
            axes[0].plot(signal[:500])  # First 500 samples
            axes[0].set_title(f"Signal: {Path(path).stem}")
            axes[0].set_xlabel("Sample")
            axes[0].set_ylabel("Amplitude")
            
            # Plot TFI
            im = axes[1].imshow(tfi, aspect='auto', cmap='hot', 
                              extent=[0, tfi.shape[1], 0, tfi.shape[0]])
            axes[1].set_title("Time-Frequency Image")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel("Frequency")
            plt.colorbar(im, ax=axes[1])
            
            plt.tight_layout()
            plt.show()
        
        return tfis
    
    def generate_tfi_from_labels(self, base_dir: str, 
                               label_noise_pairs: List[Tuple[List[int], float]]):
        """Generate TFI from one-hot labels and noise levels"""
        import matplotlib.pyplot as plt
        
        dataset_manager = DatasetManager(base_dir)
        tfis = []
        
        for label, noise_level in label_noise_pairs:
            # Find matching signal
            signal_path = self._find_signal_by_label(base_dir, label, noise_level)
            
            if signal_path:
                # Load and generate TFI
                signal = np.load(signal_path)
                tfi = self.signal_to_tfi(signal)
                tfis.append((signal_path, tfi))
                
                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                
                # Plot signal
                axes[0].plot(signal[:500])
                axes[0].set_title(f"Signal: {Path(signal_path).stem}")
                axes[0].set_xlabel("Sample")
                axes[0].set_ylabel("Amplitude")
                
                # Plot TFI
                im = axes[1].imshow(tfi, aspect='auto', cmap='hot')
                axes[1].set_title(f"TFI (SNR: {noise_level} dB)")
                axes[1].set_xlabel("Time")
                axes[1].set_ylabel("Frequency")
                plt.colorbar(im, ax=axes[1])
                
                plt.tight_layout()
                plt.show()
            else:
                print(f"No signal found for label {label} at SNR {noise_level} dB")
        
        return tfis
    
    def _find_signal_by_label(self, base_dir: str, label: List[int], 
                            noise_level: float) -> Optional[str]:
        """Find signal file path matching given label and noise level"""
        base_path = Path(base_dir)
        
        # Determine if single or dual component
        is_dual = label[15] == 1
        
        if not is_dual:
            # Single component
            mod_idx = label.index(1)
            modulation = CONFIG["modulation_types"][mod_idx]
            
            # Search in all splits
            for split in ["train", "val", "test"]:
                meta_path = base_path / "single" / split / "metadata" / "global_metadata.csv"
                if meta_path.exists():
                    df = pd.read_csv(meta_path)
                    # Filter by modulation and SNR
                    matches = df[
                        (df["modulation"] == modulation) & 
                        (df["snr_db"] == noise_level)
                    ]
                    if not matches.empty:
                        signal_path = base_path / matches.iloc[0]["signal_path"]
                        return str(signal_path)
        else:
            # Dual component
            mod_indices = [i for i, val in enumerate(label[:15]) if val == 1]
            mod1 = CONFIG["modulation_types"][mod_indices[0]]
            mod2 = CONFIG["modulation_types"][mod_indices[1]] if len(mod_indices) > 1 else mod1
            
            # Search in all splits
            for split in ["train", "val", "test"]:
                meta_path = base_path / "dual" / split / "metadata" / "global_metadata.csv"
                if meta_path.exists():
                    df = pd.read_csv(meta_path)
                    # Filter by modulations and SNR
                    matches = df[
                        ((df["modulation1"] == mod1) & (df["modulation2"] == mod2) |
                         (df["modulation1"] == mod2) & (df["modulation2"] == mod1)) &
                        (df["snr_db"] == noise_level)
                    ]
                    if not matches.empty:
                        signal_path = base_path / matches.iloc[0]["signal_path"]
                        return str(signal_path)
        
        return None

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Signal Modulation Dataset Generator")
    parser.add_argument("--base_dir", type=str, default="./signal_dataset", 
                    help="Base directory for dataset")
    parser.add_argument("--generate_single", action="store_true", 
                    help="Generate single component dataset")
    parser.add_argument("--generate_dual", action="store_true", 
                    help="Generate dual component dataset")
    parser.add_argument("--split", type=str, default="train", 
                    choices=["train", "val", "test"], help="Dataset split to generate")
    parser.add_argument("--use_checkpoint", action="store_true", 
                    help="Use checkpoint if available")
    parser.add_argument("--generate_tfi", action="store_true", 
                    help="Generate TFI plots for sample signals")
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    manager = DatasetManager(args.base_dir)
    
    # Save configuration
    config_path = Path(args.base_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2, default=str)
    
    # Generate datasets
    if args.generate_single:
        manager.generate_single_component_dataset(
            split=args.split, 
            use_checkpoint=args.use_checkpoint
        )
    
    if args.generate_dual:
        manager.generate_dual_component_dataset(
            split=args.split, 
            use_checkpoint=args.use_checkpoint
        )
    
    # Generate sample TFIs if requested
    if args.generate_tfi:
        tfi_gen = TFIGenerator(CONFIG)
        
        # Example 1: Generate TFI from specific signal files
        print("\nExample 1: Generating TFI from specific signal files")
        sample_paths = [
            # ("./signal_dataset/single/train/signals/NM_0000_+00dB.npy", 0),
            # ("./signal_dataset/single/train/signals/LFM_0000_+00dB.npy", 0),
        ]

        # Uncomment when signals exist, i.e you generated the dataset
        tfi_gen.generate_tfi_from_paths(sample_paths)
        
        # Example 2: Generate TFI from labels
        print("\nExample 2: Generating TFI from labels")
        sample_labels = [
            # ([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # NM, 0 dB
            # ([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  # LFM, 0 dB
            # ([0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  
            ([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0),  
        ]

        # Uncomment when signals exist:
        tfi_gen.generate_tfi_from_labels(args.base_dir, sample_labels)
    
    print("\nDataset generation completed!")
    print(f"Dataset stored in: {args.base_dir}")
    print(f"Configuration saved to: {config_path}")

# ============================================================================
# DATA PIPELINE FOR MODEL TRAINING
# ============================================================================

class SignalDataset:
    """PyTorch Dataset for signal modulation classification"""
    
    def __init__(self, base_dir: str, split: str, component_type: str = "single", 
                transform=None, generate_tfi: bool = False):
        """
        Args:
            base_dir: Base directory of dataset
            split: 'train', 'val', or 'test'
            component_type: 'single' or 'dual'
            transform: Optional transforms
            generate_tfi: If True, generate TFI on-the-fly
        """
        self.base_dir = Path(base_dir)
        self.split = split
        self.component_type = component_type
        self.transform = transform
        self.generate_tfi = generate_tfi
        
        # Load metadata
        meta_path = self.base_dir / component_type / split / "metadata" / "global_metadata.csv"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        
        self.metadata = pd.read_csv(meta_path)
        
        # Initialize TFI generator if needed
        if generate_tfi:
            self.tfi_generator = TFIGenerator(CONFIG)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Get metadata
        row = self.metadata.iloc[idx]
        
        # Load signal
        signal_path = self.base_dir / row["signal_path"]
        signal = np.load(signal_path)
        
        # Generate TFI if requested
        if self.generate_tfi:
            tfi = self.tfi_generator.signal_to_tfi(signal)
            # Convert to tensor and add channel dimension
            tfi = tfi[np.newaxis, :, :]  # Shape: (1, H, W)
        else:
            # Use raw signal (for alternative approaches)
            tfi = signal
        
        # Parse one-hot label
        one_hot_str = row["one_hot_label"]
        one_hot = eval(one_hot_str) if isinstance(one_hot_str, str) else one_hot_str
        label = np.array(one_hot, dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            tfi = self.transform(tfi)
        
        return tfi, label

# ============================================================================
# MEMORY-MANAGED GENERATION WITH THREADING
# ============================================================================

import threading
import queue
import psutil
import gc

class MemoryManagedGenerator:
    """Memory-managed generator with threading support"""
    
    def __init__(self, manager: DatasetManager, max_memory_percent: float = 80.0):
        self.manager = manager
        self.max_memory_percent = max_memory_percent
        self.task_queue = queue.Queue()
        self.results = []
        self.lock = threading.Lock()
    
    def check_memory(self):
        """Check if memory usage is below threshold"""
        memory_percent = psutil.virtual_memory().percent
        return memory_percent < self.max_memory_percent
    
    def worker(self):
        """Worker thread function"""
        while True:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                # Check memory before processing
                if not self.check_memory():
                    print("Memory threshold exceeded, waiting...")
                    time.sleep(5)
                    self.task_queue.put(task)  # Requeue task
                    continue
                
                # Process task
                result = self.process_task(task)
                
                with self.lock:
                    self.results.append(result)
                
                # Force garbage collection
                gc.collect()
                
                self.task_queue.task_done()
                
            except queue.Empty:
                break
    
    def process_task(self, task):
        """Process a single generation task"""
        # This would be implemented based on specific task type
        pass



# ============================================================================
# INSTALLATION AND SETUP
# ============================================================================

def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys
    
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
        "torch",
        "torchvision",
        "psutil",
        "tqdm"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("\nAll packages installed!")

if __name__ == "__main__":
    # First, install dependencies
    print("=== Signal Modulation Dataset Generator ===")
    print("1. Installing dependencies...")
    # install_dependencies()
    
    print("\n2. Running main generator...")
    print("   Usage examples:")
    print("   python dataset_manager.py --generate_single --split train")
    print("   python dataset_manager.py --generate_dual --split val")
    print("   python dataset_manager.py --generate_tfi")
    
    main()
