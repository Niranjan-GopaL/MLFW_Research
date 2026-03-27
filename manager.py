import numpy as np
import pandas as pd
import os
import json
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SignalDatasetGenerator:
    """Bulletproof Dataset Generator for Signal Classification (ALL CSV FORMAT)"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        
        self.config = {
            "signal_length": 2048,
            "sampling_frequency": 100e6,
            "pulse_width": 20.48e-6,
            
            # Scaled down configuration for testing
            "single_component": {
                "train": {"parameter_combinations": 200, "noise_levels": [-12, -6, 0, 6, 12, 18]},
                "val":   {"parameter_combinations": 50,  "noise_levels": [-12, -6, 0, 6, 12, 18]},
                "test":  {"parameter_combinations": 100, "noise_levels": [-21, -12, -6, 0, 6, 12, 18]}
            },
            "dual_component": {
                "train": {"parameter_combinations_per_mod": 8, "noise_levels": [-12, -6, 0, 6, 12, 18]},
                "val":   {"parameter_combinations_per_mod": 4, "noise_levels": [-12, -6, 0, 6, 12, 18]},
                "test":  {"parameter_combinations_per_mod": 6, "noise_levels": [-21, -12, -6, 0, 6, 12, 18]}
            },
            
            "modulation_params": {
                "NM": {"f0_range": (0.1, 0.4)},
                "LFM": {"f0_range": (0.01, 0.45), "delta_f_range": (0.05, 0.4)},
                "DLFM": {"f0_range": (0.01, 0.4), "delta_f_range": (0.05, 0.35)},
                "MLFM": {"f0_range": (0.15, 0.5), "delta_f_range": (0.1, 0.35), "r_range": (0.3, 0.7)},
                "EQFM": {"fmin_range": (0.01, 0.4), "delta_f_range": (0.05, 0.3)},
                "SFM": {"fmin_range": (0.01, 0.15), "delta_f_range": (0.05, 0.35), "fSFM_range": (0.75, 10), "phiSFM_range": (0, 2*np.pi)},
                "BFSK": {"f1_range": (0.05, 0.45), "f2_range": (0.05, 0.45), "N_values": [5, 7, 11, 13]},
                "QFSK": {"f_range": (0.05, 0.45), "N_value": 4},
                "BPSK": {"f0_range": (0.05, 0.45), "N_values": [5, 7, 11, 13]},
                "Frank": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
                "P1": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
                "P2": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
                "P3": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
                "P4": {"f0_range": (0.1, 0.4), "N_values": [6, 7, 8]},
                "LFM_BPSK": {"fmin_range": (0.05, 0.45), "delta_f_range": (0.05, 0.4), "N_values": [5, 7, 11, 13]}
            },
            "modulation_types": ["NM", "LFM", "DLFM", "MLFM", "EQFM", "SFM", "BFSK", "QFSK", "BPSK", "Frank", "P1", "P2", "P3", "P4", "LFM_BPSK"]
        }
        
        self.fs = self.config["sampling_frequency"]
        self.N = self.config["signal_length"]
        self.t = np.arange(self.N) / self.fs
        self._setup_folders()

    def _setup_folders(self):
        """Creates the folder structure. Clears old data if it exists."""
        if self.base_dir.exists():
            print(f"[System] Clearing existing directory at {self.base_dir} to start fresh...")
            shutil.rmtree(self.base_dir) 
            
        for split in ["train", "val", "test"]:
            (self.base_dir / "single" / split / "signals").mkdir(parents=True, exist_ok=True)
            (self.base_dir / "single" / split / "metadata").mkdir(parents=True, exist_ok=True)
            (self.base_dir / "dual" / split / "signals").mkdir(parents=True, exist_ok=True)
            (self.base_dir / "dual" / split / "metadata").mkdir(parents=True, exist_ok=True)

    def _get_one_hot_label(self, modulation: Union[str, List[str]], is_dual: bool) -> str:
        label = [0] * 16
        if not is_dual:
            mod_idx = self.config["modulation_types"].index(modulation)
            label[mod_idx] = 1
            label[15] = 0
        else:
            mod1_idx = self.config["modulation_types"].index(modulation[0])
            mod2_idx = self.config["modulation_types"].index(modulation[1])
            label[mod1_idx] = 1
            label[mod2_idx] = 1
            # 16th bit is 1 ONLY IF dual AND modulations are identical
            if mod1_idx == mod2_idx:
                label[15] = 1
            else:
                label[15] = 0
        return str(label)

    def build_single_dataset(self, split: str):
        print(f"\n---> Building SINGLE Component Dataset: [{split.upper()}]")
        config = self.config["single_component"][split]
        mod_types = self.config["modulation_types"]
        global_metadata = []
        
        total_expected = len(mod_types) * config["parameter_combinations"] * len(config["noise_levels"])
        
        with tqdm(total=total_expected, desc=f"Single {split.upper()}", unit="sig") as pbar:
            for mod_idx, modulation in enumerate(mod_types):
                mod_metadata = []
                for param_idx in range(config["parameter_combinations"]):
                    params = self._generate_random_params(modulation)
                    clean_signal = self._create_raw_signal(modulation, params)
                    
                    for snr_db in config["noise_levels"]:
                        noisy_signal = self._add_noise(clean_signal, snr_db)
                        real_signal = np.real(noisy_signal).astype(np.float32)
                        
                        signal_id = f"{modulation}_{param_idx:04d}_{snr_db:+03d}dB"
                        
                        # SAVING SIGNAL DATA AS .CSV
                        signal_path = self.base_dir / "single" / split / "signals" / f"{signal_id}.csv"
                        np.savetxt(signal_path, real_signal, delimiter=",", fmt='%.6f')
                        
                        # EXACTLY 8 COLUMNS for Single Component Metadata
                        metadata_entry = {
                            "signal_id": signal_id, 
                            "signal_path": f"single/{split}/signals/{signal_id}.csv",  # Path now ends in .csv
                            "modulation": modulation, 
                            "modulation_idx": mod_idx,
                            "snr_db": snr_db,
                            "params": str(params),
                            "is_dual_component": 0,
                            "one_hot_label": self._get_one_hot_label(modulation, is_dual=False)
                        }
                        mod_metadata.append(metadata_entry)
                        global_metadata.append(metadata_entry)
                        pbar.update(1)
                
                # Save Individual Modulation CSV
                pd.DataFrame(mod_metadata).to_csv(self.base_dir / "single" / split / "metadata" / f"{modulation}_metadata.csv", index=False)
        
        # Save Global Single CSV
        pd.DataFrame(global_metadata).to_csv(self.base_dir / "single" / split / "metadata" / "global_metadata.csv", index=False)

    def build_dual_dataset(self, split: str):
        print(f"\n---> Building DUAL Component Dataset: [{split.upper()}]")
        config = self.config["dual_component"][split]
        mod_types = self.config["modulation_types"]
        global_metadata = []
        
        mod_combinations = [(mod_types[i], mod_types[j]) for i in range(len(mod_types)) for j in range(i, len(mod_types))]
        total_expected = len(mod_combinations) * (config["parameter_combinations_per_mod"] ** 2) * len(config["noise_levels"])
        
        with tqdm(total=total_expected, desc=f"Dual {split.upper()}", unit="sig") as pbar:
            for (mod1, mod2) in mod_combinations:
                combo_metadata = []
                signals_1 = [{"clean": self._create_raw_signal(mod1, self._generate_random_params(mod1))} 
                             for _ in range(config["parameter_combinations_per_mod"])]
                signals_2 = [{"clean": self._create_raw_signal(mod2, self._generate_random_params(mod2))} 
                             for _ in range(config["parameter_combinations_per_mod"])]
                
                mod1_idx = self.config["modulation_types"].index(mod1)
                mod2_idx = self.config["modulation_types"].index(mod2)
                
                for i in range(config["parameter_combinations_per_mod"]):
                    for j in range(config["parameter_combinations_per_mod"]):
                        alpha = np.random.uniform(0.25, 0.5)
                        
                        for snr_db in config["noise_levels"]:
                            combined_clean = (alpha * signals_1[i]["clean"]) + ((1 - alpha) * signals_2[j]["clean"])
                            noisy_signal = self._add_noise(combined_clean, snr_db)
                            real_signal = np.real(noisy_signal).astype(np.float32)
                            
                            signal_id = f"{mod1}_{mod2}_{i:03d}_{j:03d}_{snr_db:+03d}dB"
                            
                            # SAVING SIGNAL DATA AS .CSV
                            signal_path = self.base_dir / "dual" / split / "signals" / f"{signal_id}.csv"
                            np.savetxt(signal_path, real_signal, delimiter=",", fmt='%.6f')
                            
                            # EXACTLY 11 COLUMNS for Dual Component Metadata
                            metadata_entry = {
                                "signal_id": signal_id, 
                                "signal_path": f"dual/{split}/signals/{signal_id}.csv",  # Path now ends in .csv
                                "modulation1": mod1, 
                                "modulation2": mod2,
                                "modulation_idx1": mod1_idx,
                                "modulation_idx2": mod2_idx,
                                "snr_db": snr_db, 
                                "alpha": alpha,
                                "is_dual_component": 1,
                                "is_same_modulation": 1 if mod1 == mod2 else 0,
                                "one_hot_label": self._get_one_hot_label([mod1, mod2], is_dual=True)
                            }
                            combo_metadata.append(metadata_entry)
                            global_metadata.append(metadata_entry)
                            pbar.update(1)

                pd.DataFrame(combo_metadata).to_csv(self.base_dir / "dual" / split / "metadata" / f"{mod1}_{mod2}_metadata.csv", index=False)

        pd.DataFrame(global_metadata).to_csv(self.base_dir / "dual" / split / "metadata" / "global_metadata.csv", index=False)

    def _generate_random_params(self, modulation: str) -> Dict:
        param_ranges = self.config["modulation_params"][modulation]
        params = {}
        for key, value in param_ranges.items():
            if key.endswith("_range"): 
                val = np.random.uniform(value[0], value[1]) if isinstance(value, tuple) else value
            elif key.endswith("_values"): 
                val = np.random.choice(value)
            else: 
                val = value
            params[key.replace("_range", "").replace("_values", "")] = val
            
        if modulation == "LFM": params["direction"] = np.random.choice([-1, 1])
        if modulation in ["BFSK", "BPSK", "Frank", "P1", "P2", "P3", "P4"]: 
            params["inverted"] = np.random.choice([True, False])
            
        # Clean native python types for dictionary stringification
        native_params = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, int)): native_params[k] = int(v)
            elif isinstance(v, (np.floating, float)): native_params[k] = float(v)
            elif isinstance(v, (np.bool_, bool)): native_params[k] = bool(v)
            else: native_params[k] = v
            
        return native_params

    def _create_raw_signal(self, modulation: str, params: Dict) -> np.ndarray:
        if modulation == "NM": 
            return np.exp(1j * 2 * np.pi * (params["f0"] * self.fs) * self.t)
        elif modulation in ["LFM", "DLFM"]: 
            f0 = params["f0"] * self.fs if "f0" in params else (params["fmin"] * self.fs + params.get("delta_f", 0) * self.fs / 2)
            f_inst = f0 + params.get("direction", 1) * (params["delta_f"] * self.fs) * self.t / self.t[-1]
            return np.exp(1j * 2 * np.pi * np.cumsum(f_inst) / self.fs)
        elif modulation == "MLFM": 
            f0, delta_f, split_idx = params["f0"] * self.fs, params["delta_f"] * self.fs, int(self.N * params["r"])
            f1 = f0 + delta_f * self.t[:split_idx] / self.t[:split_idx][-1] if split_idx > 0 else np.array([])
            f2 = f0 + delta_f * (1 - self.t[split_idx:] / self.t[split_idx:][-1]) if len(self.t[split_idx:]) > 0 else np.array([])
            return np.exp(1j * 2 * np.pi * np.cumsum(np.concatenate([f1, f2])) / self.fs)
        elif modulation == "EQFM": 
            f_inst = (params["fmin"] * self.fs) + (params["delta_f"] * self.fs) * (1 + np.sin(2 * np.pi * 5 * self.t / self.t[-1])) / 2
            return np.exp(1j * 2 * np.pi * np.cumsum(f_inst) / self.fs)
        elif modulation == "SFM": 
            f_inst = (params["fmin"] * self.fs) + (params["delta_f"] * self.fs) * np.sin(2 * np.pi * (params["fSFM"] / self.config["pulse_width"]) * self.t + params["phiSFM"])
            return np.exp(1j * 2 * np.pi * np.cumsum(f_inst) / self.fs)
        elif modulation == "BFSK": 
            code = self._get_barker(params["N"], params.get("inverted", False))
            sig = np.zeros(self.N, dtype=complex)
            for i, sym in enumerate(code):
                start, end = i * (self.N // params["N"]), (i + 1) * (self.N // params["N"]) if i < params["N"] - 1 else self.N
                sig[start:end] = np.exp(1j * 2 * np.pi * ((params["f1"] if sym == 1 else params["f2"]) * self.fs) * self.t[start:end])
            return sig
        elif modulation == "QFSK": 
            return np.exp(1j * 2 * np.pi * (np.random.uniform(0.05, 0.45) * self.fs) * self.t)
        elif modulation == "BPSK": 
            code = self._get_barker(params["N"], params.get("inverted", False))
            sig, phase = np.zeros(self.N, dtype=complex), 0
            for i, sym in enumerate(code):
                start, end = i * (self.N // params["N"]), (i + 1) * (self.N // params["N"]) if i < params["N"] - 1 else self.N
                if sym == -1: phase += np.pi
                sig[start:end] = np.exp(1j * (2 * np.pi * (params["f0"] * self.fs) * self.t[start:end] + phase))
            return sig
        elif modulation in ["Frank", "P1", "P2", "P3", "P4"]: 
            return np.exp(1j * 2 * np.pi * (params["f0"] * self.fs) * self.t)
        elif modulation == "LFM_BPSK": 
            lfm = self._create_raw_signal("LFM", {"fmin": params["fmin"], "delta_f": params["delta_f"], "direction": params.get("direction", 1)})
            bpsk = self._create_raw_signal("BPSK", {"f0": params["fmin"] + params["delta_f"] / 2, "N": params["N"], "inverted": params.get("inverted", False)})
            return lfm * bpsk
        return np.zeros(self.N)

    def _get_barker(self, N: int, inverted: bool) -> np.ndarray:
        codes = {5: [1,1,1,-1,1], 7: [1,1,1,-1,-1,1,-1], 11: [1,1,1,-1,-1,-1,1,-1,-1,1,-1], 13: [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]}
        return np.array(codes[N]) * (-1 if inverted else 1)

    def _add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        power = np.mean(np.abs(signal) ** 2) / (10 ** (snr_db / 10))
        return signal + np.random.normal(0, np.sqrt(power/2), len(signal)) + 1j * np.random.normal(0, np.sqrt(power/2), len(signal))


if __name__ == "__main__":
    try:
        print("==================================================")
        print("INITIALIZING SIGNAL MODULATION DATASET GENERATOR")
        print("==================================================\n")
        
        # VS Code / Python Safe Path Generation
        current_directory = os.getcwd()
        target_folder = os.path.join(current_directory, "signal_dataset")
        
        print(f"Dataset will be generated at:\n{target_folder}\n")
        
        generator = SignalDatasetGenerator(base_dir=target_folder)
        
        # Generate Datasets
        generator.build_single_dataset("train")
        generator.build_single_dataset("val")
        generator.build_single_dataset("test")
        
        generator.build_dual_dataset("train")
        generator.build_dual_dataset("val")
        generator.build_dual_dataset("test")
        
        print("\n==================================================")
        print("ALL DONE! Your dataset is ready.")
        print("==================================================")
        
    except Exception as e:
        print("\n" + "!"*50)
        print("CRITICAL ERROR ENCOUNTERED:")
        print("!"*50)
        traceback.print_exc()
        print("!"*50)