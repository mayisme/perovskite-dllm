import yaml
import os
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configurations."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged

def get_default_config() -> Dict[str, Any]:
    """Get the default configuration."""
    return {
        "data": {
            "raw_path": "data/raw/perovskites_full.json",
            "processed_path": "data/processed/perovskites.h5",
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1,
            "batch_size": 32,
            "num_workers": 4,
            "pbc": True,
            "radius_cutoff": 5.0
        },
        "model": {
            "hidden_dim": 128,
            "n_layers": 4,
            "edge_dim": 32,
            "attention": True,
            "diffusion_steps": 500,
            "noise_schedule": "cosine"
        },
        "training": {
            "epochs": 100,
            "lr": 1e-4,
            "weight_decay": 1e-6,
            "grad_clip": 1.0,
            "device": "cuda",
            "log_interval": 10,
            "save_interval": 10,
            "weights": {
                "noise": 1.0,
                "rdf": 0.1,
                "pauli": 1.0,
                "energy": 0.01
            }
        },
        "generation": {
            "num_samples": 10,
            "sampling_type": "ddpm",
            "guidance_scale": 1.0,
            "temperature": 1.0
        }
    }
