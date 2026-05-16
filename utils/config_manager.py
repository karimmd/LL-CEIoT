#!/usr/bin/env python3

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        self.config[key] = value
    
    def save(self) -> bool:
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def update(self, updates: Dict[str, Any]) -> None:
        self.config.update(updates)

def create_default_configs():
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    cloud_config = {
        "model_path": "models/mistral/snapshots/2dcff66eac0c01dc50e4c41eea959968232187fe",
        "max_concurrent_tasks": 32,
        "tasks_per_second": 8.0,
        "memory_gb": 256,
        "gpu_memory_gb": 11,
        "preemption_threshold": 1.5,
        "max_batch_size": 16,
        "min_batch_size": 4,
        "max_new_tokens": 100,
        "temperature": 0.7,
        "server_port": 8000,
        "log_level": "INFO"
    }
    
    edge_config = {
        "model_path": "models/tinyllama/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "use_cpu_inference": True,
        "quantization_bits": 8,
        "max_concurrent_tasks": 8,
        "tasks_per_second": 2.0,
        "memory_gb": 16,
        "gpu_memory_gb": 2,
        "cpu_cores": 4,
        "preemption_threshold": 1.2,
        "max_batch_size": 8,
        "min_batch_size": 2,
        "max_new_tokens": 50,
        "temperature": 0.8,
        "server_port": 8001,
        "log_level": "INFO"
    }
    
    with open(configs_dir / "cloud_config.json", 'w') as f:
        json.dump(cloud_config, f, indent=2)
    
    with open(configs_dir / "edge_config.json", 'w') as f:
        json.dump(edge_config, f, indent=2)
    
    print("Default configuration files created")

if __name__ == "__main__":
    create_default_configs()