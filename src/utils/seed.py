"""Deterministic Seed Management and Reproducibility Tracker.

Enforces deterministic operations across Python random, NumPy, and PyTorch,
and records the complete execution provenance metadata.
"""

import os
import random
import sys
import platform
import json
from datetime import datetime
from typing import Any, Dict
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across all random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def record_provenance_metadata(
    output_path: str = "models/experiment_provenance.json",
    hyperparameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Records system, Python, PyTorch, and CUDA environment specifications."""
    metadata = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "hyperparameters": hyperparameters or {},
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
