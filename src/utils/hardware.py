"""Hardware Detection and Resource Adaptation.

Detects available computing hardware (CPU, RAM, GPU, CUDA, MPS)
and automatically selects optimal PyTorch device, batch size, and worker threads.
Ensures full CPU compatibility without error.
"""

import os
import platform
import psutil
import torch
from dataclasses import dataclass


@dataclass
class HardwareProfile:
    device: str
    device_name: str
    num_cpus: int
    total_ram_gb: float
    cuda_available: bool
    cuda_device_count: int
    recommended_batch_size: int
    recommended_num_workers: int


def detect_hardware() -> HardwareProfile:
    """Detects available hardware environment and returns recommended execution profile."""
    num_cpus = os.cpu_count() or 2
    try:
        mem = psutil.virtual_memory()
        total_ram_gb = round(mem.total / (1024 ** 3), 2)
    except Exception:
        total_ram_gb = 4.0

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    if cuda_available and device_count > 0:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        recommended_batch_size = 64
        recommended_num_workers = min(4, num_cpus)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon MPS"
        recommended_batch_size = 64
        recommended_num_workers = 2
    else:
        device = "cpu"
        device_name = f"{platform.processor() or 'Generic CPU'} ({num_cpus} cores)"
        recommended_batch_size = 32
        recommended_num_workers = 0  # 0 workers avoids multiprocessing overhead on standard CPU

    return HardwareProfile(
        device=device,
        device_name=device_name,
        num_cpus=num_cpus,
        total_ram_gb=total_ram_gb,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        recommended_batch_size=recommended_batch_size,
        recommended_num_workers=recommended_num_workers,
    )


def get_torch_device() -> torch.device:
    """Returns PyTorch device object."""
    profile = detect_hardware()
    return torch.device(profile.device)
