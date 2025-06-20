# utils.py - Advanced Deep Learning Utilities
import torch
import torch.nn as nn
import numpy as np
import os
import re
import json
import hashlib
import requests
import time
import math
import socket
import psutil
import platform
import random
from pathlib import Path
from typing import *
from collections import deque, defaultdict
from functools import wraps, lru_cache
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from PIL import Image
import numpy as np

def simple_transform(image_path):
    """Simple image transformation without torchvision dependency"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize like torchvision
    image_np = np.array(image).astype("float32") / 255.0  # Normalize 0-1
    image_np = np.transpose(image_np, (2, 0, 1))  # CHW format like PyTorch
    return image_np

# ----------------------
# Enhanced Device Management
# ----------------------
class DeviceManager:
    """Comprehensive device management with automatic optimization"""
    def __init__(self):
        self._device = None
        self._gpu_ids = []
        self._gpu_status = {}
        self._memory_allocated = 0
        self.update_devices()
        
    def update_devices(self):
        """Detect available devices and optimize settings"""
        # GPU detection and prioritization
        if torch.cuda.is_available():
            self._gpu_ids = list(range(torch.cuda.device_count()))
            self._device = torch.device(f'cuda:{self._gpu_ids[0]}')
            
            # Set optimal flags
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Cache GPU status
            self._cache_gpu_status()
        else:
            self._device = torch.device('cpu')
            # Enable MPS for Apple Silicon
            if torch.backends.mps.is_available():
                self._device = torch.device('mps')
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # JIT compilation optimization
        torch.set_float32_matmul_precision('high')
        
    def _cache_gpu_status(self):
        """Cache detailed GPU information"""
        self._gpu_status = {}
        for i in self._gpu_ids:
            props = torch.cuda.get_device_properties(i)
            mem_info = torch.cuda.mem_get_info(i)
            self._gpu_status[i] = {
                'name': props.name,
                'capability': f"{props.major}.{props.minor}",
                'total_mem': props.total_memory,
                'free_mem': mem_info[0],
                'used_mem': props.total_memory - mem_info[0]
            }
    
    def memory_summary(self) -> str:
        """Generate detailed memory report"""
        if not self._gpu_ids:
            return "CPU Device - No GPU memory information"
        
        report = ["\n====== GPU Memory Summary ======"]
        for gpu_id, status in self._gpu_status.items():
            report.append(
                f"GPU {gpu_id} ({status['name']}): "
                f"{self._bytes_to_gb(status['used_mem']):.2f}GB / "
                f"{self._bytes_to_gb(status['total_mem']):.2f}GB used "
                f"({status['free_mem'] / status['total_mem'] * 100:.1f}% free)"
            )
        return "\n".join(report)
    
    @staticmethod
    def _bytes_to_gb(bytes_val: int) -> float:
        return bytes_val / (1024 ** 3)
    
    def auto_configure(self) -> torch.device:
        """Automatically configure optimal device settings"""
        # Enable TF32 where supported
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set deterministic mode if requested
        if os.environ.get('DETERMINISTIC', '0') == '1':
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        return self.device
    
    @property
    def device(self) -> torch.device:
        """Get primary compute device"""
        return self._device
    
    @property
    def available_gpus(self) -> List[int]:
        """List of available GPU IDs"""
        return self._gpu_ids.copy()
    
    def set_device(self, device_spec: Union[str, int, torch.device]):
        """Set active compute device"""
        if isinstance(device_spec, int):
            if device_spec not in self._gpu_ids:
                raise ValueError(f"Invalid GPU ID. Available: {self._gpu_ids}")
            self._device = torch.device(f'cuda:{device_spec}')
        elif isinstance(device_spec, str):
            self._device = torch.device(device_spec)
        elif isinstance(device_spec, torch.device):
            self._device = device_spec
        else:
            raise TypeError("Invalid device specification")
        
        # Update GPU cache if using GPU
        if self._device.type == 'cuda':
            self._cache_gpu_status()
    
    def clear_memory(self):
        """Aggressive memory cleanup"""
        if torch.cuda.is_available():
            for i in self._gpu_ids:
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            self._cache_gpu_status()
        import gc
        gc.collect()

# Initialize global device manager
DEVICE_MANAGER = DeviceManager()
DEVICE = DEVICE_MANAGER.device

# ----------------------
# Memory Optimization Utils
# ----------------------
class MemoryOptimizer:
    """Advanced memory management for PyTorch"""
    @staticmethod
    def reduce_model_footprint(model: nn.Module):
        """Apply multiple memory reduction techniques"""
        # 1. Convert to half precision
        model.half()
        
        # 2. Apply gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        # 3. Optimizer state on CPU
        for param in model.parameters():
            param.requires_grad = False
            
        # 4. Cleanup
        torch.cuda.empty_cache()
        
    @staticmethod
    def optimized_cuda_allocator():
        """Custom memory allocator with optimized caching"""
        # Only available in PyTorch nightly
        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'CUDAPluggableAllocator'):
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                '/path/to/custom_allocator.so'
            )
            torch.cuda.memory.change_current_allocator(allocator)
            
    @staticmethod
    def auto_batch_size(model: nn.Module, input_shape: tuple, 
                       target_device: torch.device = DEVICE, 
                       max_mem_util: float = 0.8) -> int:
        """Dynamically calculate max batch size for model"""
        # Get available memory
        if target_device.type == 'cuda':
            total_mem = torch.cuda.get_device_properties(target_device).total_memory
            reserved = torch.cuda.memory_reserved(target_device)
            free_mem = total_mem - reserved
            max_mem = free_mem * max_mem_util
        else:
            # Estimate for CPU
            max_mem = psutil.virtual_memory().available * 0.7
            
        # Create dummy input
        dummy_input = torch.randn(*input_shape).to(target_device)
        
        # Profile memory usage
        with torch.no_grad():
            try:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
                    profile_memory=True, record_shapes=True
                ) as prof:
                    model(dummy_input)
                
                # Parse memory usage
                mem_events = [evt for evt in prof.key_averages() if evt.cuda_memory_usage > 0]
                per_sample_mem = max(evt.cuda_memory_usage for evt in mem_events) if mem_events else 0
            except Exception:
                # Fallback for older PyTorch versions
                model(dummy_input)
                per_sample_mem = torch.cuda.max_memory_allocated(target_device) if target_device.type == 'cuda' else 0
        
        # Calculate batch size
        batch_size = max(1, int(max_mem / per_sample_mem)) if per_sample_mem > 0 else 8
        return batch_size

# ----------------------
# Network & Web Utilities
# ----------------------
class WebUtils:
    """Enhanced web operations with caching, retries, and fallback"""
    CACHE_DIR = Path("web_cache")
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/118.0"
    ]
    
    def __init__(self, cache_enabled: bool = True, timeout: int = 15):
        self.cache_enabled = cache_enabled
        self.timeout = timeout
        self.session = requests.Session()
        self.CACHE_DIR.mkdir(exist_ok=True, parents=True)
        
    def _cache_path(self, url: str) -> Path:
        """Generate cache path for URL"""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        return self.CACHE_DIR / f"{url_hash}.cache"
    
    def fetch(self, url: str, force_refresh: bool = False, 
             retries: int = 3, backoff_factor: float = 0.5) -> str:
        """Fetch URL content with caching and retries"""
        cache_file = self._cache_path(url)
        
        # Check cache
        if self.cache_enabled and cache_file.exists() and not force_refresh:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                print(f"Cache read error: {e}")
        
        # Fetch with retries
        headers = {'User-Agent': random.choice(self.USER_AGENTS)}
        attempt = 0
        
        while attempt < retries:
            try:
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                response.raise_for_status()
                
                # Cache response
                content = response.text
                if self.cache_enabled:
                    try:
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                    except Exception as e:
                        print(f"Cache write error: {e}")
                return content
                
            except (requests.RequestException, ConnectionError) as e:
                attempt += 1
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Request failed ({e}), retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        raise ConnectionError(f"Failed to fetch {url} after {retries} attempts")
    
    def search_web(self, query: str, engine: str = "google", 
                  max_results: int = 5, lang: str = "en") -> List[str]:
        """Multi-engine web search with fallback"""
        engines = ["google", "bing", "duckduckgo", "yandex"]
        if engine not in engines:
            engine = "google"
            
        for current_engine in [engine] + [e for e in engines if e != engine]:
            try:
                if current_engine == "google":
                    url = f"https://www.google.com/search?q={quote(query)}&num={max_results+2}&hl={lang}"
                    content = self.fetch(url)
                    soup = BeautifulSoup(content, 'html.parser')
                    results = [div.get_text(strip=True) for div in soup.select("div.VwiC3b, .MUxGbd")]
                    return results[:max_results]
                
                elif current_engine == "bing":
                    url = f"https://www.bing.com/search?q={quote(query)}&count={max_results}"
                    content = self.fetch(url)
                    soup = BeautifulSoup(content, 'html.parser')
                    return [p.get_text(strip=True) for p in soup.select("li.b_algo p")][:max_results]
                
                elif current_engine == "duckduckgo":
                    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
                    content = self.fetch(url)
                    soup = BeautifulSoup(content, 'html.parser')
                    return [div.get_text(strip=True) for div in soup.select(".result__snippet")][:max_results]
                
                elif current_engine == "yandex":
                    url = f"https://yandex.com/search/?text={quote(query)}"
                    content = self.fetch(url)
                    soup = BeautifulSoup(content, 'html.parser')
                    return [div.get_text(strip=True) for div in soup.select(".organic__content-text")][:max_results]
                    
            except Exception as e:
                print(f"Search with {current_engine} failed: {str(e)}")
                continue
        
        return []
    
    @staticmethod
    def is_connected(timeout: int = 3) -> bool:
        """Check internet connectivity"""
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=timeout)
            return True
        except OSError:
            return False

# ----------------------
# Model Utilities
# ----------------------
class ModelUtils:
    """Advanced model operations and analysis"""
    @staticmethod
    def calculate_flops(model: nn.Module, input_size: tuple = (1, 137)) -> int:
        """Calculate approximate FLOPs for token-based transformer models"""
        try:
            from fvcore.nn import FlopCountAnalysis
            # Fake token IDs: integers from 0 to vocab_size-1
            vocab_size = model.embedding.num_embeddings if hasattr(model, "embedding") else 112
            inputs = torch.randint(0, vocab_size, input_size).to(DEVICE)
            flops = FlopCountAnalysis(model, inputs)
            return flops.total()
        except ImportError:
            print("Warning: fvcore not installed. FLOP calculation unavailable.")
            return 0
    
    @staticmethod
    def model_summary(model: nn.Module, depth: int = 3) -> str:
        """Enhanced model summary with hierarchy"""
        def _get_layer_info(module, depth, current_depth=0):
            info = []
            for name, child in module.named_children():
                indent = "  " * current_depth
                params = sum(p.numel() for p in child.parameters())
                info.append(f"{indent}{name} ({child.__class__.__name__}): {params:,} params")
                if current_depth < depth and len(list(child.children())) > 0:
                    info.extend(_get_layer_info(child, depth, current_depth+1))
            return info
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = [
            f"Model: {model.__class__.__name__}",
            f"Total params: {total_params:,}",
            f"Trainable params: {trainable:,}",
            f"FLOPs (est): {ModelUtils.calculate_flops(model):,}",
            "Detailed structure:"
        ]
        summary.extend(_get_layer_info(model, depth))
        return "\n".join(summary)
    
    @staticmethod
    def init_weights(module: nn.Module, 
                    init_type: str = 'xavier_uniform',
                    gain: float = 1.0,
                    bias: float = 0.0):
        """Advanced weight initialization"""
        init_funcs = {
            'xavier_uniform': nn.init.xavier_uniform_,
            'xavier_normal': nn.init.xavier_normal_,
            'kaiming_uniform': lambda w: nn.init.kaiming_uniform_(w, nonlinearity='relu'),
            'kaiming_normal': lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'),
            'orthogonal': lambda w: nn.init.orthogonal_(w, gain=gain),
            'trunc_normal': lambda w: nn.init.trunc_normal_(w, mean=0.0, std=0.02),
        }
        
        if init_type == 'quantum':
            # Custom quantum-inspired initialization
            with torch.no_grad():
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.weight.dim() >= 2:  # Only apply SVD to matrices
                        u, s, v = torch.svd(module.weight.data)
                        module.weight.data = torch.mm(u, v.t())
        else:
            # Standard initialization
            init_func = init_funcs.get(init_type, init_funcs['xavier_uniform'])
            
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                init_func(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    @staticmethod
    def fuse_model(model: nn.Module) -> nn.Module:
        """Fuse Conv-BN layers for inference efficiency"""
        try:
            from torch.ao.quantization import fuse_modules
            fused_model = model
            if hasattr(model, 'fuse_model'):
                model.fuse_model()
            else:
                # Automatic fusion for common patterns
                for module_name, module in model.named_children():
                    if isinstance(module, nn.Sequential):
                        for i, child in enumerate(module):
                            if isinstance(child, nn.Conv2d) and i < len(module)-1:
                                next_child = module[i+1]
                                if isinstance(next_child, nn.BatchNorm2d):
                                    fuse_modules(module, [str(i), str(i+1)], inplace=True)
                    ModelUtils.fuse_model(module)
            return fused_model
        except ImportError:
            print("Warning: Quantization modules not available. Returning original model.")
            return model

# ----------------------
# Data Processing Utilities
# ----------------------
class DataUtils:
    """Advanced data processing utilities"""
    @staticmethod
    def dynamic_padding(batch: List[Dict], 
                       keys: List[str] = ['input_ids', 'attention_mask'],
                       padding_side: str = 'right',
                       pad_value: int = 0) -> Dict[str, torch.Tensor]:
        """Dynamically pad batch of dictionaries"""
        max_len = max(len(item[keys[0]]) for item in batch)
        padded_batch = {key: [] for key in keys}
        
        for item in batch:
            for key in keys:
                tensor = torch.tensor(item[key])
                pad_size = max_len - len(tensor)
                
                if padding_side == 'right':
                    padded = F.pad(tensor, (0, pad_size), value=pad_value)
                elif padding_side == 'left':
                    padded = F.pad(tensor, (pad_size, 0), value=pad_value)
                else:  # 'both'
                    left_pad = pad_size // 2
                    right_pad = pad_size - left_pad
                    padded = F.pad(tensor, (left_pad, right_pad), value=pad_value)
                
                padded_batch[key].append(padded)
                
        return {k: torch.stack(v) for k, v in padded_batch.items()}
    
    @staticmethod
    def smart_batching(dataset: List, batch_size: int, sort_key: str = 'length') -> List[List]:
        """Create batches sorted by sequence length for efficiency"""
        if sort_key == 'length':
            dataset.sort(key=lambda x: len(x['input_ids']), reverse=True)
        elif sort_key == 'random':
            random.shuffle(dataset)
            
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            batches.append(batch)
            
        return batches
    
    @staticmethod
    def image_to_tensor(image_path: str, 
                       size: tuple = (224, 224), 
                       normalize: bool = True) -> torch.Tensor:
        """Load and preprocess image to tensor without torchvision"""
        # Use our simple_transform function
        image_np = simple_transform(image_path)
        tensor = torch.tensor(image_np)
        
        if normalize:
            # Apply basic normalization
            tensor = (tensor - 0.5) / 0.5
            
        return tensor.unsqueeze(0).to(DEVICE)

# ----------------------
# Performance Utilities
# ----------------------
class PerfUtils:
    """Performance monitoring and optimization"""
    @staticmethod
    def benchmark(model: nn.Module, 
                 input_tensor: torch.Tensor, 
                 warmup: int = 10, 
                 runs: int = 100) -> dict:
        """Benchmark model inference performance"""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
            
        # Benchmark
        start_time = time.time()
        if input_tensor.is_cuda:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        for _ in range(runs):
            with torch.no_grad():
                _ = model(input_tensor)
                
        if input_tensor.is_cuda:
            end_event.record()
            torch.cuda.synchronize()
            latency = start_event.elapsed_time(end_event) / runs
        else:
            latency = (time.time() - start_time) * 1000 / runs
            
        # Memory usage
        if input_tensor.is_cuda:
            mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_usage = 0
            
        return {
            'latency_ms': latency,
            'memory_mb': mem_usage,
            'throughput': 1000 / latency * input_tensor.size(0) if latency > 0 else 0
        }
    
    @staticmethod
    def profile_model(model: nn.Module, input_tensor: torch.Tensor) -> str:
        """Generate detailed performance profile"""
        try:
            from torch.profiler import profile, record_function, ProfilerActivity
            results = []
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with record_function("model_inference"):
                    model(input_tensor)
                    
            # Process results
            table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
            return str(table)
        except ImportError:
            return "Profiler not available in this PyTorch version"
    
    @staticmethod
    def optimize_onnx(model: nn.Module, input_tensor: torch.Tensor, 
                     onnx_path: str, opset: int = 15, dynamic_axes: dict = None):
        """Export to ONNX with optimizations"""
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            # Export to ONNX
            torch.onnx.export(
                model,
                input_tensor,
                onnx_path,
                opset_version=opset,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes or {'input': {0: 'batch'}, 'output': {0: 'batch'}}
            )
            
            # Optimize model
            onnx_model = onnx.load(onnx_path)
            optimized = optimizer.optimize_model(
                onnx_model,
                model_type='bert',
                num_heads=0,  # Auto-detect
                hidden_size=0  # Auto-detect
            )
            optimized.save_model_to_file(onnx_path)
            return onnx_path
        except ImportError:
            print("ONNX dependencies not installed. Skipping optimization.")
            return None

# ----------------------
# Security & Validation
# ----------------------
class SecurityUtils:
    """Model security and validation utilities"""
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        sanitized = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        return sanitized[:5000]  # Limit input length
    
    @staticmethod
    def detect_anomalies(tensor: torch.Tensor, threshold: float = 5.0) -> bool:
        """Detect anomalous values in tensors"""
        abs_tensor = tensor.abs()
        max_val = abs_tensor.max().item()
        mean_val = abs_tensor.mean().item()
        return max_val > threshold * mean_val and mean_val > 1e-6
    
    @staticmethod
    def gradient_clipping(model: nn.Module, max_norm: float = 1.0):
        """Apply gradient clipping with anomaly detection"""
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                if SecurityUtils.detect_anomalies(param.grad):
                    print(f"Anomalous gradient detected: max={param.grad.abs().max().item()}")
                torch.nn.utils.clip_grad_norm_(param, max_norm)

# ----------------------
# Distributed Training
# ----------------------
class DistributedUtils:
    """Utilities for distributed training"""
    @staticmethod
    def setup_ddp(local_rank: int, world_size: int):
        """Initialize distributed training"""
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            rank=local_rank,
            world_size=world_size
        )
        return torch.device(f'cuda:{local_rank}')
    
    @staticmethod
    def ddp_model(model: nn.Module, device: torch.device) -> nn.Module:
        """Wrap model in DDP container"""
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[device.index], output_device=device.index)
    
    @staticmethod
    def is_main_process() -> bool:
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

# ----------------------
# Quantum-Inspired Utilities
# ----------------------
class QuantumUtils:
    """Quantum-inspired algorithms for classical ML"""
    @staticmethod
    def quantum_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Quantum-inspired attention mechanism"""
        # Apply quantum rotation
        q = QuantumUtils.apply_quantum_rotation(q)
        k = QuantumUtils.apply_quantum_rotation(k)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_probs = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_probs, v)
    
    @staticmethod
    def apply_quantum_rotation(tensor: torch.Tensor) -> torch.Tensor:
        """Apply quantum rotation to tensors"""
        # Create rotation angles
        angles = torch.linspace(0, 2 * math.pi, tensor.size(-1), device=tensor.device)
        
        # Apply rotation using real numbers only
        real = tensor * torch.cos(angles)
        imag = tensor * torch.sin(angles)
        return real + imag  # Use real number representation instead of complex

# ----------------------
# Debugging Utilities
# ----------------------
class DebugUtils:
    """Advanced debugging utilities"""
    @staticmethod
    def enable_debug_mode():
        """Enable comprehensive debug mode"""
        torch.autograd.set_detect_anomaly(True)
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.set_printoptions(precision=10, threshold=10000, linewidth=200)
    
    @staticmethod
    def tensor_debugger_hook(grad):
        """Gradient debugging hook"""
        print(f"Gradient stats - min: {grad.min().item()}, max: {grad.max().item()}, "
              f"mean: {grad.mean().item()}, isnan: {torch.isnan(grad).any().item()}")
        return grad

# ----------------------
# Environment Utilities
# ----------------------
class EnvUtils:
    """Environment and system utilities"""
    @staticmethod
    def get_environment_report() -> dict:
        """Generate comprehensive environment report"""
        return {
            'system': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'memory': psutil.virtual_memory()._asdict(),
            'disk': psutil.disk_usage('/')._asdict(),
            'environment_vars': {k: v for k, v in os.environ.items() if 'PYTHON' in k or 'CUDA' in k}
        }
    
    @staticmethod
    def setup_reproducibility(seed: int = 42):
        """Ensure full reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

# ----------------------
# Decorators
# ----------------------
def timed_execution(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        if DEVICE.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        result = func(*args, **kwargs)
        
        if DEVICE.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            print(f"{func.__name__} executed in {elapsed:.2f}ms (CUDA timed)")
        else:
            elapsed = (time.perf_counter() - start_time) * 1000
            print(f"{func.__name__} executed in {elapsed:.2f}ms")
            
        return result
    return wrapper

def memory_profiler(func):
    """Decorator to profile memory usage"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEVICE.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        result = func(*args, **kwargs)
        
        if DEVICE.type == 'cuda':
            mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"{func.__name__} used {mem_used:.2f} MB GPU memory")
        return result
    return wrapper

# Initialize critical subsystems
DEVICE_MANAGER.auto_configure()
WEB_UTILS = WebUtils()
