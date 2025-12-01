"""
Resource Monitor Module
Monitors CPU, GPU, RAM, and disk usage during training with adaptive sampling.
"""

import psutil
import torch
import threading
import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np


class ResourceMonitor:
    """Monitor CPU, GPU, RAM, and disk usage during training with smart adaptive sampling."""
    
    def __init__(self, log_dir=None, interval=1.0, adaptive=True):
        """
        Initialize resource monitor.
        
        Args:
            log_dir: Directory to save logs
            interval: Base sampling interval in seconds
            adaptive: If True, increases interval during stable periods
        """
        self.base_interval = interval
        self.current_interval = interval
        self.adaptive = adaptive
        self.monitoring = False
        self.monitor_thread = None
        
        self.stats = {
            'cpu_percent': [],
            'cpu_per_core': [],
            'ram_percent': [],
            'ram_used_gb': [],
            'ram_available_gb': [],
            'gpu_utilization': [],
            'gpu_memory_used_gb': [],
            'gpu_memory_total_gb': [],
            'gpu_memory_percent': [],
            'gpu_temp': [],
            'gpu_power_watts': [],
            'disk_io_read_mb': [],
            'disk_io_write_mb': [],
            'disk_io_read_rate_mbs': [],
            'disk_io_write_rate_mbs': [],
            'timestamps': [],
            'elapsed_seconds': []
        }
        
        self.log_dir = Path(log_dir) if log_dir else Path('logs/resources')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability - PyTorch first (works for AMD and NVIDIA)
        self.has_gpu = torch.cuda.is_available()
        self.use_pynvml = False
        self.gpu_name = "Unknown GPU"
        self.gpu_total_memory_gb = 0

        if self.has_gpu:
            # Get GPU info from PyTorch (works for AMD/NVIDIA)
            try:
                props = torch.cuda.get_device_properties(0)
                self.gpu_name = props.name
                self.gpu_total_memory_gb = props.total_memory / (1024**3)
            except:
                pass

            # Try to enhance with NVML for NVIDIA GPUs (optional)
            # This gives us utilization %, temperature, and power metrics
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.use_pynvml = True
                print(f"[OK] NVIDIA GPU detected - Enhanced monitoring available")
            except Exception:
                # AMD GPU or NVML not available - that's fine!
                self.use_pynvml = False
                print(f"[OK] AMD/ROCm GPU detected - Using PyTorch metrics")

        self.initial_disk_io = psutil.disk_io_counters()
        self.last_disk_io = self.initial_disk_io
        self.last_disk_time = time.time()
        self.start_time = None
        
        # For adaptive sampling
        self.recent_changes = []
        self.stability_threshold = 5.0  # Percent change threshold
        self.stability_checks = 0
    
    def start(self):
        """Start monitoring in background thread."""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        gpu_info = f" (GPU: {self.gpu_name})" if self.has_gpu else ""
        print(f" Resource monitoring started{gpu_info}")
        if self.adaptive:
            print(f"   Adaptive sampling enabled (base interval: {self.base_interval}s)")
    
    def stop(self):
        """Stop monitoring and save results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self._save_stats()
        self._print_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop with adaptive sampling."""
        while self.monitoring:
            self._collect_stats()
            
            # Adaptive interval adjustment
            if self.adaptive and len(self.stats['cpu_percent']) > 10:
                self._adjust_interval()
            
            time.sleep(self.current_interval)
    
    def _adjust_interval(self):
        """Adjust sampling interval based on recent stability."""
        if len(self.stats['cpu_percent']) < 5:
            return
        
        # Calculate recent variance in key metrics
        recent_cpu = self.stats['cpu_percent'][-5:]
        recent_gpu = self.stats['gpu_utilization'][-5:] if self.has_gpu else [0]
        
        cpu_change = max(recent_cpu) - min(recent_cpu)
        gpu_change = max(recent_gpu) - min(recent_gpu) if self.has_gpu else 0
        
        max_change = max(cpu_change, gpu_change)
        
        # If stable, increase interval (up to 5x base)
        # If volatile, decrease interval (down to base)
        if max_change < self.stability_threshold:
            self.stability_checks += 1
            if self.stability_checks >= 3:
                self.current_interval = min(self.current_interval * 1.5, self.base_interval * 5)
                self.stability_checks = 0
        else:
            self.current_interval = self.base_interval
            self.stability_checks = 0
    
    def _collect_stats(self):
        """Collect current resource statistics."""
        current_time = time.time()
        
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()
        
        self.stats['cpu_percent'].append(cpu_percent)
        self.stats['cpu_per_core'].append(cpu_per_core)
        self.stats['ram_percent'].append(mem.percent)
        self.stats['ram_used_gb'].append(mem.used / (1024**3))
        self.stats['ram_available_gb'].append(mem.available / (1024**3))
        
        # GPU stats
        if self.has_gpu:
            if self.use_pynvml:
                # NVIDIA GPU with enhanced metrics
                try:
                    import pynvml
                    
                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    self.stats['gpu_utilization'].append(util.gpu)
                    
                    # GPU memory
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.stats['gpu_memory_used_gb'].append(mem_info.used / (1024**3))
                    self.stats['gpu_memory_total_gb'].append(mem_info.total / (1024**3))
                    self.stats['gpu_memory_percent'].append((mem_info.used / mem_info.total) * 100)
                    
                    # GPU temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                        self.stats['gpu_temp'].append(temp)
                    except:
                        self.stats['gpu_temp'].append(0)
                    
                    # GPU power
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # mW to W
                        self.stats['gpu_power_watts'].append(power)
                    except:
                        self.stats['gpu_power_watts'].append(0)
                        
                except Exception:
                    self._append_gpu_pytorch_metrics()
            else:
                # AMD GPU or PyTorch-only monitoring
                self._append_gpu_pytorch_metrics()
        else:
            self._append_gpu_defaults()
        
        # Disk I/O with rate calculation
        disk_io = psutil.disk_io_counters()
        time_delta = current_time - self.last_disk_time
        
        if self.initial_disk_io and time_delta > 0:
            # Cumulative
            read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024**2)
            write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024**2)
            self.stats['disk_io_read_mb'].append(read_mb)
            self.stats['disk_io_write_mb'].append(write_mb)
            
            # Rate since last sample
            read_rate = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024**2) / time_delta
            write_rate = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024**2) / time_delta
            self.stats['disk_io_read_rate_mbs'].append(read_rate)
            self.stats['disk_io_write_rate_mbs'].append(write_rate)
        
        self.last_disk_io = disk_io
        self.last_disk_time = current_time
        
        # Timestamps
        self.stats['timestamps'].append(datetime.now().isoformat())
        if self.start_time:
            self.stats['elapsed_seconds'].append(current_time - self.start_time)
    
    def _append_gpu_pytorch_metrics(self):
        """Append GPU metrics using PyTorch (works for AMD and NVIDIA)."""
        try:
            # Memory allocated by tensors
            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            # Memory reserved by caching allocator
            mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            # Total GPU memory
            mem_total = self.gpu_total_memory_gb

            # Use reserved memory (more accurate for actual GPU usage)
            self.stats['gpu_memory_used_gb'].append(mem_reserved)
            self.stats['gpu_memory_total_gb'].append(mem_total)

            # Calculate percentage if we have total memory
            if mem_total > 0:
                self.stats['gpu_memory_percent'].append((mem_reserved / mem_total) * 100)
            else:
                self.stats['gpu_memory_percent'].append(0)

            # GPU utilization not available via PyTorch
            self.stats['gpu_utilization'].append(0)

            # Temperature and power not available via PyTorch
            self.stats['gpu_temp'].append(0)
            self.stats['gpu_power_watts'].append(0)
        except Exception:
            self._append_gpu_defaults()

    def _append_gpu_defaults(self):
        """Append default GPU values when GPU not available."""
        self.stats['gpu_utilization'].append(0)
        self.stats['gpu_memory_used_gb'].append(0)
        self.stats['gpu_memory_total_gb'].append(0)
        self.stats['gpu_memory_percent'].append(0)
        self.stats['gpu_temp'].append(0)
        self.stats['gpu_power_watts'].append(0)
    
    def _save_stats(self):
        """Save collected statistics to file."""
        if not self.stats['timestamps']:
            print("[WARNING]  No stats collected to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON (convert lists to regular Python types for JSON serialization)
        json_stats = {}
        for key, value in self.stats.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert numpy arrays to lists if needed
                if isinstance(value[0], (list, np.ndarray)):
                    json_stats[key] = [list(v) if isinstance(v, np.ndarray) else v for v in value]
                else:
                    json_stats[key] = value
            else:
                json_stats[key] = value
        
        json_path = self.log_dir / f"resource_stats_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f" Resource stats saved to: {json_path}")
    
    def _print_summary(self):
        """Print summary of resource utilization."""
        if not self.stats['cpu_percent']:
            return
        
        print("\n" + "="*80)
        print(" " * 25 + "RESOURCE UTILIZATION SUMMARY")
        print("="*80)
        
        # Duration
        if self.stats['elapsed_seconds']:
            duration = self.stats['elapsed_seconds'][-1]
            print(f"\n  Monitoring Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
            print(f"   Samples collected: {len(self.stats['cpu_percent'])}")
        
        # CPU
        if self.stats['cpu_percent']:
            avg_cpu = np.mean(self.stats['cpu_percent'])
            max_cpu = np.max(self.stats['cpu_percent'])
            min_cpu = np.min(self.stats['cpu_percent'])
            print(f"\n CPU Usage:")
            print(f"   Average: {avg_cpu:.1f}%")
            print(f"   Peak: {max_cpu:.1f}%")
            print(f"   Minimum: {min_cpu:.1f}%")
            print(f"   CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        
        # RAM
        if self.stats['ram_percent']:
            avg_ram = np.mean(self.stats['ram_percent'])
            max_ram = np.max(self.stats['ram_percent'])
            max_ram_gb = np.max(self.stats['ram_used_gb'])
            print(f"\n RAM Usage:")
            print(f"   Average: {avg_ram:.1f}%")
            print(f"   Peak: {max_ram:.1f}% ({max_ram_gb:.2f} GB)")
            print(f"   Total RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
        # GPU
        if self.has_gpu:
            print(f"\n GPU: {self.gpu_name}")

            # GPU Utilization (NVIDIA only via NVML)
            if self.use_pynvml and self.stats['gpu_utilization']:
                gpu_utils = [x for x in self.stats['gpu_utilization'] if x > 0]
                if gpu_utils:
                    avg_gpu = np.mean(gpu_utils)
                    max_gpu = np.max(gpu_utils)
                    print(f"   Utilization:")
                    print(f"     Average: {avg_gpu:.1f}%")
                    print(f"     Peak: {max_gpu:.1f}%")

            # GPU Memory (Works for both AMD and NVIDIA via PyTorch)
            if self.stats['gpu_memory_used_gb']:
                gpu_mem_used = np.array(self.stats['gpu_memory_used_gb'])
                gpu_mem_pct = np.array(self.stats['gpu_memory_percent'])

                # Filter out zero values
                nonzero_mem = gpu_mem_used[gpu_mem_used > 0]
                nonzero_pct = gpu_mem_pct[gpu_mem_pct > 0]

                if len(nonzero_mem) > 0:
                    avg_gpu_mem = np.mean(nonzero_mem)
                    max_gpu_mem = np.max(nonzero_mem)
                    print(f"   Memory Usage:")
                    print(f"     Average: {avg_gpu_mem:.2f} GB", end="")
                    if len(nonzero_pct) > 0:
                        print(f" ({np.mean(nonzero_pct):.1f}%)")
                    else:
                        print()
                    print(f"     Peak: {max_gpu_mem:.2f} GB", end="")
                    if len(nonzero_pct) > 0:
                        print(f" ({np.max(nonzero_pct):.1f}%)")
                    else:
                        print()
                    if self.gpu_total_memory_gb > 0:
                        print(f"     Total: {self.gpu_total_memory_gb:.2f} GB")

            # Temperature and Power (NVIDIA only via NVML)
            if self.use_pynvml:
                gpu_temps = [x for x in self.stats['gpu_temp'] if x > 0]
                if gpu_temps:
                    avg_temp = np.mean(gpu_temps)
                    max_temp = np.max(gpu_temps)
                    print(f"   Temperature:")
                    print(f"     Average: {avg_temp:.1f}°C")
                    print(f"     Peak: {max_temp:.1f}°C")

                gpu_powers = [x for x in self.stats['gpu_power_watts'] if x > 0]
                if gpu_powers:
                    avg_power = np.mean(gpu_powers)
                    max_power = np.max(gpu_powers)
                    print(f"   Power:")
                    print(f"     Average: {avg_power:.1f}W")
                    print(f"     Peak: {max_power:.1f}W")

        # Disk I/O
        if self.stats['disk_io_read_mb']:
            total_read = self.stats['disk_io_read_mb'][-1]
            total_write = self.stats['disk_io_write_mb'][-1]
            avg_read_rate = np.mean(self.stats['disk_io_read_rate_mbs'])
            avg_write_rate = np.mean(self.stats['disk_io_write_rate_mbs'])
            max_read_rate = np.max(self.stats['disk_io_read_rate_mbs'])
            max_write_rate = np.max(self.stats['disk_io_write_rate_mbs'])
            
            print(f"\n Disk I/O:")
            print(f"   Total Read: {total_read:.2f} MB")
            print(f"   Total Write: {total_write:.2f} MB")
            print(f"   Avg Read Rate: {avg_read_rate:.2f} MB/s")
            print(f"   Avg Write Rate: {avg_write_rate:.2f} MB/s")
            print(f"   Peak Read Rate: {max_read_rate:.2f} MB/s")
            print(f"   Peak Write Rate: {max_write_rate:.2f} MB/s")
        
        print("\n" + "="*80 + "\n")
    
    def get_current_stats(self):
        """Get current resource usage as dict."""
        stats = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if self.has_gpu:
            if self.use_pynvml:
                # NVIDIA GPU with enhanced metrics
                try:
                    import pynvml
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)

                    stats['gpu_utilization'] = util.gpu
                    stats['gpu_memory_gb'] = mem_info.used / (1024**3)
                    stats['gpu_memory_percent'] = (mem_info.used / mem_info.total) * 100
                    stats['gpu_temp'] = temp
                except:
                    pass
            else:
                # AMD GPU using PyTorch metrics
                try:
                    mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    mem_total = self.gpu_total_memory_gb

                    stats['gpu_memory_gb'] = mem_reserved
                    if mem_total > 0:
                        stats['gpu_memory_percent'] = (mem_reserved / mem_total) * 100
                except:
                    pass

        return stats
    
    def print_live_stats(self):
        """Print current stats in a compact format (useful for periodic updates)."""
        stats = self.get_current_stats()
        
        msg = f" CPU: {stats['cpu_percent']:.1f}% | "
        msg += f" RAM: {stats['ram_percent']:.1f}% ({stats['ram_used_gb']:.1f}GB)"
        
        if 'gpu_memory_gb' in stats:
            msg += f" |  GPU Mem: {stats['gpu_memory_gb']:.2f}GB"
            if 'gpu_memory_percent' in stats:
                msg += f" ({stats['gpu_memory_percent']:.1f}%)"
            if 'gpu_utilization' in stats:
                msg += f" Util: {stats['gpu_utilization']:.1f}%"
            if 'gpu_temp' in stats:
                msg += f" {stats['gpu_temp']:.0f}°C"

        print(msg)

