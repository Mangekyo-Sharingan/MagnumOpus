#!/usr/bin/env python3
"""
View and analyze saved resource monitoring data.
Usage: python view_resource_stats.py [path_to_json]
"""

import sys
import json
from pathlib import Path
import argparse


def load_stats(json_path):
    """Load stats from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_summary(stats):
    """Print detailed summary of stats."""
    import numpy as np
    
    print("\n" + "="*80)
    print(" "*25 + "RESOURCE MONITORING ANALYSIS")
    print("="*80)
    
    # Basic info
    if stats['timestamps']:
        print(f"\n Session:")
        print(f"   Start: {stats['timestamps'][0]}")
        print(f"   End: {stats['timestamps'][-1]}")
        print(f"   Samples: {len(stats['timestamps'])}")
    
    if stats['elapsed_seconds']:
        duration = stats['elapsed_seconds'][-1]
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    # CPU
    if stats['cpu_percent']:
        cpu_arr = np.array(stats['cpu_percent'])
        print(f"\n CPU Utilization:")
        print(f"   Mean: {cpu_arr.mean():.1f}%")
        print(f"   Std: {cpu_arr.std():.1f}%")
        print(f"   Min: {cpu_arr.min():.1f}%")
        print(f"   Max: {cpu_arr.max():.1f}%")
        print(f"   Median: {np.median(cpu_arr):.1f}%")
        
        # Efficiency rating
        if cpu_arr.mean() < 30:
            rating = "[WARNING]  LOW - Consider increasing workload"
        elif cpu_arr.mean() < 70:
            rating = "[OK] MODERATE - Good balance"
        else:
            rating = " HIGH - CPU intensive"
        print(f"   Rating: {rating}")
    
    # RAM
    if stats['ram_percent']:
        ram_pct = np.array(stats['ram_percent'])
        ram_gb = np.array(stats['ram_used_gb'])
        print(f"\n RAM Usage:")
        print(f"   Mean: {ram_pct.mean():.1f}% ({ram_gb.mean():.2f} GB)")
        print(f"   Peak: {ram_pct.max():.1f}% ({ram_gb.max():.2f} GB)")
        print(f"   Min: {ram_pct.min():.1f}% ({ram_gb.min():.2f} GB)")
        
        if ram_pct.max() > 90:
            print(f"   [WARNING]  WARNING: Peak RAM > 90% - Risk of OOM")
        elif ram_pct.max() > 80:
            print(f"   [WARNING]  CAUTION: Peak RAM > 80% - Monitor closely")
        else:
            print(f"   [OK] RAM usage healthy")
    
    # GPU
    if stats['gpu_utilization']:
        gpu_util = np.array([x for x in stats['gpu_utilization'] if x > 0])
        if len(gpu_util) > 0:
            print(f"\n GPU Utilization:")
            print(f"   Mean: {gpu_util.mean():.1f}%")
            print(f"   Peak: {gpu_util.max():.1f}%")
            
            if gpu_util.mean() < 50:
                print(f"   [WARNING]  LOW - GPU underutilized, consider increasing batch size")
            elif gpu_util.mean() < 80:
                print(f"   [OK] MODERATE - Good utilization")
            else:
                print(f"    HIGH - Excellent GPU utilization")
    
    if stats['gpu_memory_used_gb']:
        gpu_mem = np.array(stats['gpu_memory_used_gb'])
        gpu_mem_pct = np.array(stats['gpu_memory_percent'])
        print(f"\n GPU Memory:")
        print(f"   Mean: {gpu_mem.mean():.2f} GB", end="")
        if gpu_mem_pct.mean() > 0:
            print(f" ({gpu_mem_pct.mean():.1f}%)")
        else:
            print()
        print(f"   Peak: {gpu_mem.max():.2f} GB", end="")
        if gpu_mem_pct.max() > 0:
            print(f" ({gpu_mem_pct.max():.1f}%)")
        else:
            print()
    
    # GPU Temperature
    if stats['gpu_temp']:
        gpu_temp = np.array([x for x in stats['gpu_temp'] if x > 0])
        if len(gpu_temp) > 0:
            print(f"\n  GPU Temperature:")
            print(f"   Mean: {gpu_temp.mean():.1f}°C")
            print(f"   Peak: {gpu_temp.max():.1f}°C")
            
            if gpu_temp.max() > 85:
                print(f"    WARNING: Temperature > 85°C - Check cooling!")
            elif gpu_temp.max() > 75:
                print(f"   [WARNING]  WARM: Temperature > 75°C - Monitor closely")
            else:
                print(f"   [OK] Temperature healthy")
    
    # Disk I/O
    if stats['disk_io_read_mb'] and stats['disk_io_write_mb']:
        total_read = stats['disk_io_read_mb'][-1]
        total_write = stats['disk_io_write_mb'][-1]
        
        read_rate = np.array(stats['disk_io_read_rate_mbs'])
        write_rate = np.array(stats['disk_io_write_rate_mbs'])
        
        print(f"\n Disk I/O:")
        print(f"   Total Read: {total_read:.2f} MB")
        print(f"   Total Write: {total_write:.2f} MB")
        print(f"   Avg Read Rate: {read_rate.mean():.2f} MB/s")
        print(f"   Avg Write Rate: {write_rate.mean():.2f} MB/s")
        print(f"   Peak Read Rate: {read_rate.max():.2f} MB/s")
        print(f"   Peak Write Rate: {write_rate.max():.2f} MB/s")
    
    print("\n" + "="*80 + "\n")


def plot_stats(stats, save_path=None):
    """Plot stats using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[WARNING]  matplotlib not installed. Install with: pip install matplotlib")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Resource Utilization Over Time', fontsize=16, fontweight='bold')
    
    time = stats['elapsed_seconds'] if stats['elapsed_seconds'] else range(len(stats['cpu_percent']))
    
    # CPU
    ax = axes[0, 0]
    ax.plot(time, stats['cpu_percent'], linewidth=1.5, color='#2196F3')
    ax.fill_between(time, stats['cpu_percent'], alpha=0.3, color='#2196F3')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('CPU Utilization (%)')
    ax.set_title('CPU Usage', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # RAM
    ax = axes[0, 1]
    ax.plot(time, stats['ram_percent'], linewidth=1.5, color='#4CAF50')
    ax.fill_between(time, stats['ram_percent'], alpha=0.3, color='#4CAF50')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('RAM Usage (%)')
    ax.set_title('Memory Usage', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # GPU Utilization
    ax = axes[1, 0]
    if any(x > 0 for x in stats['gpu_utilization']):
        ax.plot(time, stats['gpu_utilization'], linewidth=1.5, color='#FF9800')
        ax.fill_between(time, stats['gpu_utilization'], alpha=0.3, color='#FF9800')
        ax.set_ylim(0, 100)
    else:
        ax.text(0.5, 0.5, 'GPU Utilization\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Usage', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # GPU Memory
    ax = axes[1, 1]
    if any(x > 0 for x in stats['gpu_memory_used_gb']):
        ax.plot(time, stats['gpu_memory_used_gb'], linewidth=1.5, color='#9C27B0')
        ax.fill_between(time, stats['gpu_memory_used_gb'], alpha=0.3, color='#9C27B0')
    else:
        ax.text(0.5, 0.5, 'GPU Memory\nNot Available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('GPU Memory (GB)')
    ax.set_title('GPU Memory Usage', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Plot saved to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='View resource monitoring stats')
    parser.add_argument('json_file', nargs='?', help='Path to JSON stats file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-plot', help='Save plot to file instead of displaying')
    parser.add_argument('--latest', action='store_true', help='Use latest stats file')
    
    args = parser.parse_args()
    
    # Find JSON file
    if args.latest:
        # Find latest file in logs/resources
        resources_dir = Path('logs/resources')
        if not resources_dir.exists():
            print("[FAIL] No logs/resources directory found")
            return 1
        
        json_files = sorted(resources_dir.glob('**/resource_stats_*.json'))
        if not json_files:
            print("[FAIL] No stats files found in logs/resources")
            return 1
        
        json_path = json_files[-1]
        print(f" Using latest file: {json_path}")
    elif args.json_file:
        json_path = Path(args.json_file)
    else:
        print("[FAIL] Please provide a JSON file path or use --latest")
        parser.print_help()
        return 1
    
    if not json_path.exists():
        print(f"[FAIL] File not found: {json_path}")
        return 1
    
    # Load and display
    print(f" Loading: {json_path}")
    stats = load_stats(json_path)
    print_summary(stats)
    
    # Plot if requested
    if args.plot or args.save_plot:
        plot_stats(stats, save_path=args.save_plot)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

