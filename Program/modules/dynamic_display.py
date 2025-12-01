"""
Dynamic Display Module
Provides in-place updating progress bars and status displays without scrolling.
"""

import sys
import time


class DynamicDisplay:
    """Manages dynamic, non-scrolling display updates."""

    def __init__(self, num_lines=3):
        """
        Initialize dynamic display.

        Args:
            num_lines: Number of lines to reserve for dynamic updates
        """
        self.num_lines = num_lines
        self.last_lines = [''] * num_lines
        self.is_initialized = False

    def update(self, lines):
        """
        Update the display with new content.

        Args:
            lines: List of strings to display (will be padded/truncated to num_lines)
        """
        # Ensure we have the right number of lines
        while len(lines) < self.num_lines:
            lines.append('')
        lines = lines[:self.num_lines]

        # Move cursor up if not first time
        if self.is_initialized:
            # Move cursor up num_lines
            sys.stdout.write(f'\033[{self.num_lines}A')
        else:
            self.is_initialized = True

        # Clear and write each line
        for i, line in enumerate(lines):
            # Clear line and write new content
            sys.stdout.write('\r\033[K' + line + '\n')

        # Move back to ensure we're at the right position
        sys.stdout.flush()
        self.last_lines = lines.copy()

    def clear(self):
        """Clear the dynamic display."""
        if self.is_initialized:
            sys.stdout.write(f'\033[{self.num_lines}A')
            for _ in range(self.num_lines):
                sys.stdout.write('\r\033[K\n')
            sys.stdout.flush()
            self.is_initialized = False

    def finalize(self):
        """Finalize the display (keep last content, allow scrolling)."""
        # Just ensure we're on a new line after the dynamic content
        if self.is_initialized:
            sys.stdout.write('\n')
            sys.stdout.flush()
            self.is_initialized = False


class ProgressDisplay:
    """Enhanced progress bar with resource monitoring."""

    def __init__(self, total, prefix='Progress', show_resources=True):
        """
        Initialize progress display.

        Args:
            total: Total number of items
            prefix: Prefix text for the progress bar
            show_resources: Whether to show resource stats
        """
        self.total = total
        self.current = 0
        self.prefix = prefix
        self.show_resources = show_resources
        self.start_time = time.time()
        self.display = DynamicDisplay(num_lines=2 if show_resources else 1)
        self.resource_monitor = None

    def set_resource_monitor(self, monitor):
        """Set resource monitor for live stats."""
        self.resource_monitor = monitor

    def update(self, current, suffix='', bar_length=40):
        """
        Update progress bar.

        Args:
            current: Current progress value
            suffix: Additional suffix text
            bar_length: Length of the progress bar
        """
        self.current = current

        # Calculate progress
        percent = 100 * (current / float(self.total))
        filled_length = int(bar_length * current // self.total)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        # Calculate ETA
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = elapsed * (self.total - current) / current
            eta_str = self._format_time(eta)
        else:
            eta_str = '--:--'

        # Build progress line
        progress_line = f'{self.prefix} |{bar}| {percent:.1f}% ({current}/{self.total}) ETA: {eta_str}'
        if suffix:
            progress_line += f' | {suffix}'

        lines = [progress_line]

        # Add resource line if enabled
        if self.show_resources and self.resource_monitor:
            try:
                stats = self.resource_monitor.get_current_stats()
                resource_line = f" CPU: {stats['cpu_percent']:.1f}% |  RAM: {stats['ram_percent']:.1f}% ({stats['ram_used_gb']:.1f}GB)"

                if 'gpu_memory_gb' in stats:
                    resource_line += f" |  GPU: {stats['gpu_memory_gb']:.2f}GB"
                    if 'gpu_memory_percent' in stats:
                        resource_line += f" ({stats['gpu_memory_percent']:.1f}%)"

                lines.append(resource_line)
            except:
                lines.append("Resources: N/A")

        self.display.update(lines)

    def _format_time(self, seconds):
        """Format seconds into MM:SS."""
        if seconds < 0:
            return '--:--'
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f'{mins:02d}:{secs:02d}'

    def finish(self):
        """Finalize the progress display."""
        self.display.finalize()


class EpochDisplay:
    """Display for epoch-level training information."""

    def __init__(self, total_epochs, show_resources=True):
        """
        Initialize epoch display.

        Args:
            total_epochs: Total number of epochs
            show_resources: Whether to show resource monitoring
        """
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.show_resources = show_resources
        self.display = DynamicDisplay(num_lines=3 if show_resources else 2)
        self.resource_monitor = None
        self.start_time = time.time()

    def set_resource_monitor(self, monitor):
        """Set resource monitor for live stats."""
        self.resource_monitor = monitor

    def update(self, epoch, phase, batch_info='', metrics=''):
        """
        Update epoch display.

        Args:
            epoch: Current epoch number (1-indexed)
            phase: Training phase (e.g., 'Training', 'Validation', 'Stage 1', 'Stage 2')
            batch_info: Batch progress info (e.g., '150/500')
            metrics: Current metrics (e.g., 'Loss: 0.5432 | Acc: 0.8765')
        """
        self.current_epoch = epoch

        # Epoch line
        epoch_line = f" Epoch {epoch}/{self.total_epochs} - {phase}"
        if batch_info:
            epoch_line += f" [{batch_info}]"

        # Metrics line
        metrics_line = f"   {metrics}" if metrics else ""

        lines = [epoch_line, metrics_line]

        # Resource line
        if self.show_resources and self.resource_monitor:
            try:
                stats = self.resource_monitor.get_current_stats()
                resource_line = f"    CPU: {stats['cpu_percent']:.1f}% |  RAM: {stats['ram_percent']:.1f}% ({stats['ram_used_gb']:.1f}GB)"

                if 'gpu_memory_gb' in stats:
                    resource_line += f" |  GPU: {stats['gpu_memory_gb']:.2f}GB"
                    if 'gpu_memory_percent' in stats:
                        resource_line += f" ({stats['gpu_memory_percent']:.1f}%)"

                lines.append(resource_line)
            except:
                lines.append("   Resources: N/A")

        self.display.update(lines)

    def finish(self):
        """Finalize the epoch display."""
        self.display.finalize()


class TrainingDisplay:
    """Combined training display with epoch and batch progress."""

    def __init__(self, total_epochs, show_resources=True):
        """
        Initialize training display.

        Args:
            total_epochs: Total number of epochs
            show_resources: Whether to show resource stats
        """
        self.total_epochs = total_epochs
        self.show_resources = show_resources
        # 2 lines: progress bar + resource monitor
        self.display = DynamicDisplay(num_lines=2)
        self.resource_monitor = None
        self.current_epoch = 0
        self.current_phase = ""
        self.batch_progress = ""
        self.metrics = ""
        self.start_time = time.time()

    def set_resource_monitor(self, monitor):
        """Set resource monitor for live stats."""
        self.resource_monitor = monitor

    def update(self, epoch=None, phase=None, batch_current=None, batch_total=None,
               metrics=None, bar_length=40, eta_str='--:--'):
        """
        Update the full training display.

        Args:
            epoch: Current epoch (1-indexed)
            phase: Phase name ('Training', 'Validation', etc.)
            batch_current: Current batch number
            batch_total: Total batches
            metrics: Metrics string (e.g., 'Loss: 0.5432 | Acc: 0.8765')
            bar_length: Progress bar length
            eta_str: ETA string
        """
        if epoch is not None:
            self.current_epoch = epoch
        if phase is not None:
            self.current_phase = phase
        if metrics is not None:
            self.metrics = metrics

        lines = []

        # Line 1: Progress bar with epoch, phase, batch progress, metrics, and ETA
        if batch_current is not None and batch_total is not None:
            percent = 100 * (batch_current / float(batch_total))
            filled_length = int(bar_length * batch_current // batch_total)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Build comprehensive progress line
            progress_line = f"Epoch {self.current_epoch}/{self.total_epochs} - {self.current_phase} "
            progress_line += f"|{bar}| {percent:.1f}% ({batch_current}/{batch_total})"

            if self.metrics:
                progress_line += f" | {self.metrics}"

            if eta_str and eta_str != '--:--':
                progress_line += f" | ETA: {eta_str}"

            lines.append(progress_line)
        else:
            # Fallback if no batch info
            lines.append(f"Epoch {self.current_epoch}/{self.total_epochs} - {self.current_phase}")

        # Line 2: Resource monitor
        if self.show_resources and self.resource_monitor:
            try:
                stats = self.resource_monitor.get_current_stats()
                resource_line = f" CPU: {stats['cpu_percent']:.1f}% |  RAM: {stats['ram_percent']:.1f}% ({stats['ram_used_gb']:.1f}GB)"

                if 'gpu_memory_gb' in stats:
                    resource_line += f" |  GPU Mem: {stats['gpu_memory_gb']:.2f}GB"
                    if 'gpu_memory_percent' in stats:
                        resource_line += f" ({stats['gpu_memory_percent']:.1f}%)"
                    if 'gpu_temp' in stats and stats.get('gpu_temp', 0) > 0:
                        resource_line += f" |  {stats['gpu_temp']}°C"

                lines.append(resource_line)
            except Exception:
                lines.append("Resources: N/A")
        else:
            lines.append("")  # Empty line to maintain 2-line format

        self.display.update(lines)

    def finish(self):
        """Finalize the training display."""
        self.display.finalize()

