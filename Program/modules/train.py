"""
Training module for diabetic retinopathy classification models
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
import psutil
import random
from .dynamic_display import TrainingDisplay

# Bokeh imports for live plotting in browser
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import row
from bokeh.models import HoverTool
from bokeh.io import curdoc, push_notebook
import webbrowser
from pathlib import Path


class BokehLivePlotter:
    """Live plotter using Bokeh to display training metrics in browser"""

    def __init__(self, output_dir, model_name='model'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.html_file = self.output_dir / f'{model_name}_live_training.html'

        # Data storage
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

        # Browser opened flag
        self.browser_opened = False

    def update(self, epoch, train_loss, val_loss, train_acc, val_acc):
        """Update the plots with new data"""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)

        # Create plots
        self._create_plots()

        # Open browser on first update
        if not self.browser_opened:
            self.browser_opened = True
            file_url = 'file://' + str(self.html_file.absolute())

            print(f"\n Live training plot created: {self.html_file}")
            print(f"   URL: {file_url}")

            # Try to open browser
            try:
                import subprocess
                import platform

                system = platform.system()
                opened = False

                # Try webbrowser module first
                try:
                    result = webbrowser.open(file_url, new=2)  # new=2 opens in new tab
                    if result:
                        print("   [OK] Browser opened automatically")
                        opened = True
                except:
                    pass

                # Fallback: Try system-specific commands
                if not opened:
                    try:
                        if system == 'Linux':
                            subprocess.Popen(['xdg-open', str(self.html_file)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            print("   [OK] Opened with xdg-open")
                            opened = True
                        elif system == 'Darwin':  # macOS
                            subprocess.Popen(['open', str(self.html_file)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            print("   [OK] Opened with macOS 'open'")
                            opened = True
                        elif system == 'Windows':
                            subprocess.Popen(['start', str(self.html_file)], shell=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            print("   [OK] Opened with Windows 'start'")
                            opened = True
                    except:
                        pass

                if not opened:
                    print("   [WARNING]  Could not auto-open browser")
                    print(f"    Copy and paste this URL into your browser:")
                    print(f"      {file_url}")

            except Exception as e:
                print(f"   [WARNING]  Browser opening failed: {e}")
                print(f"    Manually open: {file_url}")

            print("    Refresh the browser page after each epoch to see updates")

    def _create_plots(self):
        """Create Bokeh plots and save to HTML"""
        # Configure output
        output_file(self.html_file, title=f"{self.model_name.upper()} - Training Progress")

        # Loss plot
        loss_plot = figure(
            title='Training and Validation Loss',
            x_axis_label='Epoch',
            y_axis_label='Loss',
            width=600,
            height=400,
            tools='pan,wheel_zoom,box_zoom,reset,save'
        )

        loss_plot.line(self.epochs, self.train_loss, legend_label='Training Loss',
                      line_width=2, color='navy', alpha=0.8)
        loss_plot.scatter(self.epochs, self.train_loss, size=6, color='navy', alpha=0.5)

        loss_plot.line(self.epochs, self.val_loss, legend_label='Validation Loss',
                      line_width=2, color='red', alpha=0.8)
        loss_plot.scatter(self.epochs, self.val_loss, size=6, color='red', alpha=0.5)

        loss_plot.legend.location = "top_right"
        loss_plot.legend.click_policy = "hide"

        # Add hover tool
        loss_hover = HoverTool(tooltips=[('Epoch', '@x'), ('Loss', '@y{0.0000}')])
        loss_plot.add_tools(loss_hover)

        # Accuracy plot
        acc_plot = figure(
            title='Training and Validation Accuracy',
            x_axis_label='Epoch',
            y_axis_label='Accuracy',
            width=600,
            height=400,
            tools='pan,wheel_zoom,box_zoom,reset,save'
        )

        acc_plot.line(self.epochs, self.train_acc, legend_label='Training Accuracy',
                     line_width=2, color='green', alpha=0.8)
        acc_plot.scatter(self.epochs, self.train_acc, size=6, color='green', alpha=0.5)

        acc_plot.line(self.epochs, self.val_acc, legend_label='Validation Accuracy',
                     line_width=2, color='orange', alpha=0.8)
        acc_plot.scatter(self.epochs, self.val_acc, size=6, color='orange', alpha=0.5)

        acc_plot.legend.location = "bottom_right"
        acc_plot.legend.click_policy = "hide"

        # Add hover tool
        acc_hover = HoverTool(tooltips=[('Epoch', '@x'), ('Accuracy', '@y{0.0000}')])
        acc_plot.add_tools(acc_hover)

        # Combine plots
        layout = row(loss_plot, acc_plot)

        # Save to HTML
        save(layout)


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples without altering class balance counts"""
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0, num_classes=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        # inputs: (N, C), targets: (N,)
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0 and self.num_classes is not None:
            # Create smoothed one-hot labels
            with torch.no_grad():
                true_dist = torch.zeros_like(inputs)
                true_dist.fill_(self.label_smoothing / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            # Focal modulation
            focal_weight = (1 - probs).pow(self.gamma)
            loss = -(true_dist * focal_weight * log_probs)
        else:
            # Gather log_probs for true classes
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = log_pt.exp()
            focal_weight = (1 - pt).pow(self.gamma)
            loss = -focal_weight * log_pt

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha_t = torch.as_tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
                alpha_t = alpha_t[targets]
                loss = alpha_t * loss
            else:
                loss = float(self.alpha) * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def _rand_bbox(size, lam):
    # size: (N, C, H, W)
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = (W * cut_rat).astype(int)
    cut_h = (H * cut_rat).astype(int)

    # uniform center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def apply_mixup(inputs, targets, alpha=0.2):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_a, targets_b = targets, targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def apply_cutmix(inputs, targets, alpha=1.0):
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    x1, y1, x2, y2 = _rand_bbox(inputs.size(), lam)
    inputs_cut = inputs.clone()
    inputs_cut[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
    # adjust lam to exactly match pixel ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (inputs.size(-1) * inputs.size(-2)))
    targets_a, targets_b = targets, targets[index]
    return inputs_cut, targets_a, targets_b, lam

class Trainer:
    """Class responsible for training CNN models"""

    def __init__(self, model, config, model_name=None, resource_monitor=None):
        self.model = model
        self.config = config
        self.model_name = model_name  # Store model name for transfer learning
        self.resource_monitor = resource_monitor  # Resource monitor for live stats
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Get model-specific learning rate
        if model_name and hasattr(config, 'get_model_config'):
            model_config = config.get_model_config(model_name)
            self.learning_rate = model_config.get('learning_rate', 0.001)
        else:
            self.learning_rate = 0.001  # Default fallback

        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        # Mixed Precision Training - GradScaler for FP16 (use torch.amp API)
        self.scaler = GradScaler('cuda') if (config.use_amp and self.device.type == 'cuda') else None
        if config.use_amp and self.device.type == 'cuda':
            print("[OK] Mixed Precision Training (FP16) enabled with GradScaler (torch.amp)")
        elif config.use_amp:
            print("[INFO]  AMP requested but CUDA not available; proceeding without AMP.")

        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': [],
            'samples_per_second': [],
            'gpu_memory_used': [],
            'training_stage': []  # Track which stage each epoch belongs to
        }

        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Early stopping tracking
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False

        # Progress tracking
        self.epoch_start_time = None
        self.total_start_time = None
        self.epoch_times = []
        self.total_samples_processed = 0
        
        # Transfer learning tracking
        self.current_stage = 1
        self.stage1_epochs = 0
        self.stage2_epochs = 0

        # Mixup/CutMix configuration
        self.mixup_alpha = 0.2
        self.cutmix_alpha = 1.0
        self.mix_prob = 0.5  # Probability of applying MixUp/CutMix

        # Label smoothing configuration
        self.label_smoothing = 0.0
        self.use_label_smoothing = False

        # Focal loss configuration
        self.use_focal_loss = False
        self.focal_gamma = 2.0
        self.focal_alpha = None

        # Unfreeze layers configuration
        self.unfreeze_layers_config = {
            'vgg16': 'classifier.6',
            'resnet50': 'layer4',
            'inceptionv3': 'Mixed_7c',
        }

        # Live plotting with Bokeh
        self.live_plotter = None
        self.use_live_plotting = True  # Enable by default

    def _get_memory_usage(self):
        """Get current memory usage (GPU if available, else RAM)"""
        try:
            if torch.cuda.is_available():
                memory_gb = torch.cuda.memory_allocated() / 1024**3
                max_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                return memory_gb, max_memory_gb
            else:
                process = psutil.Process()
                return process.memory_info().rss / 1024**3, 0.0
        except:
            return 0.0, 0.0

    def unfreeze_layers(self, model_name=None):
        """
        Unfreeze specific layers for Stage 2 fine-tuning (Transfer Learning)

        Args:
            model_name: Name of the model to determine which layers to unfreeze
        """
        if model_name is None:
            model_name = self.model_name

        if not model_name or model_name not in self.config.unfreeze_layers:
            print(f"[WARNING]  Warning: Model '{model_name}' not found in unfreeze configuration. Unfreezing all layers.")
            # Fallback: unfreeze everything
            for param in self.model.parameters():
                param.requires_grad = True
            return

        unfreeze_from = self.config.unfreeze_layers[model_name]
        print(f"\n{'='*80}")
        print(f" UNFREEZING LAYERS FOR STAGE 2 FINE-TUNING")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Unfreezing from layer: {unfreeze_from}")
        print(f"{'─'*80}")

        unfrozen_count = 0
        total_params = 0

        # Get the backbone (all models use self.backbone)
        backbone = self.model.backbone

        # Flag to start unfreezing
        should_unfreeze = False

        # Iterate through named parameters
        for name, param in backbone.named_parameters():
            total_params += 1

            # Check if we've reached the target layer
            if unfreeze_from in name:
                should_unfreeze = True

            # Unfreeze if we're past the target layer
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1

        print(f"[OK] Unfrozen {unfrozen_count}/{total_params} parameter groups")
        print(f"{'='*80}\n")

    def setup_training_components(self, stage=1, learning_rate=None):
        """
        Setup optimizer, loss function, and scheduler

        Args:
            stage: Training stage (1 = head only, 2 = fine-tuning)
            learning_rate: Override learning rate (for stage 2)
        """
        stage_name = "STAGE 1 (Head Training)" if stage == 1 else "STAGE 2 (Fine-Tuning)"
        print(f"\n{'─'*80}")
        print(f"  Setting up training components for {stage_name}")
        print(f"{'─'*80}")

        # Determine learning rate
        if learning_rate is None:
            learning_rate = self.learning_rate

        # Setup optimizer with differential learning rates and AdamW
        weight_decay = self.config.weight_decay if hasattr(self.config, 'weight_decay') else 0.01
        
        backbone_params = []
        head_params = []
        
        # Identify backbone vs head parameters that require grad
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check for common backbone names or if it's part of the feature extractor
                if "backbone" in name or "features" in name or "body" in name or "Mixed" in name: 
                     backbone_params.append(param)
                else:
                     head_params.append(param)
        
        if len(backbone_params) > 0:
             # Stage 2 or Full Training (Backbone is unfrozen)
             self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': learning_rate * 0.1}, # Lower LR for backbone
                {'params': head_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
             print(f"[OK] Optimizer: AdamW (Differential LR: Backbone={learning_rate*0.1:.6f}, Head={learning_rate:.6f})")
        else:
             # Stage 1 (Backbone frozen)
             self.optimizer = optim.AdamW(
                head_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
             print(f"[OK] Optimizer: AdamW (lr={learning_rate:.6f})")

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())

        print(f"  - Trainable parameters: {trainable_count:,} ({100*trainable_count/total_count:.1f}%)")
        print(f"  - Frozen parameters: {total_count - trainable_count:,}")

        # Setup loss function (Weighted CrossEntropyLoss)
        # Approximate weights based on typical DR distribution: [No DR, Mild, Moderate, Severe, Proliferative]
        # Counts: ~[25k, 2.5k, 5k, 1k, 1k] -> Weights: [0.5, 2.0, 1.0, 3.0, 4.0]
        class_weights = torch.tensor([0.5, 2.0, 1.0, 3.0, 4.0]).to(self.device)

        if self.config.use_focal_loss:
            self.criterion = FocalLoss(
                gamma=self.config.focal_gamma,
                alpha=self.config.focal_alpha,
                label_smoothing=self.config.label_smoothing if self.config.use_label_smoothing else 0.0,
                num_classes=self.config.num_classes,
            )
            print("[OK] Loss function: FocalLoss")
        else:
            # Use label smoothing if enabled (PyTorch >= 1.10)
            if self.config.use_label_smoothing and self.config.label_smoothing > 0:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=self.config.label_smoothing)
                print(f"[OK] Loss function: CrossEntropyLoss (Weighted, label_smoothing={self.config.label_smoothing})")
            else:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                print("[OK] Loss function: CrossEntropyLoss (Weighted)")

        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        print("[OK] Scheduler: ReduceLROnPlateau")
        print(f"{'─'*80}\n")

    def _format_time(self, seconds):
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def _estimate_remaining_time(self, current_epoch, total_epochs):
        """Estimate remaining training time based on average epoch time"""
        if not self.epoch_times:
            return "Calculating..."
        
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        remaining_seconds = avg_epoch_time * remaining_epochs
        
        return self._format_time(remaining_seconds)
    
    def _print_progress_bar(self, current, total, prefix='', suffix='', length=40, fill='█'):
        """Print a progress bar"""
        percent = 100 * (current / float(total))
        filled_length = int(length * current // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        print(f'\r{prefix} |{bar}| {percent:.1f}% {suffix}', end='', flush=True)
        if current == total:
            print()

    def _print_training_header(self, num_epochs, train_loader, val_loader):
        """Print a comprehensive training header with all details"""
        print("\n" + "=" * 100)
        print(" " * 35 + "TRAINING INITIALIZATION")
        print("=" * 100)
        print(f" Model Configuration:")
        print(f"   - Model Name      : {self.model.model_name}")
        print(f"   - Device          : {self.device}")
        if torch.cuda.is_available():
            print(f"   - GPU Name        : {torch.cuda.get_device_name(0)}")
            print(f"   - GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"\n Training Configuration:")
        print(f"   - Total Epochs    : {num_epochs}")
        print(f"   - Batch Size      : {self.config.batch_size if hasattr(self.config, 'batch_size') else 'N/A'}")
        print(f"   - Learning Rate   : {self.learning_rate}")
        print(f"   - Train Batches   : {len(train_loader)}")
        print(f"   - Val Batches     : {len(val_loader)}")
        print(f"   - Train Samples   : ~{len(train_loader) * self.config.batch_size}")
        print(f"   - Val Samples     : ~{len(val_loader) * self.config.batch_size}")
        print("=" * 100 + "\n")

    def _compute_loss(self, outputs, labels, aux_outputs=None,
                      targets_a=None, targets_b=None, lam: float = 1.0):
        """Compute loss supporting Inception aux and MixUp/CutMix blending."""
        if targets_a is None or targets_b is None:
            # Standard loss
            if aux_outputs is not None:
                return self.criterion(outputs, labels) + 0.4 * self.criterion(aux_outputs, labels)
            return self.criterion(outputs, labels)
        else:
            # Mixed targets
            if aux_outputs is not None:
                loss_main = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
                loss_aux = lam * self.criterion(aux_outputs, targets_a) + (1 - lam) * self.criterion(aux_outputs, targets_b)
                return loss_main + 0.4 * loss_aux
            else:
                return lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)

    def train_epoch(self, train_loader, epoch_num=None, total_epochs=None):
        """Train for one epoch with detailed progress tracking and optional MixUp/CutMix"""
        self.model.train()
        running_loss = 0.0
        running_corrects = 0  # keep as numeric
        total_samples = 0
        batch_times = []
        total_batches = len(train_loader)

        # Initialize dynamic display if epoch and total_epochs are provided
        display = None
        if epoch_num and total_epochs:
            display = TrainingDisplay(total_epochs=total_epochs, show_resources=bool(self.resource_monitor))
            if self.resource_monitor:
                display.set_resource_monitor(self.resource_monitor)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            batch_start = time.time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Optional MixUp/CutMix per-batch (does not change dataset sizes)
            targets_a = targets_b = None
            lam = 1.0
            self.mix_strategy = None
            if random.random() < self.config.mix_prob:
                if self.config.use_cutmix and random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = apply_cutmix(inputs, labels, self.config.cutmix_alpha)
                    self.mix_strategy = 'cutmix'
                elif self.config.use_mixup:
                    inputs, targets_a, targets_b, lam = apply_mixup(inputs, labels, self.config.mixup_alpha)
                    self.mix_strategy = 'mixup'

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed Precision Training with autocast (torch.amp) only on CUDA
            if self.device.type == 'cuda' and self.config.use_amp and self.scaler is not None:
                with autocast('cuda', enabled=True):
                    outputs = self.model(inputs)
                    aux_outputs = None
                    if isinstance(outputs, tuple):
                        outputs, aux_outputs = outputs

                    # Loss
                    loss = self._compute_loss(outputs, labels,
                                              aux_outputs=aux_outputs,
                                              targets_a=targets_a, targets_b=targets_b, lam=lam)

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()

                # Gradient clipping (unscale first for correct clipping)
                if self.config.use_gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training without mixed precision
                outputs = self.model(inputs)
                aux_outputs = None
                if isinstance(outputs, tuple):
                    outputs, aux_outputs = outputs

                # Loss
                loss = self._compute_loss(outputs, labels,
                                          aux_outputs=aux_outputs,
                                          targets_a=targets_a, targets_b=targets_b, lam=lam)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_value)

                self.optimizer.step()

            # Statistics
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            # Accumulate as Python int
            running_corrects += (preds == labels).sum().item()
            total_samples += inputs.size(0)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Dynamic progress display
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                current_loss = running_loss / total_samples
                current_acc = running_corrects / total_samples
                avg_batch_time = np.mean(batch_times[-10:]) if batch_times else 0
                remaining_batches = total_batches - (batch_idx + 1)
                eta_seconds = remaining_batches * avg_batch_time
                eta_str = self._format_time(eta_seconds)

                # Build metrics string
                metrics = f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}"
                if self.mix_strategy:
                    metrics += f" | Aug: {self.mix_strategy}"

                # Update display
                if display:
                    display.update(
                        epoch=epoch_num,
                        phase="Training",
                        batch_current=batch_idx + 1,
                        batch_total=total_batches,
                        metrics=metrics,
                        eta_str=eta_str
                    )
                else:
                    # Fallback to old progress bar
                    gpu_memory, max_gpu_memory = self._get_memory_usage()
                    memory_str = f" | GPU Mem: {gpu_memory:.2f}GB / {max_gpu_memory:.2f}GB" if gpu_memory > 0 else ""
                    suffix = f"{metrics} | ETA: {eta_str}{memory_str}"
                    prefix = "Training"
                    self._print_progress_bar(batch_idx + 1, total_batches, prefix=prefix, suffix=suffix, length=30)

        # Finalize display
        if display:
            display.finish()

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = (running_corrects / max(total_samples, 1))
        denom = max(1e-6, float(np.sum(batch_times)))
        samples_per_second = total_samples / denom
        self.training_history['samples_per_second'].append(samples_per_second)
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader, epoch_num=None, total_epochs=None):
        """Validate for one epoch with progress tracking"""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        total_batches = len(val_loader)

        # Initialize dynamic display if epoch and total_epochs are provided
        display = None
        if epoch_num and total_epochs:
            display = TrainingDisplay(total_epochs=total_epochs, show_resources=bool(self.resource_monitor))
            if self.resource_monitor:
                display.set_resource_monitor(self.resource_monitor)

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Mixed Precision inference with autocast (torch.amp) only on CUDA
                if self.device.type == 'cuda' and self.config.use_amp:
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        # Handle InceptionV3 auxiliary outputs (only use main output for validation)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = self.criterion(outputs, labels)
                else:
                    # Standard inference
                    outputs = self.model(inputs)
                    # Handle InceptionV3 auxiliary outputs (only use main output for validation)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels)

                # Statistics
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total_samples += inputs.size(0)
                
                # Dynamic progress display
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    current_loss = running_loss / total_samples
                    current_acc = running_corrects / total_samples

                    # Build metrics string
                    metrics = f"Loss: {current_loss:.4f} | Acc: {current_acc:.4f}"

                    # Update display
                    if display:
                        display.update(
                            epoch=epoch_num,
                            phase="Validation",
                            batch_current=batch_idx + 1,
                            batch_total=total_batches,
                            metrics=metrics
                        )
                    else:
                        # Fallback to old progress bar
                        prefix = "Validation"
                        self._print_progress_bar(batch_idx + 1, total_batches, prefix=prefix, suffix=metrics, length=30)

        # Finalize display
        if display:
            display.finish()

        epoch_loss = running_loss / max(total_samples, 1)
        epoch_acc = (running_corrects / max(total_samples, 1))

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=None):
        """
        Main training loop with TWO-STAGE TRANSFER LEARNING

        Stage 1: Train only the new classification head (frozen backbone)
        Stage 2: Fine-tune the entire network with reduced learning rate
        """
        if num_epochs is None:
            num_epochs = self.config.epochs

        # Calculate stage epochs
        if self.config.use_transfer_learning and self.model_name:
            self.stage1_epochs = int(num_epochs * self.config.stage1_percentage)
            self.stage2_epochs = num_epochs - self.stage1_epochs
            use_two_stage = True
        else:
            self.stage1_epochs = num_epochs
            self.stage2_epochs = 0
            use_two_stage = False

        print("\n" + "=" * 100)
        print(" " * 30 + "TRAINING INITIALIZATION")
        print("=" * 100)
        print(f" Model Configuration:")
        print(f"   - Model Name      : {self.model.model_name}")
        print(f"   - Device          : {self.device}")
        if torch.cuda.is_available():
            print(f"   - GPU Name        : {torch.cuda.get_device_name(0)}")
            print(f"   - GPU Memory      : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        print(f"\n Training Strategy:")
        if use_two_stage:
            print(f"   - Transfer Learning: ENABLED (Two-Stage Fine-Tuning)")
            print(f"   - Stage 1 (Head)   : {self.stage1_epochs} epochs @ LR {self.learning_rate:.6f}")
            print(f"   - Stage 2 (Full)   : {self.stage2_epochs} epochs @ LR {self.learning_rate * self.config.stage2_lr_multiplier:.6f}")
        else:
            print(f"   - Transfer Learning: DISABLED (Standard Training)")
            print(f"   - Total Epochs     : {num_epochs}")

        print(f"\n Training Configuration:")
        print(f"   - Total Epochs    : {num_epochs}")
        print(f"   - Batch Size      : {self.config.get_model_config(self.model_name).get('batch_size', 'N/A') if self.model_name else 'N/A'}")
        print(f"   - Base Learn Rate : {self.learning_rate}")
        print(f"   - Train Batches   : {len(train_loader)}")
        print(f"   - Val Batches     : {len(val_loader)}")
        batch_size = self.config.get_model_config(self.model_name).get('batch_size', 32) if self.model_name else 32
        print(f"   - Train Samples   : ~{len(train_loader) * batch_size}")
        print(f"   - Val Samples     : ~{len(val_loader) * batch_size}")
        print("=" * 100)

        self.total_start_time = time.time()
        best_epoch = 0

        # Initialize Bokeh live plotter for browser-based visualization
        print(f"\n{'─'*100}")
        print(" LIVE PLOTTING INITIALIZATION")
        print(f"{'─'*100}")
        print(f"use_live_plotting flag: {self.use_live_plotting}")

        if self.use_live_plotting:
            try:
                # Create output directory for live plots
                live_plots_dir = self.config.base_dir / "logs" / "live_plots"
                print(f"Creating live plots directory: {live_plots_dir}")
                live_plots_dir.mkdir(parents=True, exist_ok=True)

                print(f"Initializing BokehLivePlotter for model: {self.model_name or 'model'}")
                self.live_plotter = BokehLivePlotter(live_plots_dir, model_name=self.model_name or 'model')

                print(f"[OK] Live plotting enabled (Bokeh)")
                print(f"   Plot file: {self.live_plotter.html_file}")
                print(f"   Browser will open automatically after first epoch")
                print(f"   Refresh browser (F5) after each epoch to see updates")
            except Exception as e:
                print(f"[FAIL] ERROR: Could not enable live plotting: {e}")
                import traceback
                traceback.print_exc()
                print("   Continuing without live plots...")
                self.use_live_plotting = False
                self.live_plotter = None
        else:
            print(f"[WARNING]  Live plotting is DISABLED")
            print(f"   To enable, set trainer.use_live_plotting = True")

        print(f"{'─'*100}\n")

        # ============================================================================
        # STAGE 1: Train only the classification head (backbone frozen)
        # ============================================================================
        if self.stage1_epochs > 0:
            self.current_stage = 1
            print(f"\n{''*40}")
            print(f"{'  '*15}STAGE 1: HEAD TRAINING")
            print(f"{''*40}")
            print(f"Training only the classification head with frozen backbone")
            print(f"Epochs: {self.stage1_epochs} | Learning Rate: {self.learning_rate:.6f}")
            print(f"{'─'*100}\n")

            # Setup optimizer for Stage 1 (only head parameters)
            self.setup_training_components(stage=1, learning_rate=self.learning_rate)

            for epoch in range(self.stage1_epochs):
                self.epoch_start_time = time.time()

                print(f'\n{"="*80}')
                print(f'STAGE 1 | Epoch {epoch+1}/{self.stage1_epochs} (Overall: {epoch+1}/{num_epochs})')
                print(f'{"="*80}')

                # Training phase
                train_loss, train_acc = self.train_epoch(train_loader, epoch+1, num_epochs)

                # Validation phase
                val_loss, val_acc = self.validate_epoch(val_loader, epoch+1, num_epochs)

                # Calculate epoch time
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update learning rate
                self.scheduler.step(val_loss)

                # Save best model and check early stopping
                improvement = ""
                if val_loss < self.best_val_loss - (self.config.early_stopping_min_delta if hasattr(self.config, 'early_stopping_min_delta') else 0.001):
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    best_epoch = epoch + 1
                    improvement = "  NEW BEST!"
                    self.early_stopping_counter = 0  # Reset counter on improvement
                else:
                    self.early_stopping_counter += 1

                # Check for early stopping
                if self.config.use_early_stopping and self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.early_stopping_triggered = True
                    print(f'\n{"[WARNING] "*40}')
                    print(f'EARLY STOPPING TRIGGERED at Epoch {epoch+1}')
                    print(f'Validation loss has not improved for {self.config.early_stopping_patience} consecutive epochs')
                    print(f'Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}')
                    print(f'{"[WARNING] "*40}\n')

                # Record history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['epoch_times'].append(epoch_time)
                self.training_history['learning_rates'].append(current_lr)
                self.training_history['training_stage'].append(1)

                # Calculate statistics
                avg_epoch_time = np.mean(self.epoch_times)
                eta = self._estimate_remaining_time(epoch + 1, num_epochs)
                total_elapsed = time.time() - self.total_start_time

                # Print epoch summary
                print(f'\n{"─"*80}')
                print(f' STAGE 1 EPOCH {epoch+1}/{self.stage1_epochs} SUMMARY{improvement}')
                print(f'{"─"*80}')
                print(f'Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
                print(f'Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}')
                print(f'{"─"*80}')
                print(f'  Epoch Time: {self._format_time(epoch_time)} | Avg: {self._format_time(avg_epoch_time)}')
                print(f' Learning Rate: {current_lr:.6f}')
                print(f'⏳ Elapsed: {self._format_time(total_elapsed)} | ETA: {eta}')
                print(f' Best Val Loss: {self.best_val_loss:.4f} (Epoch {best_epoch})')
                print(f'{"─"*80}')

                # Update live plot (Bokeh)
                print(f"\n[DEBUG] Checking live plot update: use_live_plotting={self.use_live_plotting}, live_plotter={self.live_plotter is not None}")
                if self.use_live_plotting and self.live_plotter:
                    try:
                        print(f"[DEBUG] Updating live plot for epoch {epoch + 1}...")
                        self.live_plotter.update(
                            epoch=epoch + 1,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            train_acc=train_acc,
                            val_acc=val_acc
                        )
                        print(f"[DEBUG] Live plot update successful!")
                    except Exception as e:
                        print(f"\n Warning: Live plot update failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[DEBUG] Live plotting is disabled or plotter not initialized")

                # Break if early stopping triggered
                if self.early_stopping_triggered:
                    break

        # ============================================================================
        # STAGE 2: Fine-tune entire network with reduced learning rate
        # ============================================================================
        if self.stage2_epochs > 0 and use_two_stage and not self.early_stopping_triggered:
            self.current_stage = 2
            print(f"\n{''*40}")
            print(f"{'  '*14}STAGE 2: FINE-TUNING")
            print(f"{''*40}")
            print(f"Unfreezing backbone layers for fine-tuning with reduced learning rate")
            print(f"Epochs: {self.stage2_epochs} | Learning Rate: {self.learning_rate * self.config.stage2_lr_multiplier:.6f}")
            print(f"{'─'*100}\n")

            # Reset early stopping counter for new stage
            self.early_stopping_counter = 0
            self.early_stopping_triggered = False

            # Unfreeze layers
            self.unfreeze_layers(self.model_name)

            # Setup new optimizer for Stage 2 with reduced learning rate
            stage2_lr = self.learning_rate * self.config.stage2_lr_multiplier
            self.setup_training_components(stage=2, learning_rate=stage2_lr)

            for epoch in range(self.stage2_epochs):
                self.epoch_start_time = time.time()

                overall_epoch = self.stage1_epochs + epoch + 1
                print(f'\n{"="*80}')
                print(f'STAGE 2 | Epoch {epoch+1}/{self.stage2_epochs} (Overall: {overall_epoch}/{num_epochs})')
                print(f'{"="*80}')

                # Training phase
                train_loss, train_acc = self.train_epoch(train_loader, overall_epoch, num_epochs)

                # Validation phase
                val_loss, val_acc = self.validate_epoch(val_loader, overall_epoch, num_epochs)

                # Calculate epoch time
                epoch_time = time.time() - self.epoch_start_time
                self.epoch_times.append(epoch_time)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Update learning rate
                self.scheduler.step(val_loss)

                # Save best model and check early stopping
                improvement = ""
                if val_loss < self.best_val_loss - (self.config.early_stopping_min_delta if hasattr(self.config, 'early_stopping_min_delta') else 0.001):
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    best_epoch = overall_epoch
                    improvement = "  NEW BEST!"
                    self.early_stopping_counter = 0  # Reset counter on improvement
                else:
                    self.early_stopping_counter += 1

                # Check for early stopping
                if self.config.use_early_stopping and self.early_stopping_counter >= self.config.early_stopping_patience:
                    self.early_stopping_triggered = True
                    print(f'\n{"[WARNING] "*40}')
                    print(f'EARLY STOPPING TRIGGERED at Epoch {overall_epoch}')
                    print(f'Validation loss has not improved for {self.config.early_stopping_patience} consecutive epochs')
                    print(f'Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch}')
                    print(f'{"[WARNING] "*40}\n')

                # Record history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)
                self.training_history['epoch_times'].append(epoch_time)
                self.training_history['learning_rates'].append(current_lr)
                self.training_history['training_stage'].append(2)

                # Calculate statistics
                avg_epoch_time = np.mean(self.epoch_times)
                eta = self._estimate_remaining_time(overall_epoch, num_epochs)
                total_elapsed = time.time() - self.total_start_time

                # Print epoch summary
                print(f'\n{"─"*80}')
                print(f' STAGE 2 EPOCH {epoch+1}/{self.stage2_epochs} SUMMARY{improvement}')
                print(f'{"─"*80}')
                print(f'Training   → Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
                print(f'Validation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}')
                print(f'{"─"*80}')
                print(f'  Epoch Time: {self._format_time(epoch_time)} | Avg: {self._format_time(avg_epoch_time)}')
                print(f' Learning Rate: {current_lr:.6f}')
                print(f'⏳ Elapsed: {self._format_time(total_elapsed)} | ETA: {eta}')
                print(f' Best Val Loss: {self.best_val_loss:.4f} (Epoch {best_epoch})')
                print(f'{"─"*80}')

                # Update live plot (Bokeh)
                print(f"\n[DEBUG] Checking live plot update: use_live_plotting={self.use_live_plotting}, live_plotter={self.live_plotter is not None}")
                if self.use_live_plotting and self.live_plotter:
                    try:
                        print(f"[DEBUG] Updating live plot for epoch {overall_epoch}...")
                        self.live_plotter.update(
                            epoch=overall_epoch,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            train_acc=train_acc,
                            val_acc=val_acc
                        )
                        print(f"[DEBUG] Live plot update successful!")
                    except Exception as e:
                        print(f"\n Warning: Live plot update failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[DEBUG] Live plotting is disabled or plotter not initialized")

                # Break if early stopping triggered
                if self.early_stopping_triggered:
                    break

        # ============================================================================
        # Training Complete
        # ============================================================================
        total_time = time.time() - self.total_start_time
        
        # Derive final metrics safely
        final_train_acc = self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0.0
        final_val_acc = self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0.0
        avg_epoch_time = float(np.mean(self.epoch_times)) if self.epoch_times else 0.0
        best_val_loss = float(self.best_val_loss) if self.best_val_loss != float('inf') else 0.0
        print(f'\n{"="*100}')
        print(f'[OK] TRAINING COMPLETED!')
        print(f'{"="*100}')
        if use_two_stage:
            print(f' Stage 1 Epochs: {self.stage1_epochs}')
            print(f' Stage 2 Epochs: {self.stage2_epochs}')
        if self.early_stopping_triggered:
            print(f'[WARNING]  Early Stopping: TRIGGERED (stopped early to prevent overfitting)')
        print(f'  Total Time: {self._format_time(total_time)}')
        print(f' Average Epoch Time: {self._format_time(avg_epoch_time)}')
        print(f' Best Validation Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
        print(f' Final Training Accuracy: {final_train_acc:.4f}')
        print(f' Final Validation Accuracy: {final_val_acc:.4f}')
        print(f'{"="*100}\n')

        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"[OK] Loaded best model from epoch {best_epoch}")

        # Display live plot info
        if self.use_live_plotting and self.live_plotter:
            print("\n" + "="*100)
            print(" LIVE TRAINING VISUALIZATION")
            print("="*100)
            print(f"Interactive training plots available in your browser!")
            print(f"Plot file: {self.live_plotter.html_file}")
            print(f"Browser should have opened automatically. If not, open the file manually.")
            print("="*100 + "\n")

    def train_model(self, train_loader, val_loader, num_epochs=None):
        """Alias for train() method for compatibility with main.py"""
        return self.train(train_loader, val_loader, num_epochs)

    def save_training_history(self, filepath):
        """Save training history to file"""
        import numpy as np
        np.save(filepath, self.training_history)
        print(f"Training history saved to: {filepath}")

    def plot_training_metrics(self, save_path=None):
        """Alias for plot_training_history for compatibility with main.py"""
        return self.plot_training_history(save_path)

    def plot_training_history(self, save_path=None):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot loss
        ax1.plot(self.training_history['train_loss'], label='Training Loss')
        ax1.plot(self.training_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot accuracy
        ax2.plot(self.training_history['train_acc'], label='Training Accuracy')
        ax2.plot(self.training_history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training history plot saved to: {save_path}")

        plt.show()

    def save_checkpoint(self, filepath, epoch, optimizer_state=True):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_model_state_dict': self.best_model_state,
            'optimizer_state_dict': self.optimizer.state_dict() if optimizer_state else None,
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to: {filepath}")

    def save_complete_model(self, save_dir, model_name=None):
        """
        Save complete model package for future predictions
        Includes: model weights, architecture info, training config, and metadata
        """
        from pathlib import Path
        import json
        from datetime import datetime

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            model_name = self.model.model_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model-specific directory
        model_dir = save_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f" SAVING COMPLETE MODEL PACKAGE: {model_name}")
        print(f"{'='*80}")

        # 1. Save model state (weights + architecture)
        model_weights_path = model_dir / f"{model_name}_model.pth"
        torch.save({
            'model_state_dict': self.best_model_state if self.best_model_state else self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'model_name': model_name,
            'num_classes': self.config.num_classes,
            'input_size': getattr(self.config, 'input_size', (224, 224)),
        }, model_weights_path)
        print(f"[OK] Model weights saved: {model_weights_path.name}")

        # 2. Save training history
        history_path = model_dir / f"{model_name}_training_history.npy"
        np.save(history_path, self.training_history)
        print(f"[OK] Training history saved: {history_path.name}")

        # 3. Save optimizer state (for potential resume)
        if self.optimizer:
            optimizer_path = model_dir / f"{model_name}_optimizer.pth"
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, optimizer_path)
            print(f"[OK] Optimizer state saved: {optimizer_path.name}")

        # 4. Save configuration and metadata
        metadata = {
            'model_name': model_name,
            'model_class': self.model.__class__.__name__,
            'num_classes': self.config.num_classes,
            'training_date': timestamp,
            'total_epochs_trained': len(self.training_history['train_loss']),
            'best_val_loss': float(self.best_val_loss),
            'best_val_acc': float(max(self.training_history['val_acc'])) if self.training_history['val_acc'] else 0.0,
            'final_train_loss': float(self.training_history['train_loss'][-1]) if self.training_history['train_loss'] else 0.0,
            'final_train_acc': float(self.training_history['train_acc'][-1]) if self.training_history['train_acc'] else 0.0,
            'final_val_loss': float(self.training_history['val_loss'][-1]) if self.training_history['val_loss'] else 0.0,
            'final_val_acc': float(self.training_history['val_acc'][-1]) if self.training_history['val_acc'] else 0.0,
            'batch_size': self.config.get_model_config(self.model_name).get('batch_size', 32) if self.model_name else 32,
            'learning_rate': self.learning_rate,
            'device': str(self.device),
            'total_training_time': sum(self.epoch_times) if self.epoch_times else 0.0,
            'average_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0.0,
        }

        metadata_path = model_dir / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"[OK] Metadata saved: {metadata_path.name}")

        # 5. Save training plots
        plot_path = model_dir / f"{model_name}_training_plots.png"
        self.plot_training_history(save_path=plot_path)
        print(f"[OK] Training plots saved: {plot_path.name}")

        # 6. Create README with instructions
        readme_content = f"""# {model_name} - Trained Model Package

## Model Information
- **Model Architecture**: {self.model.__class__.__name__}
- **Training Date**: {timestamp}
- **Number of Classes**: {self.config.num_classes}
- **Best Validation Loss**: {self.best_val_loss:.4f}
- **Best Validation Accuracy**: {max(self.training_history['val_acc']) if self.training_history['val_acc'] else 0.0:.4f}

## Training Summary
- **Total Epochs**: {len(self.training_history['train_loss'])}
- **Batch Size**: {self.config.get_model_config(self.model_name).get('batch_size', 32) if self.model_name else 32}
- **Learning Rate**: {self.learning_rate}
- **Total Training Time**: {self._format_time(sum(self.epoch_times)) if self.epoch_times else '0s'}
- **Device**: {self.device}

## Files in this Package
1. `{model_name}_model.pth` - Model weights and architecture info
2. `{model_name}_training_history.npy` - Complete training history
3. `{model_name}_optimizer.pth` - Optimizer state (for resume training)
4. `{model_name}_metadata.json` - Training metadata and metrics
5. `{model_name}_training_plots.png` - Training/validation curves
6. `README.md` - This file

## How to Load and Use for Predictions

```python
import torch
from modules import ModelFactory, Config

# Load configuration
config = Config()

# Create model instance
model = ModelFactory.create_model('{model_name.lower()}', config)

# Load trained weights
checkpoint = torch.load('{model_name}_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)
```

## Performance Metrics
- Final Training Loss: {self.training_history['train_loss'][-1] if self.training_history['train_loss'] else 0.0:.4f}
- Final Training Accuracy: {self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0.0:.4f}
- Final Validation Loss: {self.training_history['val_loss'][-1] if self.training_history['val_loss'] else 0.0:.4f}
- Final Validation Accuracy: {self.training_history['val_acc'][-1] if self.training_history['val_acc'] else 0.0:.4f}
"""

        readme_path = model_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"[OK] README created: {readme_path.name}")

        print(f"\n{'='*80}")
        print(f"[OK] Complete model package saved to: {model_dir}")
        print(f"{'='*80}\n")

        return model_dir

    def load_checkpoint(self, filepath):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_model_state = checkpoint['best_model_state_dict']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']

        if checkpoint['optimizer_state_dict'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from: {filepath}")
        return checkpoint['epoch']

# Test function for independent execution
def test_train_module():
    """Test the training module independently"""
    print("Testing Train module...")

    try:
        # Import required modules
        from config import Config
        from models import get_model

        # Create config
        config = Config()
        print("[OK] Config imported and created successfully")

        # Test trainer creation for each model
        model_names = ['vgg16', 'resnet50', 'inceptionv3']

        for model_name in model_names:
            try:
                print(f"Testing Trainer with {model_name.upper()} model...")

                # Create model
                model = get_model(model_name, config)
                print(f"[OK] {model_name.upper()} model created")

                # Create trainer
                trainer = Trainer(model, config)
                print(f"[OK] Trainer created for {model_name.upper()}")

                # Test training components setup
                trainer.setup_training_components()
                print(f"[OK] Training components setup successful for {model_name.upper()}")

                # Test that trainer has all required attributes
                assert hasattr(trainer, 'optimizer'), "Trainer missing optimizer"
                assert hasattr(trainer, 'criterion'), "Trainer missing criterion"
                assert hasattr(trainer, 'scheduler'), "Trainer missing scheduler"
                assert hasattr(trainer, 'training_history'), "Trainer missing training_history"

                print(f"[OK] All trainer attributes present for {model_name.upper()}")

                # Test dummy training step (without actual data)
                print(f"[OK] {model_name.upper()} trainer test completed")

            except Exception as e:
                print(f"[FAIL] {model_name.upper()} trainer test failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        print("[OK] All trainer tests successful")
        print("[OK] Train module test PASSED")
        return True

    except Exception as e:
        print(f"[FAIL] Train module test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


class TrainingMonitor:
    """Monitor training progress and provide callbacks"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if training should stop early"""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop

    def reset(self):
        """Reset the monitor"""
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
