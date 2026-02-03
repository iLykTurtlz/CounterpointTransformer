import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from .dataset import PolyphonyTorchDataset
from .neural_model import PolyphonyTransformer
import multiprocessing
import gc
from tqdm.auto import tqdm
import json
import os
import time


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
    }

class Trainer:
    """
    Trainer for the Renaissance polyphony transformer.
    Supports indefinite/resume-anytime training using WarmRestarts.
    """

    def __init__(
        self,
        model: PolyphonyTransformer,
        train_dataset: PolyphonyTorchDataset,
        val_dataset: PolyphonyTorchDataset | None = None,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "./content/drive/MyDrive/checkpoints",
        log_file: str = None,
        log_interval: int = 100,
        eval_interval: int = 1000,
        device: str | None = None,
        # Scheduler settings
        restart_period_epochs: int = 10,  # Restart LR every 10 epochs
        num_workers: int = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval

        # Internal state tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Logging
        if log_file is None:
            self.log_file = os.path.join(checkpoint_dir, 'training_log.json')
        else:
            self.log_file = log_file

        self.training_history = {
            'config': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'scheduler': 'CosineAnnealingWarmRestarts',
                'restart_period_epochs': restart_period_epochs
            },
            'epochs': [],
            'steps': [],
        }

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Setup loaders
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
            )
        else:
            self.val_loader = None

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # --- SCHEDULER CHANGE ---
        # CosineAnnealingWarmRestarts does not care about total epochs.
        # It only cares about T_0 (how many steps until the first restart).
        steps_per_epoch = len(self.train_loader) // gradient_accumulation_steps

        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=steps_per_epoch * restart_period_epochs, # Restart every X epochs
            T_mult=1, # Period remains constant (or set to 2 to double period every restart)
            eta_min=learning_rate * 0.1 # Don't let LR go below 10% of base
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_id)
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _save_logs(self):
        """Save training history to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    def train_step(self, batch) -> float:
        """Perform a single training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        # Shift for next-token prediction
        # Input: all tokens except last
        # Target: all tokens except first
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        mask = attention_mask[:, :-1]

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
          # Forward pass
          logits = self.model(inputs, attention_mask=mask)

          # Compute loss
          loss = self.criterion(
              logits.reshape(-1, logits.size(-1)),
              targets.reshape(-1)
          )

          # Scale loss for gradient accumulation
          loss = loss / self.gradient_accumulation_steps

        # Backward pass (no scaler needed for bfloat16 usually, but safe to do standard)
        loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> float:
        """Evaluate on validation set."""
        if self.val_loader is None:
            return 0.0

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            inputs = input_ids[:, :-1]
            targets = input_ids[:, 1:]
            mask = attention_mask[:, :-1]

            logits = self.model(inputs, attention_mask=mask)
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0


    def train(self, epochs: int):
        """
        Train for a specific number of *additional* epochs.
        Call this repeatedly to extend training.
        """
        self.model.train()

        start_epoch = self.current_epoch
        target_epoch = start_epoch + epochs

        print(f"Resuming from epoch {start_epoch + 1}")
        print(f"Training until epoch {target_epoch}")
        print(f"Logging to: {self.log_file}")

        running_loss = 0.0

        for epoch in range(start_epoch, target_epoch):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{target_epoch}",
                leave=True
            )

            for batch_idx, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                running_loss += loss
                epoch_loss += loss
                num_batches += 1

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:



                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step() # Step scheduler every batch
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.log_interval == 0:
                        avg_loss = running_loss / self.log_interval
                        lr = self.scheduler.get_last_lr()[0]
                        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
                        self.training_history['steps'].append({
                            'step': self.global_step, 'train_loss': avg_loss,
                            'lr': lr, 'epoch': epoch + 1
                        })
                        running_loss = 0.0

                    # Mid-epoch validation
                    if self.val_loader and self.global_step % self.eval_interval == 0:
                        self._run_val(epoch)

            # End of epoch processing
            self.current_epoch += 1 # Update internal counter

            self.optimizer.zero_grad()

            epoch_duration = time.time() - epoch_start_time
            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
            val_loss = self.evaluate() if self.val_loader else None

            # Log epoch
            self.training_history['epochs'].append({
                'epoch': self.current_epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'duration': epoch_duration,
                'lr': self.scheduler.get_last_lr()[0]
            })

            print(f"Epoch {self.current_epoch} done. Train: {avg_train_loss:.4f}, Val: {val_loss if val_loss else 'N/A'}")

            # Save checkpoint
            self.save_checkpoint(os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pt'))
            self._save_logs()

            self._cleanup_memory()


    def _run_val(self, epoch):
        val_loss = self.evaluate()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'), is_best=True)
            print(f" ★ New best val loss: {val_loss:.4f}")

    # ... Include your other helper methods (train_step, evaluate, etc.) here ...

    def save_checkpoint(self, path: str, is_best: bool = False):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': {
              'vocab_size': self.model.output_proj.out_features,
              'd_model': self.model.d_model,
              'n_heads': self.model.transformer.layers[0].self_attn.num_heads,
              'n_layers': len(self.model.transformer.layers),
              'd_ff': self.model.transformer.layers[0].linear1.out_features,
              'max_seq_len': self.model.pos_encoding.pe.size(1),
              'pad_token_id': self.model.pad_token_id,
            }
        }
        torch.save(checkpoint, path)
        if is_best:
            torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_model.pt'))

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint.get('current_epoch', 0) # Load current epoch
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))


    def _cleanup_memory(self):
        """Explicitly clean up GPU memory"""
        # Clear optimizer state that might be holding references
        self.optimizer.zero_grad(set_to_none=True)

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Force garbage collection
        gc.collect()
