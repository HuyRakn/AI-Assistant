import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Callable, Iterable
from .transformer import AetherForCausalLM
from .manual import ManualCrossEntropy
from .utils import CheckpointManager

class AetherTrainer:
    """
    The Crucible: Where the model is forged (White-Box Edition).
    Handles the training loop using manual math.
    """
    def __init__(self, model: AetherForCausalLM, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn_metric = ManualCrossEntropy() # White-Box Metric
        self.loss_and_grad_fn = nn.value_and_grad(self.model, self.loss_fn)
        self.checkpoint_manager = CheckpointManager()

    def loss_fn(self, model, inputs, targets, mask=None):
        # inputs: [B, L]
        # targets: [B, L]
        logits, _ = model(inputs, mask=mask)
        # logits: [B, L, V]
        
        # Reshape for computation
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        
        # White-Box Manual Cross Entropy
        loss = self.loss_fn_metric(logits, targets)
        return loss

    def train_step(self, start_tokens: mx.array, target_tokens: mx.array):
        loss, grads = self.loss_and_grad_fn(self.model, start_tokens, target_tokens)
        self.optimizer.update(self.model, grads)
        return loss

    def train(self, dataset: Iterable, steps: int = 1000, report_interval: int = 10, save_interval: int = 100):
        print(f"🔥 Ignition: Starting Training for {steps} steps...")
        
        start_time = time.time()
        tokens_processed = 0
        
        # Compile step for speed
        state = [self.model.state, self.optimizer.state, mx.random.state]
        
        @mx.compile
        def step_fn(inputs, targets):
            return self.train_step(inputs, targets)

        step_idx = 0
        for batch in dataset:
            if step_idx >= steps: break
            
            # Unpack batch (assuming simple ingestion pipeline output)
            # input_ids: [B, L]
            inputs = mx.array(batch['input_ids'])
            # Shift for Causal LM: input [:-1], target [1:] is usually handled by data loader
            # Assuming dataset checks out.
            
            # Simple shifting if dataset is raw sequence
            if inputs.shape[1] > 1:
                targets = inputs[:, 1:]
                inputs = inputs[:, :-1]
            else:
                continue
                
            loss = step_fn(inputs, targets)
            mx.eval(state) # Ensure computation
            
            tokens_processed += inputs.size
            
            if step_idx % report_interval == 0:
                dt = time.time() - start_time
                tps = tokens_processed / dt
                print(f"Step {step_idx}: Loss = {loss.item():.4f} | TPS: {tps:.2f} | Time: {dt:.2f}s")
                
            if step_idx % save_interval == 0 and step_idx > 0:
                self.checkpoint_manager.save(self.model, step_idx)
                
            step_idx += 1
            
        print("✅ Training Sequence Complete.")
