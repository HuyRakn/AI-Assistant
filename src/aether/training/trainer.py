import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Callable, Iterable, Optional, Tuple
from ..model.transformer import AetherForCausalLM
from .manual import ManualCrossEntropy
from .optimizer import SovereignAdamW
from ..model.utils import CheckpointManager

class SovereignTrainer:
    """
    The Perpetual Engine (Phase 3.4).
    Full White-Box Training Loop with:
    - Manual Gradient Accumulation (for huge batch sizes on limited RAM).
    - Manual Gradient Clipping (to prevent explosion).
    - Sovereign Optimizer Integration.
    """
    def __init__(self, model: AetherForCausalLM, optimizer: SovereignAdamW, 
                 grad_accumulation_steps: int = 1, max_grad_norm: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # White-Box Metric
        self.loss_fn_metric = ManualCrossEntropy()
        
        # Compile the gradient function
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

    def compute_grad(self, inputs, targets):
        """
        Compute gradients for a micro-batch.
        """
        loss, grads = self.loss_and_grad_fn(self.model, inputs, targets)
        return loss, grads

    def clip_grad_norm(self, grads, max_norm):
        """
        Manual Gradient Clipping.
        TotalNorm = sqrt( sum( ||g||^2 ) )
        Scale = min(1.0, max_norm / (TotalNorm + eps))
        """
        # Flatten all leaf gradients to compute global L2 norm
        # We need to traverse the tree structure
        total_norm = 0.0
        
        def reduce_norm(g):
            nonlocal total_norm
            if isinstance(g, mx.array):
                total_norm += mx.sum(g * g).item()
            elif isinstance(g, list):
                for item in g: reduce_norm(item)
            elif isinstance(g, dict):
                for item in g.values(): reduce_norm(item)

        reduce_norm(grads)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            # Scale down all gradients
            # Using tree_map utility from MLX would be nice, but let's do manual recursive walk
            def scale_grads(g):
                if isinstance(g, mx.array):
                    return g * clip_coef
                elif isinstance(g, list):
                    return [scale_grads(item) for item in g]
                elif isinstance(g, dict):
                    return {k: scale_grads(v) for k, v in g.items()}
                return g
                
            return scale_grads(grads), total_norm
            
        return grads, total_norm

    def evaluate(self, dataset: Iterable, max_steps: Optional[int] = None) -> Tuple[float, float]:
        """
        The Sovereign Judge.
        evaluates the model on a held-out dataset.
        Returns: (Average Loss, Perplexity)
        """
        print("🧪 Starting Evaluation...")
        total_loss = 0.0
        total_steps = 0
        
        # We don't need gradients
        # MLX doesn't have a global "no_grad" mode like PyTorch?
        # Actually `mx.eval` just runs computation. 
        # Gradients are only computed if we use `nn.value_and_grad`.
        # So standard forward pass is implicitly no-grad.
        
        for i, batch in enumerate(dataset):
            if max_steps and i >= max_steps:
                break
                
            inputs = mx.array(batch['input_ids'])
            if inputs.shape[1] > 1:
                targets = inputs[:, 1:]
                inputs = inputs[:, :-1]
            else:
                continue
            
            # Forward Pass Only
            loss = self.loss_fn(self.model, inputs, targets)
            total_loss += loss.item()
            total_steps += 1
            
        if total_steps == 0:
            return 0.0, 0.0
            
        avg_loss = total_loss / total_steps
        perplexity = np.exp(avg_loss)
        
        return avg_loss, perplexity

    def train(self, dataset: Iterable, eval_dataset: Optional[Iterable] = None, 
              steps: int = 1000, report_interval: int = 10, save_interval: int = 100, 
              eval_interval: int = 50, limit_val_batches: int = 50):
        print(f"🔥 Ignition: Starting Sovereign Training Loop...")
        print(f"   - Steps: {steps}")
        print(f"   - Accumulation: {self.grad_accumulation_steps}")
        print(f"   - Clip Norm: {self.max_grad_norm}")
        if eval_dataset:
            print(f"   - Evaluation active (Interval: {eval_interval})")
        
        start_time = time.time()
        tokens_processed = 0
        current_step = 0
        
        # Accumulator for gradients
        # We need a zero-tree matching structure.
        # Simplest way: Initialize with first batch's grads.
        accumulated_grads = None
        accumulated_loss = 0.0
        micro_step = 0
        
        # Optimization: Compilation
        # We compile the compute_grad function.
        # Note: Gradients accumulation logic (add tree) handles best outside compile 
        # unless we write a full accumulated step function.
        # For readability and "White-Box" explicit control, we compile the heavy compute only.
        
        # @mx.compile
        def compiled_compute_grad(inputs, targets):
            return self.compute_grad(inputs, targets)

        for batch in dataset:
            if current_step >= steps: break
            
            # --- Data Pipeline ---
            inputs = mx.array(batch['input_ids'])
            if inputs.shape[1] > 1:
                targets = inputs[:, 1:]
                inputs = inputs[:, :-1]
            else:
                continue
                
            # --- Forward & Backward ---
            # Run micro-step
            loss, grads = self.compute_grad(inputs, targets)
            
            # Ensure evaluation to get value
            # mx.eval(loss) # Deferred until accumulation?
            
            # --- Gradient Accumulation ---
            accumulated_loss += loss.item()
            
            # Helper to add trees
            def add_trees(t1, t2):
                if t1 is None: return t2
                if isinstance(t1, mx.array):
                    return t1 + t2
                elif isinstance(t1, dict):
                    return {k: add_trees(t1[k], t2[k]) for k in t1}
                elif isinstance(t1, list):
                    return [add_trees(v1, v2) for v1, v2 in zip(t1, t2)]
                return t1

            accumulated_grads = add_trees(accumulated_grads, grads)
            micro_step += 1
            
            # --- Optimizer Step (Once per Logical Batch) ---
            if micro_step >= self.grad_accumulation_steps:
                # 1. Average Gradients
                def div_tree(tree, div):
                    if isinstance(tree, mx.array): return tree / div
                    if isinstance(tree, dict): return {k: div_tree(v, div) for k, v in tree.items()}
                    if isinstance(tree, list): return [div_tree(v, div) for v in tree]
                    return tree
                    
                final_grads = div_tree(accumulated_grads, self.grad_accumulation_steps)
                avg_loss = accumulated_loss / self.grad_accumulation_steps
                
                # 2. Gradient Clipping
                final_grads, grad_norm = self.clip_grad_norm(final_grads, self.max_grad_norm)
                
                # 3. Update Weights (Sovereign Optimization)
                # optimizer.update returns new params tree (if stateless model update)
                # But here passing (model.parameters(), grads) -> updates internal model?
                # Our SovereignAdamW returns new_params_tree.
                # Use model.update(new_params_tree).
                
                # Need model.parameters() as dict
                new_params = self.optimizer.update(self.model.parameters(), final_grads)
                self.model.update(new_params)
                
                # Reset Accumulator
                accumulated_grads = None
                accumulated_loss = 0.0
                micro_step = 0
                current_step += 1
                
                # Ensure computation happens on the graph update
                mx.eval(self.model.parameters())
                
                tokens_processed += inputs.size * self.grad_accumulation_steps
                
                # Reporting
                if current_step % report_interval == 0:
                    dt = time.time() - start_time
                    tps = tokens_processed / dt
                    print(f"Step {current_step}: Loss = {avg_loss:.4f} | Norm = {grad_norm:.4f} | TPS: {tps:.2f}")
                
                # Evaluation
                if eval_dataset and current_step % eval_interval == 0:
                    val_loss, val_ppl = self.evaluate(eval_dataset, max_steps=limit_val_batches)
                    print(f"📊 Eval Result: Val Loss = {val_loss:.4f} | PPL = {val_ppl:.2f}")

                # Checkpointing
                if current_step % save_interval == 0 and current_step > 0:
                    self.checkpoint_manager.save(self.model, current_step)
            
        print("✅ Training Sequence Complete.")
