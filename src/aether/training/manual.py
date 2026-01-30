import mlx.core as mx
import mlx.nn as nn
from typing import List, Optional

class ManualCrossEntropy:
    """
    White-Box Cross Entropy Loss.
    Formula: - sum(target * log(softmax(logits)))
    """
    def __call__(self, logits: mx.array, targets: mx.array) -> mx.array:
        # logits: [N, V]
        # targets: [N] (Indices)
        
        # 1. Log Softmax (Numerical Stability Trick: x - max(x) - log(sum(exp(...))))
        # We can implement it physically or use primitives. 
        # User asked for "mathematical formula code".
        
        # log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # But for raw white-box, maybe we assume `nn.log_softmax` is acceptable?
        # Let's compute it strictly manually to prove sovereignty.
        
        # Max for numerical stability
        max_logits = mx.max(logits, axis=-1, keepdims=True)
        stable_logits = logits - max_logits
        
        # Softmax Denominator
        exp_logits = mx.exp(stable_logits)
        sum_exp = mx.sum(exp_logits, axis=-1, keepdims=True)
        log_sum_exp = mx.log(sum_exp)
        
        # Log Softmax
        log_probs = stable_logits - log_sum_exp
        
        # 2. NLL Loss (Negative Log Likelihood)
        # Select log_probs corresponding to target indices
        # Gather logic: [range(N), targets]
        
        # Create mask or use gather. MLX gather/take:
        n_samples = logits.shape[0]
        indices = mx.arange(n_samples)
        
        # target_log_probs: [N]
        # We manually index: logits[i, targets[i]]
        # Using mlx.take? Or simple array indexing?
        # Aether White-Box Style: 
        # Since MLX is array based, we use intelligent indexing.
        target_log_probs = log_probs[indices, targets]
        
        # 3. Mean Loss
        loss = -mx.mean(target_log_probs)
        
        return loss

class ManualAdamW:
    """
    White-Box AdamW Optimizer.
    The Engine of Learning.
    
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    w_t = w_{t-1} - lr * (m_t / (sqrt(v_t) + eps) + lambda * w_{t-1})
    """
    def __init__(self, learning_rate: float = 1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State: {param_id: {'m': ..., 'v': ...}}
        self.state = {}
        self.step = 0
        
    def update(self, model: nn.Module, grads: dict):
        self.step += 1
        
        # Iterate over all parameters
        # In MLX, model.parameters() returns the tree of weights.
        current_params = model.parameters()
        
        # Calculate updates recursively
        new_params = self.apply_gradients(current_params, grads, prefix="root")
        
        # Apply updates to the model
        model.update(new_params)
    
    def apply_gradients(self, params: dict, grads: dict, prefix: str):
        """
        Recursive update function for MLX parameter tree.
        """
        updates = {}
        for k, p in params.items():
            if k not in grads:
                continue
            
            # Construct unique key for state tracking
            current_key = f"{prefix}.{k}"
            
            if isinstance(p, dict):
                # Recurse
                updates[k] = self.apply_gradients(p, grads[k], prefix=current_key)
            else:
                g = grads[k]
                
                # --- STATEFUL OPTIMIZER LOGIC ---
                # Retrieve previous state or initialize
                if current_key not in self.state:
                    self.state[current_key] = {
                        'm': mx.zeros_like(p),
                        'v': mx.zeros_like(p)
                    }
                
                state = self.state[current_key]
                m_prev = state['m']
                v_prev = state['v']
                
                # 1. Update Moments (Exponential Moving Average)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m = self.beta1 * m_prev + (1 - self.beta1) * g
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v = self.beta2 * v_prev + (1 - self.beta2) * (g * g)
                
                # Update State
                self.state[current_key]['m'] = m
                self.state[current_key]['v'] = v
                
                # 2. Bias correction
                # Note: self.step starts at 1
                bias_correction1 = 1 - self.beta1 ** self.step
                bias_correction2 = 1 - self.beta2 ** self.step
                
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2
                
                # 3. Update Weights
                # w_t = w_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + lambda * w_{t-1})
                # Apply Weight Decay (Decoupled AdamW style)
                p_with_decay = p - (self.lr * self.weight_decay * p)
                
                # Apply Gradients
                p_new = p_with_decay - self.lr * (m_hat / (mx.sqrt(v_hat) + self.eps))
                
                updates[k] = p_new
                
        return updates
