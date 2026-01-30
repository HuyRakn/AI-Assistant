import mlx.core as mx
from typing import Dict, List, Tuple

class SovereignAdamW:
    """
    Sovereign Implementation of AdamW (Adam with Decoupled Weight Decay).
    
    Why Manual?
    Many libraries mix Weight Decay with L2 Regularization (adding to Loss).
    For Adaptive Optimizers (Adam), this is mathematically incorrect because
    the decay term gets scaled by the adaptive factor (1/sqrt(v)).
    
    Correct Formula (arXiv:1711.05101):
    1. Decay: w_t = w_{t-1} - eta * lambda * w_{t-1}
    2. Step:  w_t = w_t - eta * (m_t / (sqrt(v_t) + eps))
    """
    def __init__(self, learning_rate: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), 
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.lr = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # State: Map parameter ID -> (m, v)
        # Assuming parameters are passed as a tree, we need to maintain a congruent state tree.
        self.state = {}
        self.step = 0

    def init_single(self, key, param):
        """Initialize state for a single parameter tensor."""
        if key not in self.state:
            self.state[key] = {
                "m": mx.zeros_like(param),
                "v": mx.zeros_like(param)
            }

    def update(self, model, gradients):
        """
        Apply gradients to model parameters.
        model: Dictionary/PyTree of parameters (can be obtained via model.parameters())
        gradients: Dictionary/PyTree of gradients (same structure)
        
        Returns: Updated parameters tree.
        """
        self.step += 1
        lr = self.lr # Can add schedule here later
        
        # Correct Bias Correction
        bias_correction1 = 1 - self.beta1 ** self.step
        bias_correction2 = 1 - self.beta2 ** self.step
        
        def apply_adamw(key, param, grad):
            # Ensure state exists
            if key not in self.state:
                 self.state[key] = {
                    "m": mx.zeros_like(param),
                    "v": mx.zeros_like(param)
                }
            
            state = self.state[key]
            m = state["m"]
            v = state["v"]
            
            # 1. Update Moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            
            # Save state
            self.state[key]["m"] = m
            self.state[key]["v"] = v
            
            # 2. Bias Correction
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2
            
            # 3. Decoupled Weight Decay
            # w = w - lr * lambda * w
            param = param - lr * self.weight_decay * param
            
            # 4. Adaptive Step
            # w = w - lr * m_hat / (sqrt(v_hat) + eps)
            update = m_hat / (mx.sqrt(v_hat) + self.eps)
            param = param - lr * update
            
            return param

        # Recursively apply to the tree
        def walk_apply(params, grads, prefix=""):
            new_params = {}
            for (p_key, p), (g_key, g) in zip(params.items(), grads.items()):
                # Unpack tuple from zip. Note: params.items() yields (key, value).
                
                # Check consistency
                if p_key != g_key:
                    raise ValueError(f"Parameter/Gradient mismatch: {p_key} vs {g_key}")
                
                curr_key = f"{prefix}.{p_key}" if prefix else p_key
                
                if isinstance(p, dict):
                    new_params[p_key] = walk_apply(p, g, curr_key)
                elif isinstance(p, list):
                    # List handling
                    new_list = []
                    for i, (p_item, g_item) in enumerate(zip(p, g)):
                        list_key = f"{curr_key}.{i}"
                        if isinstance(p_item, dict):
                             new_list.append(walk_apply(p_item, g_item, list_key))
                        elif isinstance(p_item, mx.array):
                             new_list.append(apply_adamw(list_key, p_item, g_item))
                    new_params[p_key] = new_list
                elif isinstance(p, mx.array):
                    new_params[p_key] = apply_adamw(curr_key, p, g)
                    
            return new_params

        # In MLX, model.parameters() returns a dict-like tree. 
        # But to update, we usually use `model.update(new_params)`.
        # However, `value_and_grad` returns grads matching `model.parameters()`.
        # So we can iterate them.
        
        # Note: calling model.parameters() creates a COPY of structure references?
        # Actually `model.parameters()` returns a dict.
        # We need to construct the new dictionary of parameters.
        
        # We assume `model` passed here is the `parameters()` dict, OR the `Module`.
        # Standard MLX optimizer takes `model` (Module) or `params`.
        # Let's assume we pass `model.parameters()` and `gradients`.
        # But `trainer` usually updates the model in place or returns new params.
        
        # Let's standardize: `update(params_tree, grads_tree)` -> `new_params_tree`.
        # The Trainer will then call `model.update(new_params_tree)`.
        
        # Wait, if we want `optimizer.update(model, grads)` where model is Module...
        # Let's accept `params` (The Dict) as first arg, not Module.
        # The Trainer handles the Module update.
        
        return walk_apply(model, gradients)

