import mlx.core as mx
import mlx.nn as nn
import math

class SovereignInit:
    """
    The Foundation Layer (Phase 3.2).
    Manual initialization schemes to prevent 'Dead Neurons' or 'Exploding Gradients'.
    Implements 'The Mathematics of Variance'.
    """
    
    @staticmethod
    def xavier_uniform(tensor: mx.array, fan_in: int, fan_out: int, gain: float = 1.0) -> mx.array:
        """
        Xavier/Glorot Uniform.
        Target: Linear layers in Attention (Q, K, V).
        Limit = gain * sqrt(6 / (fan_in + fan_out))
        """
        limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return mx.random.uniform(low=-limit, high=limit, shape=tensor.shape)

    @staticmethod
    def kaiming_normal(tensor: mx.array, fan_in: int, gain: float = 1.0) -> mx.array:
        """
        He/Kaiming Normal. 
        Target: SwiGLU Gate/Up projections.
        Std = gain / sqrt(fan_in)
        """
        std = gain / math.sqrt(fan_in)
        return mx.random.normal(shape=tensor.shape, scale=std)

    @staticmethod
    def gpt2_residual_scale(tensor: mx.array, n_layers: int) -> mx.array:
        """
        GPT-2 Residual Projection Scaling.
        Target: o_proj (Attention) and down_proj (MLP).
        Scale = 1 / sqrt(2 * n_layers)
        
        Rationale: As depth increases, the variance of the residual path sums up.
        To keep variance ~1 at the end, we dampen the blocks' contributions.
        """
        if n_layers is None or n_layers < 1:
            return tensor
            
        scale_factor = 1.0 / math.sqrt(2 * n_layers)
        # We assume the tensor coming in is already initialized (e.g. Xavier).
        # We just multiply it.
        return tensor * scale_factor

    @staticmethod
    def init_model(model, n_layers: int = 12):
        """
        Walks the model tree and re-initializes weights manually.
        """
        
        def _init_recursive(module, name_path=""):
            # Handle List of Layers (e.g. model.layers)
            if isinstance(module, list):
                for i, m in enumerate(module):
                    _init_recursive(m, f"{name_path}.{i}")
                return

            # Handle MLX Module
            if not hasattr(module, "children"):
                return
                
            for name, child in module.children().items():
                full_name = f"{name_path}.{name}" if name_path else name
                
                # Check for specific layer types by name or class
                if isinstance(child, nn.Linear):
                    # Default linear init
                    fan_in = child.weight.shape[1]
                    fan_out = child.weight.shape[0]
                    
                    new_w = None
                    
                    # 1. SwiGLU Gating (gate_proj, up_proj) -> Kaiming Normal
                    if "gate_proj" in name or "up_proj" in name:
                        # Gain sqrt(2) for ReLU-like (SiLU)
                        new_w = SovereignInit.kaiming_normal(child.weight, fan_in, gain=math.sqrt(2))
                        
                    # 2. Residual Projections (o_proj, down_proj) -> Xavier + Residual Scale
                    elif "o_proj" in name or "down_proj" in name:
                        # Base: Xavier Uniform (Symmetric)
                        new_w = SovereignInit.xavier_uniform(child.weight, fan_in, fan_out, gain=1.0)
                        # Scaling
                        new_w = SovereignInit.gpt2_residual_scale(new_w, n_layers)
                        
                    # 3. Attention Projections (q_proj, k_proj, v_proj) -> Xavier Uniform
                    elif "q_proj" in name or "k_proj" in name or "v_proj" in name:
                        new_w = SovereignInit.xavier_uniform(child.weight, fan_in, fan_out, gain=1.0)
                        
                    # 4. Head Projection (lm_head) -> Xavier
                    elif "lm_head" in name:
                        new_w = SovereignInit.xavier_uniform(child.weight, fan_in, fan_out, gain=1.0)
                        
                    else:
                        # Fallback for other linears
                        new_w = SovereignInit.xavier_uniform(child.weight, fan_in, fan_out, gain=1.0)
                        
                    if new_w is not None:
                        child.update({"weight": new_w})
                        
                elif isinstance(child, nn.Embedding):
                    # Normal(0, 0.02) for Embeddings
                    new_w = mx.random.normal(shape=child.weight.shape, scale=0.02)
                    child.update({"weight": new_w})
                    
                # Recurse
                _init_recursive(child, full_name)

        _init_recursive(model)
        print(f"✅ Sovereign Initialization Applied (L={n_layers}, Residual Scale={1.0/math.sqrt(2*n_layers):.4f})")

class CheckpointManager:
    # Stub
    def save(self, model, step):
        pass
