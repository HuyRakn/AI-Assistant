import mlx.core as mx
import mlx.nn as nn
import math
from typing import Optional, Tuple
from .configuration import AetherConfig
from .layers import RMSNorm, AetherAttention, SwiGLU

class AetherDecoderLayer(nn.Module):
    """
    A single Transformer Block.
    Flow:
    Input -> RMSNorm -> Attention -> + Residual
          -> RMSNorm -> SwiGLU    -> + Residual
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.self_attn = AetherAttention(config)
        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.dim, eps=config.norm_eps)
        self.post_attention_layernorm = RMSNorm(config.dim, eps=config.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        # Attention Block (Pre-Norm)
        r = self.input_layernorm(x)
        h, cache = self.self_attn(r, mask=mask, cache=cache)
        x = x + h # Residual Connection
        
        # FFN Block (Pre-Norm)
        r = self.post_attention_layernorm(x)
        x = x + self.mlp(r) # Residual Connection
        
        return x, cache

class AetherModel(nn.Module):
    """
    The Aether Titan Backbone.
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = [
            AetherDecoderLayer(config) for _ in range(config.n_layers)
        ]
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        
    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None, cache=None):
        h = self.embed_tokens(inputs)
        
        # Iterate layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, mask=mask, cache=layer_cache)
            if cache is not None:
                new_cache.append(c)
                
        h = self.norm(h)
        
        return h, new_cache

    def apply_initialization(self):
        """
        Sovereign Initialization (White-Box).
        Reset weights with mathematical variance scaling (He/Xavier) 
        instead of relying on framework defaults.
        """
        def init_fn(key, param):
            # 1. Embeddings: Normal(0, 1) or Uniform
            if "embed_tokens" in key:
                 # Standard normal * scaling
                 return mx.random.normal(param.shape) * 0.02
                 
            # 2. Linear Layers
            if "weight" in key and len(param.shape) == 2:
                fan_in = param.shape[1] # Input dim
                fan_out = param.shape[0] # Output dim
                
                # Xavier/Glorot for Attention (Symmetric)
                if "self_attn" in key:
                    # print(f"DEBUG INIT: Re-init {key} (Xavier)")
                    limit = math.sqrt(6 / (fan_in + fan_out))
                    return mx.random.uniform(-limit, limit, param.shape)
                    
                # He Kaiming for SwiGLU (ReLU/SiLU activation)
                if "mlp" in key:
                    # print(f"DEBUG INIT: Re-init {key} (He)")
                    std = math.sqrt(2 / fan_in)
                    return mx.random.normal(param.shape) * std
                    
            return param

        # Recursively apply initialization
        def walk_and_init(params, prefix=""):
            updates = {}
            for k, v in params.items():
                curr_key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    updates[k] = walk_and_init(v, curr_key)
                elif isinstance(v, list):
                    # Handle lists (e.g. layers)
                    # Note: MLX parameters for a list of modules is a list of dicts
                    new_list = []
                    for i, item in enumerate(v):
                         # Item could be dict (submodule params) or array (if list of weights)
                         # We treat it recursively.
                         list_key = f"{curr_key}.{i}"
                         if isinstance(item, dict):
                             new_list.append(walk_and_init(item, list_key))
                         elif isinstance(item, mx.array):
                             new_list.append(init_fn(list_key, item))
                         else:
                             new_list.append(item)
                    updates[k] = new_list
                elif isinstance(v, mx.array):
                    updates[k] = init_fn(curr_key, v)
            return updates

        # Get current parameters (tree structure)
        current_tree = self.parameters()
        
        # Calculate new initialized weights
        new_tree = walk_and_init(current_tree)
        
        # Update the model
        self.update(new_tree)
        print("✅ Sovereign Initialization Applied (Xavier/He).")


class AetherForCausalLM(nn.Module):
    """
    End-to-End Language Model Head.
    Adds a linear projection to vocab size at the end.
    """
    def __init__(self, config: AetherConfig):
        super().__init__()
        self.model = AetherModel(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights if usually done (Llama doesn't mandate tied weights but it matches embedding often)
        # For simplicity/explicit structure, keeping separate or tied?
        # Typically SOTA models have separate or tied. Let's keep separate for flexibility unless specified.
        # But wait, weights sharing is good for size. Let's verify Titan specs? 
        # User didn't specify. Standard Llama 2 uses non-tied.
        
        # --- SURVIVAL TECH: Manual Weight Initialization ---
        from .utils import WeightInit
        WeightInit.init_model(self)
        # ---------------------------------------------------
        
    def __call__(self, inputs, mask=None, cache=None):
        out, new_cache = self.model(inputs, mask, cache)
        logits = self.lm_head(out)
        return logits, new_cache
