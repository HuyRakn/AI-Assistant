import mlx.core as mx
from typing import Tuple, Optional
from .configuration import AetherConfig

class StaticKVCache:
    """
    Titan Memory Manager (Production Grade).
    Uses Pre-allocated buffers to prevent O(N^2) memory copying during generation.
    White-Box Implementation of "PagedAttention" concept (Simplified).
    
    Shape: [Batch, n_kv_heads, Max_Seq_Len, Head_Dim]
    """
    def __init__(self, config: AetherConfig, max_seq_len: int = 4096, batch_size: int = 1):
        self.k_cache = mx.zeros((batch_size, config.n_kv_heads, max_seq_len, config.head_dim), dtype=mx.float32)
        self.v_cache = mx.zeros((batch_size, config.n_kv_heads, max_seq_len, config.head_dim), dtype=mx.float32)
        self.offset = 0 # Current write pointer
        
    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        """
        In-place update of KV Cache.
        keys, values: [Batch, n_kv_heads, Seq_Len_Update, Head_Dim]
        
        Returns:
            The valid slice of cache up to current position.
            keys: [Batch, n_kv_heads, Total_Seq_Len, Head_Dim]
        """
        # Input keys/values usually have shape [B, H, L, D] (axis 2 is seq)
        # But wait, typically RoPE and projection output [B, L, H, D] or [B, H, L, D].
        # In `layers.py`, we transposed to [B, H, L, D].
        # So we expect [B, H, L, D].
        
        B, H, L, D = keys.shape
        
        # Calculate insert step
        start = self.offset
        end = start + L
        
        if end > self.k_cache.shape[2]:
            raise ValueError(f"Cache Overflow! Capacity: {self.k_cache.shape[2]}, Needed: {end}")
            
        # Update Cache (In-place Functional Pattern)
        # MLX arrays are immutable. However, the compiler optimizes `at[...].set(...)` 
        # to an in-place update if the original array is not referenced elsewhere.
        # This allows for efficient rolling buffer management.
        
        self.k_cache = self.k_cache.at[:, :, start:end, :].set(keys)
        self.v_cache = self.v_cache.at[:, :, start:end, :].set(values)
        
        self.offset += L
        
        # Return valid slice
        return (
            self.k_cache[:, :, :self.offset, :],
            self.v_cache[:, :, :self.offset, :]
        )
