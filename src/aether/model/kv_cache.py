import mlx.core as mx

class KVCache:
    """
    Titan Memory Manager.
    Stores Key/Value states for efficient Autoregressive Generation.
    """
    def __init__(self, max_length: int = 2048):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 0
        
    def update_and_fetch(self, new_keys: mx.array, new_values: mx.array):
        """
        Updates cache and returns full history.
        """
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys = mx.concatenate([self.keys, new_keys], axis=1)
            self.values = mx.concatenate([self.values, new_values], axis=1)
            
        self.offset = self.keys.shape[1]
        self.step += 1
        
        return self.keys, self.values
        
    def reset(self):
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 0
