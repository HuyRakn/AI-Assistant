import mlx.core as mx
import mlx.nn as nn
import math

class WeightInit:
    """
    The Foundation Layer.
    Manual initialization schemes to prevent 'Dead Neurons' or 'Exploding Gradients'.
    Survival Tech.
    """
    
    @staticmethod
    def xavier_uniform(tensor: mx.array, fan_in: int, fan_out: int, gain: float = 1.0) -> mx.array:
        """
        Xavier/Glorot Uniform.
        Good for Sigmoid/Tanh/Linear.
        Limit = gain * sqrt(6 / (fan_in + fan_out))
        """
        limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return mx.random.uniform(low=-limit, high=limit, shape=tensor.shape)

    @staticmethod
    def he_normal(tensor: mx.array, fan_in: int, gain: float = 1.0) -> mx.array:
        """
        He/Kaiming Normal. 
        Critical for ReLU/GeLU networks (Aether).
        Std = gain / sqrt(fan_in) (Adjusted for "fan_in" mode)
        """
        std = gain / math.sqrt(fan_in)
        return mx.random.normal(shape=tensor.shape, scale=std)

    @staticmethod
    def init_model(model):
        """
        Walks the model tree and re-initializes weights manually.
        """
        # Handle list container (e.g. self.layers = [...])
        if isinstance(model, list):
            for m in model:
                WeightInit.init_model(m)
            return
            
        # Handle MLX Module
        if not hasattr(model, "children"):
            return

        # Iterate over all submodules
        for name, module in model.children().items():
             if isinstance(module, nn.Linear):
                 # He Normal for Linear layers (assuming ReLu/SiLU/SwiGLU)
                 fan_in = module.weight.shape[1]
                 new_w = WeightInit.he_normal(module.weight, fan_in, gain=math.sqrt(2))
                 module.update({"weight": new_w})
                 
             elif isinstance(module, nn.Embedding):
                 # Normal(0, 0.02) for Embeddings
                 new_w = mx.random.normal(shape=module.weight.shape, scale=0.02)
                 module.update({"weight": new_w})
                 
             elif isinstance(module, (nn.Module, list)):
                 # Recurse
                 WeightInit.init_model(module)

class CheckpointManager:
    # Keep stub for compatibility
    def save(self, model, step):
        pass
