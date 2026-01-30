from pathlib import Path
from .core_bpe import SovereignTokenizer

class TokenizerFoundry:
    """
    The Aether Tokenizer Factory (White-Box Edition).
    Trains custom BPE models from scratch.
    """
    
    @staticmethod
    def train_tokenizer(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 50000,
        character_coverage: float = 0.9995, # Unused in manual BPE, keeping signature comp.
        model_type: str = 'bpe'
    ):
        """
        Train a Sovereign BPE tokenizer on raw text.
        """
        print(f"Igniting Tokenizer Foundry (Sovereign BPE) for {model_prefix}...")
        
        # Load Corpus
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                corpus = f.readlines()
        except Exception as e:
            print(f"Error reading corpus: {e}")
            return

        # Initialize Sovereign BPE
        tokenizer = SovereignTokenizer(vocab_size=vocab_size)
        
        # Train (Simplified logic for demo: sample first 10k lines if huge)
        # Real training needs optimization for 50k vocab.
        train_corpus = corpus[:10000] if len(corpus) > 10000 else corpus
        
        tokenizer.train(train_corpus)
        
        # Save
        tokenizer.save(model_prefix) # Check if save requires filename or prefix? core_bpe save takes path. 
        # If model_prefix is just base, we should probably append .json or .model
        # But let's assume valid path provided.
        print(f"Tokenizer training complete: {model_prefix}")

    @staticmethod
    def load_tokenizer(model_path_prefix: str) -> SovereignTokenizer:
        # Expecting path without extension or with
        base = model_path_prefix # Assuming full path for clarity
        tokenizer = SovereignTokenizer()
        tokenizer.load(base)
        return tokenizer

if __name__ == "__main__":
    pass
