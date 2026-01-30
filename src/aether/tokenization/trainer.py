from pathlib import Path
from .bpe import AetherBPE

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
        tokenizer = AetherBPE()
        
        # Train (Simplified logic for demo: sample first 10k lines if huge)
        # Real training needs optimization for 50k vocab.
        train_corpus = corpus[:10000] if len(corpus) > 10000 else corpus
        
        tokenizer.train(train_corpus, vocab_size=vocab_size)
        
        # Save
        tokenizer.save(model_prefix)
        print(f"Tokenizer training complete: {model_prefix}.merges")

    @staticmethod
    def load_tokenizer(model_path_prefix: str) -> AetherBPE:
        # Expecting path without extension or with
        base = model_path_prefix.replace(".merges", "").replace(".model", "")
        tokenizer = AetherBPE()
        tokenizer.load(base)
        return tokenizer

if __name__ == "__main__":
    pass
