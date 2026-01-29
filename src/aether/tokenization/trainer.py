import sentencepiece as spm
import os
from pathlib import Path

class TokenizerFoundry:
    """
    The Aether Tokenizer Factory.
    Trains custom SentencePiece models optimized for Vietnamese/Bilingual context.
    """
    
    @staticmethod
    def train_tokenizer(
        input_file: str,
        model_prefix: str,
        vocab_size: int = 50000,
        character_coverage: float = 0.9995,
        model_type: str = 'bpe'
    ):
        """
        Train a BPE tokenizer on raw text.
        
        Args:
            input_file: Path to one large text file containing the corpus.
            model_prefix: Output prefix for .model and .vocab files.
            vocab_size: Target vocabulary size (Aether Spec: 50000).
            character_coverage: Coverage of characters (high for bilingual).
            model_type: 'bpe', 'unigram', 'char', or 'word'.
        """
        print(f"Igniting Tokenizer Foundry for {model_prefix}...")
        
        # Aether White-Box Specifications:
        # 1. byte_fallback=True -> No <UNK> tokens for rare unicode.
        # 2. normalization_rule_name='nmt_nfkc_cf' -> Standard clean, but we enforce NFC upstream.
        #    Actually, we should use 'identity' if we trust our upstream normalizer completely,
        #    but 'nmt_nfkc_cf' is safer for general robust BPE.
        #    Wait, Masterplan said: "Tắt chuẩn hóa nfkc mặc định để dùng custom NFC của ta".
        #    So we define normalization_rule_name='identity'.
        
        # Construct the command string for SentencePiece Trainer
        # input_sentence_size: Max sentences to load (memory constraint).
        # shuffle_input_sentence: Better convergence.
        
        cmd = (
            f"--input={input_file} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--character_coverage={character_coverage} "
            f"--model_type={model_type} "
            f"--byte_fallback=true "
            f"--normalization_rule_name=identity "
            f"--train_extremely_large_corpus=true "
            f"--shuffle_input_sentence=true "
            f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
            f"--user_defined_symbols=<|user|>,<|assistant|>,<|system|>,<|tool|>"
        )
        
        # Execute training
        spm.SentencePieceTrainer.Train(cmd)
        
        print(f"Tokenizer training complete: {model_prefix}.model")

    @staticmethod
    def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp

if __name__ == "__main__":
    # Test stub
    pass
