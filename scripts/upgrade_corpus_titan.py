import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.tokenization.gluer import CompoundGluer
from aether.tokenization.trainer import TokenizerFoundry

def run_titan_upgrade():
    input_corpus = "data/processed/clean_corpus_vi.txt"
    glued_corpus = "data/processed/glued_corpus_vi.txt"
    model_prefix = "models/tokenizer/aether_vi_titan_50k"
    
    if not os.path.exists(input_corpus):
        print("❌ Clean corpus missing. Run Phase 2 first.")
        return

    # 1. Initialize and Train Gluer
    # High threshold to only glue very strong compounds (e.g. "Hồ_Chí_Minh", "Cộng_hòa")
    # PMI > 2 or 3 usually indicates strong association
    gluer = CompoundGluer(min_count=50, threshold=4.0) 
    
    gluer.train(input_corpus)
    
    # 2. Glue
    gluer.glue(input_corpus, glued_corpus)
    
    # 3. Train Titan Tokenizer
    print("⚔️  Forging Titan Tokenizer...")
    TokenizerFoundry.train_tokenizer(
        input_file=glued_corpus,
        model_prefix=model_prefix,
        vocab_size=50000,
        model_type='bpe',
        # Need to ensure '_' is treated as a character part of word? 
        # SP treats '_' as literal if it sees it in text
        character_coverage=0.9995
    )
    print("✅ Titan Upgrade Complete.")

if __name__ == "__main__":
    run_titan_upgrade()
