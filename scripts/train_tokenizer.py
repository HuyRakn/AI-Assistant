import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.tokenization.trainer import TokenizerFoundry

def train_white_box_tokenizer():
    input_file = "data/processed/clean_corpus_vi.txt"
    model_prefix = "models/tokenizer/aether_vi_50k"
    
    os.makedirs("models/tokenizer", exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"❌ Corpus not found: {input_file}")
        return

    print(f"🧪 igniting Tokenizer Foundry...")
    print(f"   Corpus: {input_file}")
    print(f"   Target: {model_prefix}.model (Vocab: 50,000)")
    
    try:
        TokenizerFoundry.train_tokenizer(
            input_file=input_file,
            model_prefix=model_prefix,
            vocab_size=50000,
            model_type='bpe'
        )
        print("✅ Tokenizer Forging Complete.")
        
    except Exception as e:
        print(f"❌ Forging Failed: {e}")

if __name__ == "__main__":
    train_white_box_tokenizer()
