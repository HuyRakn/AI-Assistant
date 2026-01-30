import sys
import os
import glob
sys.path.append(os.getcwd())

from src.aether.tokenization.gluer import CompoundGluer

def main():
    print("🛡️  Titan Linguistic Engine: Compound Word Gluer")
    
    # Configuration
    data_dir = "data/processed/shards"
    output_dir = "data/processed/glued_shards"
    os.makedirs(output_dir, exist_ok=True)
    
    input_files = sorted(glob.glob(f"{data_dir}/*.txt"))
    if not input_files:
        print(f"❌ No corpus shards found in {data_dir}")
        return

    # Initialize Gluer
    # threshold 4.0 is decent for filtering only significant pairs
    gluer = CompoundGluer(min_count=5, threshold=4.0)
    
    # 1. Train (Count Stats)
    print(f"phase 1: Learning Bigram Statistics from {len(input_files)} shards...")
    for fpath in input_files:
        gluer.train(fpath)
        
    # 2. Glue (Rewrite)
    print("Phase 2: Gluing Corpus...")
    for fpath in input_files:
        fname = os.path.basename(fpath)
        out_path = os.path.join(output_dir, fname)
        gluer.glue(fpath, out_path)
        
    print("✨ Compound Gluing Complete.")

if __name__ == "__main__":
    main()
