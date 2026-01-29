import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from aether.data.miners.knowledge import WikipediaMiner

def main():
    output_dir = "data/raw/wikipedia"
    
    # Tier 1: Vietnamese Wikipedia
    print("🚀 Launching Aether Knowledge Miner...")
    miner = WikipediaMiner(output_dir, language='vi', date='20231101')
    
    parquet_path = miner.process_to_parquet()
    
    if parquet_path:
        print(f"\n🎉 Mission Accomplished: Data secured at {parquet_path}")
        # DVC Track
        print(f"👉 To track with DVC: dvc add {parquet_path}")
    else:
        print("\n💥 Mission Failed.")

if __name__ == "__main__":
    main()
