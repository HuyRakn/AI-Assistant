import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from aether.dedup.minhash import MinHashLSH

def test_redis_dedup():
    print("🧪 Testing Redis Deduplication...")
    try:
        lsh = MinHashLSH(redis_host='localhost', redis_port=6379)
        lsh.clear() # Start fresh
        
        text1 = "The quick brown fox jumps over the lazy dog."
        text2 = "The quick brown fox jumps over the lazy dog." # Exact duplicate
        text3 = "The quick brown fox jumps over the lazy cat." # Near duplicate
        text4 = "Completely different text about machine learning."
        
        sig1 = lsh.compute_signature(text1)
        sig2 = lsh.compute_signature(text2)
        sig3 = lsh.compute_signature(text3)
        sig4 = lsh.compute_signature(text4)
        
        print("1. Querying empty DB...")
        if lsh.query(sig1):
            print("❌ Error: Found text1 in empty DB")
        else:
            print("✅ Empty DB query correct")
            
        print("2. Inserting text1...")
        lsh.insert(sig1, "doc_1")
        
        print("3. Querying text1 again...")
        if lsh.query(sig1):
            print("✅ Found text1")
        else:
            print("❌ Error: Could not find text1")

        print("4. Querying text2 (Duplicate)...")
        if lsh.query(sig2):
             print("✅ Found text2 (Duplicate)")
        else:
             print("❌ Error: Failed to detect duplicate")

        # Check Near Duplicate (Jaccard check needed usually, but LSH might collide if similar enough)
        # 128 perms, 20 bands -> threshold ~0.5? (1/20)^(1/rows)?
        # Threshold 0.8 usually implies tight bands.
        
        print("5. Querying text4 (Different)...")
        if lsh.query(sig4):
             print("❌ Error: False Positive on text4")
        else:
             print("✅ Correctly ignored text4")

    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    test_redis_dedup()
