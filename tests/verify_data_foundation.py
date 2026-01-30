import unittest
import sys
import os
import shutil
sys.path.append(os.getcwd())

from src.aether.tokenization.bpe import AetherBPE
from src.aether.preprocessing.english import PorterStemmer

class TestDataFoundation(unittest.TestCase):
    
    def test_bpe_determinism(self):
        print("\n🧪 Testing BPE Determinism & Serialization...")
        texts = ["The quick brown fox", "jumps over the lazy dog", "the dog is brown"]
        
        # Train Model A
        bpe1 = AetherBPE(vocab_size=100)
        bpe1.train(texts)
        encoded1 = bpe1.encode("The quick fox")
        
        # Save Model A
        os.makedirs("tests/artifacts", exist_ok=True)
        path = "tests/artifacts/test_vocab.json"
        bpe1.save(path)
        
        # Load Model B
        bpe2 = AetherBPE()
        bpe2.load(path)
        encoded2 = bpe2.encode("The quick fox")
        
        # Check Identity
        print(f"Original: {encoded1}")
        print(f"Loaded:   {encoded2}")
        self.assertEqual(encoded1, encoded2, "BPE Serialization Failed: IDs do not match!")
        
        # Check Deterministic IDs (Check 'The' always same ID if trained on same data deterministically)
        # Note: 'The' is 'T', 'h', 'e', '</w>' if no merge.
        # But 'The' is distinct from 'the'.
        
        # Decode check
        decoded = bpe2.decode(encoded2)
        print(f"Decoded: '{decoded}'")
        # should match input roughly (case sensitive in our BPE?)
        # Our BPE implementation preserves case in token map
        # But wait, input was "The quick fox".
        # Spaces are handled via </w>.
        
        self.assertEqual(decoded, "The quick fox")
        print("✅ BPE Determinism Verified.")
        
    def test_porter_stemmer(self):
        print("\n🧪 Testing Porter Stemmer Accuracy...")
        cases = {
            "caresses": "caress",
            "ponies": "poni",
            "ties": "ti",
            "caress": "caress",
            "cats": "cat",
            "feed": "feed",
            "agreed": "agre",
            "plastered": "plaster",
            "bled": "bled",
            "motoring": "motor",
            "sing": "sing", # s-ing -> m(s)=0? NO. m(s) = measure. 's' is C. m=0.
            # wait, 'sing' -> 's' is C, 'i' is V, 'ng' are C.
            # contains vowel? yes. step1b 'ing'.
            # 'sing' -> 's'. m('s')? C. m=0.
            # Rule: replace ing if stem contains vowel. stem='s'. contains vowel? No.
            # So 'sing' -> 'sing'. Correct.
            "conflated": "conflat",
            "troubled": "troubl",
            "sized": "size",
            "hopping": "hop", # step1b: double consonant reduction l/s/z? No. p -> hop.
            "tanned": "tan",
            "falling": "fall",
            "hissing": "hiss",
            "fizzed": "fizz",
            "fail": "fail",
            "filing": "file",
            "relational": "relat",
            "conditional": "condit",
            "rational": "ration",
            "valuing": "valu",
            "necessitating": "necessit",
            "step": "step"
        }
        
        failures = []
        for word, expected in cases.items():
            got = PorterStemmer.stem(word)
            if got != expected:
                failures.append(f"{word}: {got} != {expected}")
                
        if failures:
            print("❌ Stemming Failures:")
            for f in failures: print(f"   {f}")
            self.fail("Porter Stemmer failed on some cases.")
        else:
            print("✅ Porter Stemmer Passed Standard Cases.")

if __name__ == "__main__":
    unittest.main()
