import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import mlx.core as mx
from aether.preprocessing.english import EnglishNormalizer
from aether.tokenization.bpe import AetherBPE
from aether.training.manual import ManualCrossEntropy, ManualAdamW

def test_english_norm():
    print("Test 1: English Normalizer")
    text = "The quick brown Foxes are Running smoothly with Apples"
    norm = EnglishNormalizer.normalize(text)
    # foxes -> fox, running -> run, apples -> appl (porter style)
    # Case fold -> lower
    print(f"Input: {text}")
    print(f"Output: {norm}")
    
    # Assert stems for specific words (simplified implementation check)
    assert "fox" in norm.split() or "foxe" in norm.split()
    assert "run" in norm.split()
    print("✅ English Normalizer Passed")

def test_bpe():
    print("\nTest 2: AetherBPE")
    bpe = AetherBPE()
    corpus = ["low low low low", "lowest lowest", "newer newer", "wider"]
    # low: l o w </w>
    # er common suffix...
    bpe.train(corpus, vocab_size=300) # tiny vocab
    
    text = "lower"
    ids = bpe.encode(text)
    print(f"Encoded '{text}': {ids}")
    
    # Check if 'low' or 'er' got merged if frequently enough
    print("✅ BPE Logic Executed")

def test_manual_math():
    print("\nTest 3: Manual Math (Loss & Optimizer)")
    
    # Loss
    logits = mx.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]]) # B=2, V=3
    targets = mx.array([0, 1]) # Correct classes
    
    criterion = ManualCrossEntropy()
    loss = criterion(logits, targets)
    print(f"Manual Loss: {loss.item():.4f}")
    
    # Compare with Reference?
    # loss1 = -log(softmax(2,1,0.1)[0]) = -log(0.659) = 0.417
    # loss2 = -log(softmax(0.5,2.5,0.3)[1]) = -log(0.843) = 0.170
    # Mean = 0.293
    # Let's see output.
    
    opt = ManualAdamW(learning_rate=0.1)
    opt.step = 1 # Simulate step 1 for manual apply_gradients call
    
    # Dummy param
    p = mx.array([1.0])
    g = mx.array([0.5])
    
    # Step 1
    # Note: verify_manual passed dicts by hand, but in real code model provides structure.
    # We must match key structure for state tracking.
    
    updates1 = opt.apply_gradients({'p': p}, {'p': g}, prefix="test")
    p_new1 = updates1['p']
    
    print(f"Step 1: Param {p.item():.4f} -> {p_new1.item():.4f}")
    
    # Check if state saved
    state_key = "test.p"
    assert state_key in opt.state, "Optimizer failed to save state"
    m1 = opt.state[state_key]['m']
    print(f"State m1: {m1.item():.4f}")
    assert m1.item() != 0
    
    # Step 2 (Same gradient)
    # If momentum works, update should be slightly different or consistent logic
    updates2 = opt.apply_gradients({'p': p_new1}, {'p': g}, prefix="test")
    p_new2 = updates2['p']
    print(f"Step 2: Param {p_new1.item():.4f} -> {p_new2.item():.4f}")
    
    m2 = opt.state[state_key]['m']
    print(f"State m2: {m2.item():.4f}")
    # m should accumulate: m2 = 0.9*m1 + 0.1*g
    expected_m2 = 0.9 * m1.item() + 0.1 * 0.5
    assert abs(m2.item() - expected_m2) < 1e-5
    
    print("✅ Manual Optimizer State Persistence Verified")

if __name__ == "__main__":
    test_english_norm()
    test_bpe()
    test_manual_math()
