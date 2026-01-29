
import sys
import time
from src.brain import AetherBrain

def test_cognition():
    print(">>> INITIALIZING AETHER COGNITION TEST <<<")
    try:
        brain = AetherBrain()
    except Exception as e:
        print(f"CRITICAL ERROR: Initialization failed - {e}")
        sys.exit(1)
        
    conversation = [
        "Xin chào, bạn tên là gì?",
        "Bạn có thể giải thích ngắn gọn về cơ chế hoạt động của RAM máy tính không?",
        "Vậy 24GB RAM trên M4 Pro thì làm được gì đặc biệt?",
    ]
    
    print("\n--- STARTING INTERACTION LOOPS ---")
    
    for i, user_query in enumerate(conversation):
        print(f"\n[Turn {i+1}] User: {user_query}")
        print("[Aether]: ", end="", flush=True)
        
        full_response = ""
        start_t = time.time()
        
        # Test Streaming
        for token in brain.think(user_query):
            print(token, end="", flush=True)
            full_response += token
            
        print("\n")
        duration = time.time() - start_t
        print(f">>> Metrics: {len(full_response)} chars in {duration:.2f}s")
        
    print("\n--- MEMORY CHECK ---")
    print(f"History length: {len(brain.history)}")
    if len(brain.history) == len(conversation) * 2:
        print("PASS: History preserved correctly.")
    else:
        print(f"FAIL: History mismatch. Expected {len(conversation)*2}, got {len(brain.history)}")
        
    print("\n>>> DAY 2 VERIFICATION COMPLETE <<<")

if __name__ == "__main__":
    test_cognition()
