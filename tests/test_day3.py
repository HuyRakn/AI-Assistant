
import sys
import time
from src.brain import AetherBrain

def test_memory_safety():
    print(">>> INITIALIZING MEMORY SAFETY STRESS TEST <<<")
    # Using a fake long prompt to fill up context quickly
    LONG_TEXT = "Tri thức là sức mạnh. " * 50 # ~200 tokens each
    
    brain = AetherBrain()
    
    print("\n--- FILLING CONTEXT (Target > 4096 tokens) ---")
    
    start_history_size = 0
    
    # We will simulate a loop until we see pruning happening
    for i in range(1, 40): # Loop enough to overflow
        user_input = f"Input {i}: {LONG_TEXT}"
        print(f"Adding Turn {i}...", end="\r")
        
        # We assume think() calls _construct_prompt which calls _prune_history
        # We don't need to actually run generation (slow), just check the prompt construction logic
        # But `think` runs gen. Let's use a private method check or just run `think` with max_tokens=1
        
        # To speed up, we just construct the prompt state
        brain._construct_prompt(user_input)
        
        # Fake an assistant response to balance the conversation
        brain.history.append({"role": "assistant", "content": "Acknowledged."})
        
        current_history_len = len(brain.history)
        
        # If history length stops growing despite us adding items, pruning is working
        if i > 10 and current_history_len < i * 2:
            print(f"\n[!] Pruning Detected at Turn {i}!")
            print(f"Current History items: {current_history_len}")
            print("PASS: Rolling Window is active.")
            break
    else:
        print("\n[!] WARNING: Pruning did not trigger. Check max_tokens setting.")

if __name__ == "__main__":
    test_memory_safety()
