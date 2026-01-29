
import sys
import time
from src.brain import AetherBrain

def test_semantic_memory():
    print(">>> INITIALIZING RAG TEST <<<")
    brain = AetherBrain()
    
    # 1. Teach Aether a specific fact
    fact = "Mật khẩu bí mật của dự án Aether là 'Sentience2026'."
    print(f"\n[Turn 1] User: {fact}")
    # Run think loop to ensure it saves to memory
    for _ in brain.think(fact): pass
    
    # 2. Clear Short-term Memory (Simulate new session)
    brain.forget()
    print("\n[!] Short-term memory wiped.")
    
    # 3. Ask about the fact (Relies on RAG)
    query = "Mật khẩu bí mật của dự án là gì?"
    print(f"\n[Turn 2] User: {query}")
    print("[Aether]: ", end="", flush=True)
    
    response = ""
    for token in brain.think(query):
        print(token, end="", flush=True)
        response += token
    print("\n")
    
    if "Sentience2026" in response:
        print("PASS: RAG successfully retrieved the secret.")
    else:
        print("FAIL: AI forgot the secret.")

if __name__ == "__main__":
    test_semantic_memory()
