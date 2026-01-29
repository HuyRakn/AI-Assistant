
import time
import mlx.core as mx
from mlx_lm import load, generate

def benchmark():
    model_path = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    
    prompt = "Viết một bài thơ về trí tuệ nhân tạo bằng tiếng Việt."
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("\nStarting generation...")
    start_time = time.time()
    response = generate(model, tokenizer, prompt=prompt_formatted, verbose=True, max_tokens=200)
    end_time = time.time()
    
    print("\n--- Benchmark Results ---")
    print(f"Total time: {end_time - start_time:.2f}s")
    # Note: verbose=True already prints token generation speed
    
if __name__ == "__main__":
    benchmark()
