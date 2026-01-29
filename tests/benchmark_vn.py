
import time
import sys
from src.brain import AetherBrain
from src.logger import setup_logger

logger = setup_logger("Benchmark")

QUESTIONS = [
    # 1. Logic & Reasoning
    "Nếu tôi có 3 quả táo, tôi ăn 1 quả, sau đó mua thêm 2 quả nữa. Tôi còn bao nhiêu quả? Giải thích các bước.",
    "Tại sao bầu trời lại có màu xanh?",
    # 2. Coding
    "Viết một hàm Python để kiểm tra số nguyên tố và giải thích code.",
    "Viết một đoạn code HTML/CSS tạo một nút bấm chuyển màu khi di chuột vào.",
    # 3. Creative Writing
    "Viết một bài thơ 4 câu về buổi sáng Đà Lạt.",
    "Sáng tác một câu chuyện ngắn về một con mèo biết lập trình.",
    # 4. Cultural Knowledge
    "Giải thích ý nghĩa của món Bánh Chưng trong ngày Tết cổ truyền.",
    "Vịnh Hạ Long nằm ở tỉnh nào và được UNESCO công nhận là gì?",
    # 5. Technical Explanation
    "Giải thích khái niệm 'Zero-copy' trên Apple Silicon cho người không biết kỹ thuật.",
    "Sự khác biệt giữa RAM và SSD là gì?"
]

def run_benchmark():
    logger.info(">>> STARTING AETHER VIETNAMESE BENCHMARK <<<")
    
    try:
        brain = AetherBrain()
    except Exception:
        sys.exit(1)
        
    total_tokens = 0
    total_time = 0
    
    for i, q in enumerate(QUESTIONS):
        logger.info(f"\n[Question {i+1}/{len(QUESTIONS)}]: {q}")
        
        response = ""
        # We invoke think(), which includes retrieval time.
        # But we want to measure perceived latency vs generation speed.
        
        # To get more accurate generation speed from the client side, we can measure time between first token and last token
        first_token_received = False
        gen_start_time = 0
        
        t0 = time.time() # Start of call
        
        for token in brain.think(q):
            if not first_token_received:
                first_token_received = True
                gen_start_time = time.time() # First token arrived
            response += token
            
        t_end = time.time()
        
        # Total latency (Click to Done)
        total_duration = t_end - t0
        
        # Generation Duration (First Token to Done)
        gen_duration = t_end - gen_start_time if first_token_received else 0
        
        token_count = len(brain.tokenizer.encode(response))
        
        # Speed calculation: Tokens / Generation Duration (Pure Neural Speed)
        gen_speed = token_count / gen_duration if gen_duration > 0 else 0
        
        # E2E Speed: Tokens / Total Duration (User Experience Speed)
        e2e_speed = token_count / total_duration if total_duration > 0 else 0
        
        total_tokens += token_count
        # We sum up tokens and duration. For average speed, let's use Generation Speed
        total_time += gen_duration 
        
        logger.info(f"Len: {len(response)} chars | Pure Gen Speed: {gen_speed:.2f} t/s | E2E Speed: {e2e_speed:.2f} t/s")
        
    avg_speed = total_tokens / total_time if total_time > 0 else 0
    logger.info("\n>>> BENCHMARK COMPLETE <<<")
    logger.info(f"Average Speed: {avg_speed:.2f} t/s")
    logger.info(f"Total Questions: {len(QUESTIONS)}")
    
    if avg_speed > 25:
        logger.info("RESULT: EXCELLENT (Production Ready)")
    elif avg_speed > 20:
        logger.info("RESULT: GOOD (Acceptable)")
    else:
        logger.warning("RESULT: SLOW (Needs Optimization)")

if __name__ == "__main__":
    run_benchmark()
