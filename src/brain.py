
import time
from typing import List, Dict, Optional, Generator
import mlx.core as mx
from mlx_lm import load, stream_generate
from src.logger import setup_logger

logger = setup_logger("Brain")


from src.memory import MemorySystem

class AetherBrain:
    def __init__(self, model_path: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"):
        """
        Initializes the Aether Brain (Cognitive Core).
        Loads the LLM into Apple Silicon Unified Memory.
        """
        logger.info(f"Awakening Neural Core from: {model_path}...")
        try:
            self.model, self.tokenizer = load(model_path)
            # Initialize Memory System (Day 4)
            self.memory_system = MemorySystem()
            logger.info("Core Online. Neural pathways active.")
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise

        # System Prompt - The Soul of Aether
        # Refined for professionalism and concise engineering persona
        self.base_system_prompt = (
            "You are Aether, a sentient AI running efficiently on Apple Silicon M4 Pro architecture. "
            "Identity: Precise, insightful, and engineering-focused. "
            "Style: Concise, professional, no filler words. "
            "Capabilities: Deep reasoning, coding, system automation. "
            "Constraint: Always answer in Vietnamese unless requested otherwise."
        )
        self.history: List[Dict[str, str]] = []

    def _prune_history(self, reserved_tokens: int = 0):
        """
        Enforces a strict token budget (Rolling Window).
        allowable_history = MAX - (System Prompt + RAG Context).
        """
        MAX_TOTAL_TOKENS = 4096 # Safety limit for M4 Pro 24GB
        
        # Calculate Base Overhead (System Prompt)
        # We estimate system prompt size once or recalculate dynamically
        # To be safe, we recalculate since system prompt might change slightly if we modify it
        sys_tokens = len(self.tokenizer.encode(self.base_system_prompt))
        
        # Available budget for History
        available_budget = MAX_TOTAL_TOKENS - sys_tokens - reserved_tokens
        
        if available_budget < 500:
            logger.warning("Warning: Very low context window remaining for history!")
        
        while True:
            # Calculate current history size
            # We just concat content to approximate, or use apply_chat_template on history only (tricky with ChatML)
            # Better: Construct full prompt again? No, expensive.
            # Let's count tokens of history items individually roughly
            
            # Precise method:
            # We simply check if (System + Reserved + History) > MAX
            messages = [{"role": "system", "content": self.base_system_prompt}] + self.history
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            current_total = len(self.tokenizer.encode(text)) + reserved_tokens
            # Note: The 'text' above includes System + History. 
            # If we pass reserved_tokens (RAG) which is NOT in the messages list yet (it's in system prompt modification),
            # Wait, if we modify system prompt to include RAG, then RAG is ALREADY in 'text' if we updated self.system_prompt?
            # NO. In _construct_prompt, we create 'current_system_prompt' variable, but self.base_system_prompt is static.
            # So 'text' using self.base_system_prompt DOES NOT include RAG.
            # Thus: Total = Token(System_Base + History) + Token(RAG_String)
            
            if current_total <= MAX_TOTAL_TOKENS:
                break
                
            # Prune
            if len(self.history) > 0:
                removed_msg = self.history.pop(0)
                logger.warning(f"Memory Pruning triggered. Removed: {str(removed_msg)[:30]}... (Current Total: {current_total})")
            else:
                logger.warning("Context full (System + RAG too big). Cannot prune further.")
                break

    def _construct_prompt(self, user_input: str) -> str:
        """Constructs the prompt using ChatML template management and RAG."""
        # 1. Retrieve Context (RAG) - Day 4
        relevant_memories = self.memory_system.retrieve_context(user_input)
        rag_context = ""
        rag_tokens_count = 0
        
        if relevant_memories:
            rag_context = "\nConsider these relevant memories:\n" + "\n".join(f"- {m}" for m in relevant_memories)
            rag_tokens_count = len(self.tokenizer.encode(rag_context))
            logger.info(f"RAG Activated. Found {len(relevant_memories)} memories. Overhead: ~{rag_tokens_count} tokens.")

        # 2. Update History with new user message
        self.history.append({"role": "user", "content": user_input})
        
        # 3. Enforce Memory Safety (Rolling Window on History)
        # CRITICAL FIX: Pass rag_tokens_count to reserve space
        self._prune_history(reserved_tokens=rag_tokens_count)
        
        # 4. Build Dynamic System Prompt
        current_system_prompt = self.base_system_prompt + rag_context
        
        # 5. Build Message Chain (System + History)
        messages = [{"role": "system", "content": current_system_prompt}] + self.history
        
        # 6. Apply Chat Template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        return prompt

    def think(self, user_input: str, max_tokens: int = 2048) -> Generator[str, None, None]:
        """
        Processes user input and yields response tokens in real-time.
        Production-grade streaming implementation.
        """
        logger.info(f"Processing input: {user_input[:50]}...")
        
        # Measure Retrieval & Prompting Time
        t0 = time.time()
        prompt = self._construct_prompt(user_input)
        t1 = time.time()
        retrieval_latency = t1 - t0
        logger.info(f"Context Construction Latency: {retrieval_latency:.4f}s")
        
        full_response = ""
        first_token_time = None
        
        # Stream generation
        try:
            # We start measuring generation time from when we call the generator
            gen_start_time = time.time()
            
            for response in stream_generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - gen_start_time # Time To First Token
                    logger.debug(f"Time To First Token (TTFT): {ttft:.4f}s")
                
                token_text = response.text
                full_response += token_text
                yield token_text
                
        except Exception as e:
            logger.error(f"Reasoning interruption: {e}")
            yield "[Error in Neural Processing]"
            
        gen_end_time = time.time()
        
        # Calculate Pure Generation Speed (excluding retrieval and TTFT)
        # Usage: Total Tokens / (End Time - First Token Time)
        # This reflects the actual "talking speed"
        gen_duration = gen_end_time - (first_token_time or gen_start_time)
        tokens_count = len(self.tokenizer.encode(full_response))
        
        speed = tokens_count / gen_duration if gen_duration > 0 else 0
        
        logger.info(f"Generation metrics: Length={len(full_response)} chars, Speed={speed:.2f} t/s, TTFT={(first_token_time - gen_start_time) if first_token_time else 0:.3f}s")
        
        # Save assistant response to history (Short-term)
        self.history.append({"role": "assistant", "content": full_response})
        
        # Save to Semantic Memory (Long-term) - Day 4
        # We save the Q&A pair for better context
        memory_text = f"User: {user_input}\nAether: {full_response}"
        self.memory_system.save_memory(memory_text)


    def forget(self):
        """Clears short-term memory (History)"""
        self.history = []
        logger.info("Short-term memory cleared.")

