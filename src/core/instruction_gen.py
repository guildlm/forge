import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class InstructionGenerator:
    """
    Takes raw code content and uses a larger teacher model (e.g., Llama 3.1 70B, Qwen2.5 72B)
    to generate high-quality Instruction-Response pairs for supervised fine-tuning (SFT).
    """
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize the instruction generator.
        In a real deployment, this would connect to an OpenAI-compatible endpoint (like vLLM)
        running the teacher model.
        """
        self.api_url = api_url or "http://localhost:8000/v1/chat/completions"
        self.api_key = api_key or "dummy-key"
        
        # We define roles for different guild specialists
        self.prompts = {
            "go_reviewer": "You are a senior Go engineer. Review the following code, point out bugs or race conditions, and suggest improvements.",
            "go_generator": "You are an expert Go programmer. Given the following context, write idiomatic Go code that fulfills the requirement.",
            "go_explainer": "You are a technical writer. Explain what the following Go code does in clear, concise language."
        }

    def generate_pairs(self, code_content: str, role: str = "go_explainer", max_pairs: int = 1) -> List[Dict[str, str]]:
        """
        Generate Q&A pairs for a given snippet of code.
        
        Args:
            code_content: The raw source code.
            role: The perspective the teacher model should take.
            max_pairs: Number of synthetic instructions to generate.
            
        Returns:
            List of dictionaries containing {"instruction": "...", "response": "..."}
        """
        # Truncate to avoid context window explosion
        if len(code_content) > 8000:
            code_content = code_content[:8000] + "\n// ... truncated"
            
        system_prompt = self.prompts.get(role, self.prompts["go_explainer"])
        
        logger.debug(f"Generating instructions for role: {role}")
        
        # MOCK IMPLEMENTATION:
        # In the actual pipeline, we would call requests.post(self.api_url, ...) here.
        # For demonstration and local testing, we return a mock pair.
        
        mock_instruction = "Explain the purpose of this Go file and its main components."
        mock_response = (
            "This Go file implements core functionality. "
            "It defines several structs and interfaces to handle the domain logic. "
            "The functions are highly concurrent and use channels for synchronization.\n\n"
            "```go\n"
            "// Example based on the provided code\n"
            "```"
        )
        
        pairs = []
        for _ in range(max_pairs):
            pairs.append({
                "instruction": mock_instruction,
                "response": mock_response,
                "context": code_content[:500] + "..."  # Save a snippet of context
            })
            
        return pairs

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    generator = InstructionGenerator()
    sample_code = "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"Hello GuildLM\")\n}"
    
    pairs = generator.generate_pairs(sample_code, role="go_reviewer")
    for idx, p in enumerate(pairs):
        print(f"\n--- Pair {idx+1} ---")
        print(f"Instruction: {p['instruction']}")
        print(f"Response: {p['response'][:100]}...")
