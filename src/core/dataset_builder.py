import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Takes generated instruction pairs and builds the final dataset (JSONL).
    This output is what Anvil (Training phase) consumes.
    """
    
    def __init__(self, output_dir: str = "data/datasets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_to_jsonl(self, pairs: List[Dict[str, Any]], filename: str) -> str:
        """
        Exports a list of instruction pairs to a JSONL format suitable for SFT/LoRA training.
        
        Args:
            pairs: List of dicts, typically containing 'instruction', 'context' (optional), and 'response'.
            filename: Name of the output file (e.g., 'go_reviewer_v1.jsonl')
            
        Returns:
            The absolute path to the saved file.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        logger.info(f"Exporting {len(pairs)} pairs to {filepath}")
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    # HuggingFace Datasets standard format
                    json_record = {
                        "messages": [
                            {"role": "system", "content": "You are a GuildLM specialist."},
                            {"role": "user", "content": pair.get("instruction", "") + "\n\n" + pair.get("context", "")},
                            {"role": "assistant", "content": pair.get("response", "")}
                        ]
                    }
                    f.write(json.dumps(json_record) + "\n")
                    
            logger.info(f"Successfully exported dataset: {filepath}")
            return os.path.abspath(filepath)
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    builder = DatasetBuilder()
    mock_pairs = [
        {"instruction": "Explain this code", "context": "x := 5", "response": "It assigns 5 to x."},
        {"instruction": "Find bug", "context": "fmt.Print(x)", "response": "x is undefined."}
    ]
    
    saved_path = builder.export_to_jsonl(mock_pairs, "mock_dataset.jsonl")
    print(f"Dataset ready for Anvil at: {saved_path}")
