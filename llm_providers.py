import json

from typing import Any


class LLMProvider:
    def __init__(self, name: str):
        self.name = name
        
    def load_responses(self) -> Any:
        with open(f"./responses/{self.name}.json") as f:
            return json.load(f)
    
    def write_similarity_output(self, embedding_model_name: str, output: str):
        with open(f"./outputs/{self.name}_{embedding_model_name}_output.txt", "w+") as f:
            f.write(output)
            
openai_ = LLMProvider("openai")
deepseek = LLMProvider("deepseek")