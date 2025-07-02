"""
Mistral client for generating Elasticsearch DSL via Ollama
Handles communication with local Ollama instance
"""

import requests
import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class MistralClient:
    """Client for interacting with Mistral via Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "mistral"
        self.generation_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": 2048
        }
    
    def is_available(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_dsl(self, prompt: str, max_retries: int = 3) -> str:
        """Generate Elasticsearch DSL from natural language prompt"""
        
        if not self.is_available():
            raise ConnectionError("Ollama service is not available. Please ensure it's running.")
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Generating DSL - Attempt {attempt}/{max_retries}")
                
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": self.generation_config
                    },
                    timeout=300  # 5 minutes timeout
                )
                
                elapsed_time = time.time() - start_time
                logger.info(f"DSL generation completed in {elapsed_time:.1f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"HTTP Error {response.status_code}: {response.text}")
                    if attempt == max_retries:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt}")
                if attempt == max_retries:
                    raise TimeoutError("Mistral generation timed out after multiple attempts")
                
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Connection error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise ConnectionError(f"Failed to connect to Ollama: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt}: {e}")
                if attempt == max_retries:
                    raise
            
            # Wait before retry
            if attempt < max_retries:
                wait_time = min(30, attempt * 10)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        raise Exception("Failed to generate DSL after all retry attempts")
    
    def get_available_models(self) -> list:
        """Get list of available models in Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            return []
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a specific model exists in Ollama"""
        available_models = self.get_available_models()
        return any(model_name in model for model in available_models)
    
    def set_model(self, model_name: str) -> bool:
        """Set the model to use for generation"""
        if self.check_model_exists(model_name):
            self.model = model_name
            logger.info(f"Model set to: {model_name}")
            return True
        else:
            logger.error(f"Model {model_name} not found in Ollama")
            return False
    
    def update_generation_config(self, **kwargs):
        """Update generation configuration parameters"""
        self.generation_config.update(kwargs)
        logger.info(f"Updated generation config: {self.generation_config}")
    
    def test_generation(self) -> bool:
        """Test basic generation capability"""
        try:
            test_prompt = "Generate a simple Elasticsearch match_all query in JSON format."
            response = self.generate_dsl(test_prompt)
            return len(response.strip()) > 0
        except Exception as e:
            logger.error(f"Generation test failed: {e}")
            return False
