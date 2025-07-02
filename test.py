import requests

# Example: Get embedding from Ollama (Mistral)
response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={"model": "mistral", "prompt": "Sample product description"}
)
embedding = response.json()["embedding"]
