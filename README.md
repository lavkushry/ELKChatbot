# ELK Chatbot API

A FastAPI-based backend that converts natural language queries into Elasticsearch DSL using Mistral AI via Ollama.

## Features

- 🔍 **Natural Language to DSL**: Convert plain English queries to Elasticsearch DSL
- 🤖 **AI-Powered**: Uses Mistral AI via local Ollama instance
- 🔧 **Smart Corrections**: Automatically fixes common DSL generation errors
- 📊 **Multiple Endpoints**: Query execution, index management, and validation
- ⚡ **FastAPI**: High-performance async API with automatic docs
- 🛡️ **Error Handling**: Comprehensive error handling and validation

## Project Structure

```
ELChatbot/
├── main.py               # FastAPI application
├── es_utils.py          # Elasticsearch client and utilities
├── mistral_client.py    # Mistral AI client via Ollama
├── prompt_builder.py    # Prompt engineering and DSL correction
├── .env                 # Environment configuration
├── requirements.txt     # Python dependencies
├── prototype.py         # Original prototype (reference)
└── README.md           # This file
```

## Prerequisites

1. **Elasticsearch** running locally or remotely
2. **Ollama** with Mistral model installed
3. **Python 3.8+**

### Setup Ollama & Mistral

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# Pull Mistral model
ollama pull mistral
```

### Setup Elasticsearch

Ensure Elasticsearch is running and accessible. Update `.env` with your credentials.

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/lavkushry/ELKChatbot.git
cd ELChatbot
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
# Edit .env file with your Elasticsearch credentials
```

4. **Run the application**:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Health Check
```http
GET /
```

### 2. Query Processing (Main Endpoint)
```http
POST /query
```

**Request**:
```json
{
  "index": "kibana_sample_data_ecommerce",
  "query": "Show me all orders from France in the last 30 days",
  "max_results": 10
}
```

### 3. List Indices
```http
GET /indices
```

### 4. Get Index Mapping
```http
GET /mapping/{index}
```

### 5. Validate DSL Query
```http
POST /validate-dsl?index=your_index
```

## Usage Examples

### Basic Text Search
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "index": "kibana_sample_data_ecommerce",
    "query": "Find all customers named Mary"
  }'
```

## API Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

**Built with ❤️ using FastAPI, Elasticsearch, and Mistral AI**