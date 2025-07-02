#!/bin/bash

# ELK Chatbot FastAPI Server Startup Script

echo "🚀 Starting ELK Chatbot API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Check if Ollama is running
echo "🤖 Checking Ollama status..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Please start it with: ollama serve"
    echo "💡 Also make sure to pull Mistral: ollama pull mistral"
    exit 1
fi

echo "✅ Ollama is running"

# Start FastAPI server
echo "🔥 Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8001
