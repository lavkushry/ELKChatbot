"""
FastAPI backend for ELK Chatbot
Converts natural language queries to Elasticsearch DSL using Mistral via Ollama
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import logging

from es_utils import ElasticsearchClient
from mistral_client import MistralClient
from prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ELK Chatbot API",
    description="Convert natural language queries to Elasticsearch DSL using Mistral AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
es_client = ElasticsearchClient()
mistral_client = MistralClient()
prompt_builder = PromptBuilder()

# Pydantic models
class QueryRequest(BaseModel):
    index: str
    query: str
    max_results: Optional[int] = 10

class QueryResponse(BaseModel):
    dsl_query: Dict[str, Any]
    total_hits: int
    results: List[Dict[str, Any]]
    execution_time_ms: float

class IndexInfo(BaseModel):
    name: str
    doc_count: int
    store_size: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "ELK Chatbot API is running",
        "status": "healthy",
        "elasticsearch": es_client.is_connected(),
        "mistral": mistral_client.is_available()
    }

@app.get("/indices", response_model=List[IndexInfo])
async def get_indices():
    """Get list of available Elasticsearch indices"""
    try:
        indices = es_client.get_indices()
        return indices
    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch indices: {str(e)}")

@app.get("/mapping/{index}")
async def get_mapping(index: str):
    """Get mapping for a specific index"""
    try:
        mapping = es_client.get_mapping(index)
        return {"index": index, "mapping": mapping}
    except Exception as e:
        logger.error(f"Error fetching mapping for {index}: {e}")
        raise HTTPException(status_code=404, detail=f"Index '{index}' not found or mapping unavailable")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Main endpoint: Convert natural language query to Elasticsearch DSL and execute
    """
    try:
        logger.info(f"Processing query: '{request.query}' for index: '{request.index}'")
        
        # Step 1: Validate index exists and get mapping
        try:
            mapping = es_client.get_mapping(request.index)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Index '{request.index}' not found")
        
        # Step 2: Build prompt for Mistral
        prompt = prompt_builder.build_prompt(request.query, mapping)
        
        # Step 3: Get DSL from Mistral
        try:
            dsl_response = mistral_client.generate_dsl(prompt)
            dsl_query = prompt_builder.clean_and_validate_dsl(dsl_response)
        except Exception as e:
            logger.error(f"Mistral generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate DSL: {str(e)}")
        
        # Step 4: Execute query on Elasticsearch
        try:
            results = es_client.execute_query(request.index, dsl_query, max_results=request.max_results)
        except Exception as e:
            logger.error(f"Elasticsearch query failed: {e}")
            raise HTTPException(status_code=400, detail=f"Query execution failed: {str(e)}")
        
        # Step 5: Format response
        response = QueryResponse(
            dsl_query=dsl_query,
            total_hits=results["total_hits"],
            results=results["hits"],
            execution_time_ms=results["execution_time_ms"]
        )
        
        logger.info(f"Query completed successfully. Found {results['total_hits']} hits")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/validate-dsl")
async def validate_dsl(index: str, dsl_query: Dict[str, Any]):
    """Validate a DSL query without executing it"""
    try:
        is_valid, error = es_client.validate_query(index, dsl_query)
        return {
            "valid": is_valid,
            "error": error,
            "query": dsl_query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
