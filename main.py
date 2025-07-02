"""
FastAPI backend for ELK Chatbot
Converts natural language queries to Elasticsearch DSL using Mistral via Ollama
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import logging

from es_utils import ElasticsearchClient
from mistral_client import MistralClient
from prompt_builder import PromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with comprehensive OpenAPI documentation
app = FastAPI(
    title="🤖 ELK Chatbot API",
    description="""
    ## Natural Language to Elasticsearch DSL Converter
    
    Convert plain English queries into valid Elasticsearch DSL queries using Mistral AI.
    
    ### Features:
    - 🔍 **Natural Language Processing**: Convert English queries to Elasticsearch DSL
    - 🛠️ **Smart Error Correction**: Automatic fixing of common query mistakes
    - ✅ **Real-time Validation**: Query validation before execution
    - 📊 **Multiple Index Support**: Works with any Elasticsearch index
    - 🔧 **RESTful API**: Comprehensive endpoints for all operations
    
    ### Usage Flow:
    1. **Check System Health**: Use `/health` to verify all services are running
    2. **List Indices**: Use `/indices` to see available Elasticsearch indices
    3. **Get Field Mappings**: Use `/indices/{index}/mapping` to understand index structure
    4. **Convert Queries**: Use `/query` to convert natural language to DSL
    5. **Execute Queries**: Use `/search` to run queries and get results
    
    ### Authentication:
    Requires valid Elasticsearch credentials configured via environment variables.
    """,
    version="1.0.0",
    contact={
        "name": "ELK Chatbot Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "System health and status endpoints"
        },
        {
            "name": "indices",
            "description": "Elasticsearch index management and discovery"
        },
        {
            "name": "query",
            "description": "Natural language to DSL conversion"
        },
        {
            "name": "search",
            "description": "Query execution and result retrieval"
        }
    ]
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

# Enhanced Pydantic models with comprehensive documentation
class QueryRequest(BaseModel):
    """Request model for natural language to DSL conversion"""
    index: str = Field(
        ...,
        description="Target Elasticsearch index name",
        example="kibana_sample_data_ecommerce"
    )
    query: str = Field(
        ...,
        description="Natural language query to convert to Elasticsearch DSL",
        example="Show me orders from France with total price greater than 100 euros"
    )
    max_results: Optional[int] = Field(
        default=10,
        description="Maximum number of results to return (1-1000)",
        ge=1,
        le=1000,
        example=10
    )

class DSLConversionRequest(BaseModel):
    """Request model for DSL conversion only (no execution)"""
    index: str = Field(
        ...,
        description="Target Elasticsearch index name for context",
        example="kibana_sample_data_ecommerce"
    )
    query: str = Field(
        ...,
        description="Natural language query to convert",
        example="Find products with price between 50 and 200"
    )

class QueryResponse(BaseModel):
    """Complete response model for executed queries"""
    success: bool = Field(description="Whether the query executed successfully")
    dsl_query: Dict[str, Any] = Field(description="Generated Elasticsearch DSL query")
    total_hits: int = Field(description="Total number of matching documents")
    results: List[Dict[str, Any]] = Field(description="Array of matching documents")
    execution_time_ms: float = Field(description="Query execution time in milliseconds")
    took_ms: Optional[int] = Field(description="Elasticsearch execution time", default=None)

class DSLResponse(BaseModel):
    """Response model for DSL conversion only"""
    success: bool = Field(description="Whether the conversion was successful")
    dsl_query: Dict[str, Any] = Field(description="Generated Elasticsearch DSL query")
    validation_errors: Optional[List[str]] = Field(description="Any validation warnings", default=None)

class IndexInfo(BaseModel):
    """Information about an Elasticsearch index"""
    name: str = Field(description="Index name")
    health: str = Field(description="Index health status (green/yellow/red)")
    status: str = Field(description="Index status (open/close)")
    doc_count: int = Field(description="Number of documents in the index")
    store_size: str = Field(description="Storage size of the index")
    primary_shards: Optional[int] = Field(description="Number of primary shards", default=None)

class FieldMapping(BaseModel):
    """Field mapping information"""
    field_name: str = Field(description="Name of the field")
    field_type: str = Field(description="Elasticsearch field type")
    searchable: bool = Field(description="Whether the field is searchable")
    aggregatable: bool = Field(description="Whether the field supports aggregations")

class HealthStatus(BaseModel):
    """System health status"""
    elasticsearch: bool = Field(description="Elasticsearch connection status")
    mistral_ai: bool = Field(description="Mistral AI service status")
    overall_status: str = Field(description="Overall system status")
    timestamp: str = Field(description="Health check timestamp")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(description="Error code for programmatic handling", default=None)
    details: Optional[str] = Field(description="Additional error details", default=None)
    suggestions: Optional[List[str]] = Field(description="Suggested solutions", default=None)

# API Endpoints with comprehensive documentation

@app.get(
    "/",
    tags=["health"],
    summary="System Health Check",
    description="Check the overall health and status of all system components",
    response_model=HealthStatus
)
async def root():
    """
    **System Health Check**
    
    Returns the current status of all system components:
    - Elasticsearch connection
    - Mistral AI service availability
    - Overall system health
    
    Use this endpoint to verify that the API is running and all dependencies are available.
    """
    from datetime import datetime
    
    es_status = es_client.is_connected()
    mistral_status = mistral_client.is_available()
    
    return HealthStatus(
        elasticsearch=es_status,
        mistral_ai=mistral_status,
        overall_status="healthy" if es_status and mistral_status else "degraded",
        timestamp=datetime.utcnow().isoformat()
    )

@app.get(
    "/indices",
    tags=["indices"],
    summary="List Available Indices",
    description="Retrieve all available Elasticsearch indices with metadata",
    response_model=List[IndexInfo],
    responses={
        200: {
            "description": "Successfully retrieved indices",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "kibana_sample_data_ecommerce",
                            "health": "green",
                            "status": "open",
                            "doc_count": 4675,
                            "store_size": "4.8mb",
                            "primary_shards": 1
                        }
                    ]
                }
            }
        },
        500: {"description": "Failed to connect to Elasticsearch"}
    }
)
async def get_indices():
    """
    **List Available Indices**
    
    Returns a list of all available Elasticsearch indices with their metadata including:
    - Index name and health status
    - Document count and storage size
    - Primary shard information
    
    System indices (starting with '.') are automatically filtered out.
    """
    try:
        indices = es_client.get_indices()
        return indices
    except Exception as e:
        logger.error(f"Error fetching indices: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch indices: {str(e)}"
        )

@app.get(
    "/indices/{index}/mapping",
    tags=["indices"],
    summary="Get Index Field Mapping",
    description="Retrieve field mappings and structure for a specific index",
    responses={
        200: {
            "description": "Successfully retrieved index mapping",
            "content": {
                "application/json": {
                    "example": {
                        "index": "kibana_sample_data_ecommerce",
                        "fields": [
                            {
                                "field_name": "customer_full_name",
                                "field_type": "text",
                                "searchable": True,
                                "aggregatable": False
                            },
                            {
                                "field_name": "taxful_total_price",
                                "field_type": "float",
                                "searchable": True,
                                "aggregatable": True
                            }
                        ]
                    }
                }
            }
        },
        404: {"description": "Index not found"}
    }
)
async def get_mapping(
    index: str = Field(
        ...,
        description="Name of the Elasticsearch index",
        example="kibana_sample_data_ecommerce"
    )
):
    """
    **Get Index Field Mapping**
    
    Returns detailed field mapping information for the specified index including:
    - Field names and data types
    - Searchability and aggregation capabilities
    - Nested object structures
    
    This information is essential for understanding what fields are available
    for querying and how to construct effective natural language queries.
    """
    try:
        mapping = es_client.get_mapping(index)
        return {"index": index, "mapping": mapping}
    except Exception as e:
        logger.error(f"Error fetching mapping for {index}: {e}")
        raise HTTPException(
            status_code=404, 
            detail=f"Index '{index}' not found or mapping unavailable"
        )

@app.post(
    "/convert",
    tags=["query"],
    summary="Convert Natural Language to DSL",
    description="Convert natural language query to Elasticsearch DSL without execution",
    response_model=DSLResponse,
    responses={
        200: {
            "description": "Successfully converted query to DSL",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "dsl_query": {
                            "query": {
                                "bool": {
                                    "must": [
                                        {"match": {"customer_full_name": "Mary"}},
                                        {"range": {"taxful_total_price": {"gte": 100}}}
                                    ]
                                }
                            }
                        },
                        "validation_errors": None
                    }
                }
            }
        },
        400: {"description": "Invalid request or query conversion failed"},
        404: {"description": "Index not found"}
    }
)
async def convert_to_dsl(request: DSLConversionRequest):
    """
    **Convert Natural Language to Elasticsearch DSL**
    
    Converts a natural language query into valid Elasticsearch DSL without executing it.
    This endpoint is useful for:
    - Understanding how queries are translated
    - Testing query logic before execution
    - Building and debugging complex queries
    
    **Example queries:**
    - "Find customers named Mary with orders over $100"
    - "Show products from France sold in 2023"
    - "Get orders with price between 50 and 200 euros"
    """
    try:
        logger.info(f"Converting query: '{request.query}' for index: '{request.index}'")
        
        # Validate index exists and get mapping
        try:
            mapping = es_client.get_mapping(request.index)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Index '{request.index}' not found")
        
        # Build prompt and generate DSL
        prompt = prompt_builder.build_prompt(request.query, mapping)
        dsl_response = mistral_client.generate_dsl(prompt)
        dsl_query = prompt_builder.clean_and_validate_dsl(dsl_response)
        
        return DSLResponse(
            success=True,
            dsl_query=dsl_query,
            validation_errors=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DSL conversion failed: {e}")
        return DSLResponse(
            success=False,
            dsl_query={},
            validation_errors=[str(e)]
        )

@app.post(
    "/search",
    tags=["search"],
    summary="Execute Natural Language Query",
    description="Convert natural language to DSL and execute against Elasticsearch",
    response_model=QueryResponse,
    responses={
        200: {
            "description": "Query executed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "dsl_query": {
                            "query": {
                                "bool": {
                                    "must": [
                                        {"term": {"geoip.country_iso_code": "FR"}},
                                        {"range": {"taxful_total_price": {"gte": 100}}}
                                    ]
                                }
                            }
                        },
                        "total_hits": 156,
                        "results": [
                            {
                                "_source": {
                                    "customer_full_name": "Mary Bailey",
                                    "taxful_total_price": 174.0,
                                    "geoip": {"country_iso_code": "FR"}
                                }
                            }
                        ],
                        "execution_time_ms": 23.5,
                        "took_ms": 12
                    }
                }
            }
        },
        400: {"description": "Invalid query or execution failed"},
        404: {"description": "Index not found"},
        500: {"description": "Internal server error"}
    }
)
async def execute_query(request: QueryRequest):
    """
    **Execute Natural Language Query**
    
    Complete query processing pipeline:
    1. Validates the target index exists
    2. Retrieves index field mappings for context
    3. Converts natural language to Elasticsearch DSL
    4. Executes the query against Elasticsearch
    5. Returns results with metadata
    
    **Query Examples:**
    - `"Show me orders from France with total price greater than 100 euros"`
    - `"Find customers who bought products containing 'shirt' last month"`
    - `"Get top 5 most expensive orders from Germany"`
    
    **Tips for better results:**
    - Use specific field names when known (e.g., "customer_full_name" instead of "customer")
    - Include units for numbers (e.g., "100 euros", "last 30 days")
    - Be specific about date ranges and filters
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
            success=True,
            dsl_query=dsl_query,
            total_hits=results["total_hits"],
            results=results["hits"],
            execution_time_ms=results["execution_time_ms"],
            took_ms=results.get("took", None)
        )
        
        logger.info(f"Query completed successfully. Found {results['total_hits']} hits")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post(
    "/validate",
    tags=["query"],
    summary="Validate Elasticsearch DSL",
    description="Validate a DSL query against an index without executing it",
    responses={
        200: {"description": "Validation completed"},
        404: {"description": "Index not found"},
        500: {"description": "Validation service error"}
    }
)
async def validate_dsl(
    index: str = Field(..., description="Target index name"),
    dsl_query: Dict[str, Any] = Field(..., description="Elasticsearch DSL query to validate")
):
    """
    **Validate Elasticsearch DSL Query**
    
    Validates a DSL query against the specified index without executing it.
    Useful for:
    - Testing custom DSL queries
    - Debugging query syntax issues
    - Verifying field names and types
    
    Returns validation status and any error messages.
    """
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
