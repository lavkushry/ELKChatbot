# ELK Chatbot - Complete Knowledge Base

## 🎯 Project Overview

The ELK Chatbot is an intelligent natural language to Elasticsearch DSL (Domain Specific Language) converter that allows users to query Elasticsearch indices using plain English. The system leverages Mistral AI (via Ollama) to understand natural language queries and convert them into proper Elasticsearch DSL queries.

### Key Features
- **Natural Language Processing**: Convert plain English queries to Elasticsearch DSL
- **Smart Error Correction**: Automatic fixing of common LLM-generated query mistakes
- **Real-time Validation**: Query validation before execution
- **Multiple Index Support**: Works with any Elasticsearch index
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Robust Error Handling**: Intelligent retry mechanisms and error recovery

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │───▶│  FastAPI Server │───▶│ Elasticsearch   │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Mistral AI      │
                       │ (via Ollama)    │
                       └─────────────────┘
```

### Core Modules

1. **main.py** - FastAPI server with REST endpoints
2. **es_utils.py** - Elasticsearch client and utilities
3. **mistral_client.py** - Mistral AI integration (to be created)
4. **prompt_builder.py** - Prompt engineering and DSL validation (to be created)
5. **prototype.py** - Development prototype with smart correction features

## 🚀 Implementation Approach

### Phase 1: Foundation Setup (Week 1)

#### 1.1 Environment Preparation
**Project Structure Design:**
- Create modular directory structure separating source code, tests, configuration, logs, and documentation
- Set up isolated Python virtual environment for dependency management
- Define core dependencies for web framework, search engine client, AI integration, and testing

**Key Dependencies:**
- **Web Framework**: Modern async-capable API framework
- **Search Engine Client**: Official Elasticsearch client library
- **AI Integration**: HTTP client for LLM communication
- **Configuration Management**: Environment variable and settings management
- **Testing Framework**: Async-compatible testing tools
- **Development Tools**: Code formatting, linting, and development servers

#### 1.2 Core Configuration Strategy
**Configuration Architecture:**
- Environment-based configuration management using structured settings
- Separation of development, testing, and production configurations
- Secure credential management through environment variables
- Validation of configuration parameters at application startup

**Configuration Categories:**

**Search Engine Configuration:**
- Connection parameters (host, port, authentication)
- Security settings (certificates, verification options)
- Connection pooling and timeout configurations

**AI Model Configuration:**
- Model service endpoints and authentication
- Model-specific parameters (temperature, context window, prediction limits)
- Retry policies and fallback mechanisms

**API Server Configuration:**
- Network binding settings (host, port)
- Development vs production modes
- Middleware and CORS policies

**Query Processing Configuration:**
- Default result limits and pagination settings
- Query timeout and performance thresholds
- Caching and optimization parameters

### Phase 2: Core Module Development (Week 2-3)

#### 2.1 Search Engine Client Architecture
**Client Design Principles:**
- **Connection Management**: Implement robust connection pooling with automatic reconnection
- **Authentication Strategy**: Support multiple authentication methods (basic auth, API keys, certificates)
- **Error Handling**: Comprehensive exception handling with meaningful error messages
- **Performance Optimization**: Implement request timeouts, connection reuse, and response caching

**Core Functionality Requirements:**

**Index Management:**
- Retrieve available indices with metadata (health status, document count, size)
- Filter system indices from user-visible results
- Handle index access permissions and availability checks

**Schema Discovery:**
- Extract field mappings from index configurations
- Build searchable field inventories for query construction
- Handle nested object structures and complex field types

**Query Execution:**
- Execute DSL queries with configurable result limits
- Transform search results into consistent response formats
- Implement query validation before execution
- Provide detailed error reporting for failed queries

**Health Monitoring:**
- Connection status monitoring and reporting
- Performance metrics collection (response times, success rates)
- Circuit breaker pattern for handling service unavailability

#### 2.2 Mistral AI Integration Module
```python
# src/mistral_client.py
import requests
import json
import time
import re
from typing import Dict, Any, Tuple
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class MistralClient:
    def __init__(self):
        self.base_url = f"http://{settings.ollama_host}:{settings.ollama_port}"
        self.model = settings.mistral_model
        
    def check_availability(self) -> bool:
        """Check if Ollama service is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_dsl(self, prompt: str, max_retries: int = 3) -> str:
        """Generate DSL from natural language using Mistral"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                            "num_ctx": 4096,
                            "num_predict": 2048
                        }
                    },
                    timeout=settings.query_timeout * 2
                )
                
                if response.status_code == 200:
                    return self._clean_response(response.json()["response"])
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Error generating DSL: {e}")
                
        raise Exception("Failed to generate DSL after all retries")
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and extract JSON from Mistral response"""
        # Remove markdown code blocks
        match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Extract JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
            
        return response_text.strip()

# Global client instance
mistral_client = MistralClient()
```

#### 2.3 Prompt Engineering Module
```python
# src/prompt_builder.py
import json
from typing import Dict, Any
from src.es_utils import es_client

class PromptBuilder:
    def __init__(self):
        self.base_template = self._load_base_template()
        
    def _load_base_template(self) -> str:
        return """You are an expert Elasticsearch DSL generator. Convert natural language queries into valid Elasticsearch DSL.

CRITICAL GUIDELINES:
1. Use exact field names from the provided mapping
2. Use lowercase 'y' for years in date math (not 'Y')
3. Don't quote numeric values in range queries
4. Return only valid JSON without markdown formatting
5. Use appropriate query types: term, match, range, bool

INDEX MAPPING:
{mapping}

AVAILABLE FIELDS:
{fields}

COMMON FIELD MAPPINGS FOR ECOMMERCE:
- "price" → "taxful_total_price"
- "country" → "geoip.country_iso_code"  
- "customer" → "customer_full_name"
- "product" → "products.product_name"

NATURAL LANGUAGE QUERY:
"{query}"

Generate only the Elasticsearch DSL JSON:"""

    async def build_prompt(self, nl_query: str, index: str) -> str:
        """Build complete prompt with mapping context"""
        try:
            mapping = await es_client.get_mapping(index)
            fields = self._extract_fields(mapping)
            
            return self.base_template.format(
                mapping=json.dumps(mapping, indent=2),
                fields=", ".join(sorted(fields)),
                query=nl_query
            )
        except Exception as e:
            raise ValueError(f"Error building prompt: {e}")
    
    def _extract_fields(self, mapping: Dict[str, Any]) -> set:
        """Extract all available fields from mapping"""
        fields = set()
        
        def extract_recursive(obj: Dict, prefix: str = ""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'properties':
                        extract_recursive(value, prefix)
                    elif isinstance(value, dict) and 'type' in value:
                        field_name = f"{prefix}.{key}" if prefix else key
                        fields.add(field_name)
                    elif isinstance(value, dict):
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        extract_recursive(value, new_prefix)
        
        for index_name, index_mapping in mapping.items():
            extract_recursive(index_mapping.get('mappings', {}))
            
        return fields

# Global instance
prompt_builder = PromptBuilder()
```

### Phase 3: DSL Correction System (Week 4)

#### 3.1 Enhanced DSL Corrector
```python
# src/dsl_corrector.py
import json
import re
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DSLCorrector:
    def __init__(self, index_mapping: Dict[str, Any]):
        self.mapping = index_mapping
        self.available_fields = self._extract_fields()
        self.correction_rules = self._load_correction_rules()
    
    def _extract_fields(self) -> set:
        """Extract all fields from mapping"""
        # Implementation similar to prompt_builder
        pass
    
    def _load_correction_rules(self) -> Dict[str, Dict]:
        """Load field correction rules"""
        return {
            'ecommerce': {
                '"country"': '"geoip.country_iso_code"',
                '"price"': '"taxful_total_price"',
                '"customer_name"': '"customer_full_name"',
                '"product"': '"products.product_name"',
                '"quantity"': '"total_quantity"',
                '"date"': '"order_date"'
            },
            'logs': {
                '"timestamp"': '"@timestamp"',
                '"ip"': '"clientip"',
                '"status"': '"response"'
            }
        }
    
    def fix_all_issues(self, dsl_query: str) -> Tuple[str, List[str]]:
        """Apply comprehensive fixes"""
        fixes_applied = []
        
        # 1. Fix date math
        dsl_query, date_fixes = self._fix_date_math(dsl_query)
        fixes_applied.extend(date_fixes)
        
        # 2. Fix field names
        dsl_query, field_fixes = self._fix_field_names(dsl_query)
        fixes_applied.extend(field_fixes)
        
        # 3. Fix numeric ranges
        dsl_query, range_fixes = self._fix_range_queries(dsl_query)
        fixes_applied.extend(range_fixes)
        
        # 4. Fix JSON structure
        dsl_query, structure_fixes = self._fix_json_structure(dsl_query)
        fixes_applied.extend(structure_fixes)
        
        # 5. Validate final query
        validation_errors = self._validate_query(dsl_query)
        
        return dsl_query, fixes_applied, validation_errors
    
    def _fix_date_math(self, dsl_query: str) -> Tuple[str, List[str]]:
        """Fix date math expressions"""
        fixes = []
        
        # Fix uppercase date units
        patterns = [
            (r'now-([0-9]+)Y', r'now-\1y', 'Year unit: Y → y'),
            (r'now-([0-9]+)M', r'now-\1M', 'Month unit verified'),
            (r'now-([0-9]+)D', r'now-\1d', 'Day unit: D → d'),
            (r'now-([0-9]+)H', r'now-\1h', 'Hour unit: H → h'),
        ]
        
        for pattern, replacement, description in patterns:
            if re.search(pattern, dsl_query):
                dsl_query = re.sub(pattern, replacement, dsl_query)
                fixes.append(f"Fixed date math: {description}")
        
        return dsl_query, fixes
    
    # Additional methods for other fixes...
```

### Phase 4: FastAPI Application (Week 5)

#### 4.1 Main Application Structure
```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from contextlib import asynccontextmanager

from src.es_utils import es_client
from src.mistral_client import mistral_client
from src.prompt_builder import prompt_builder
from src.dsl_corrector import DSLCorrector
from config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting ELK Chatbot API...")
    
    # Verify connections
    if not mistral_client.check_availability():
        logger.warning("Mistral/Ollama not available")
    
    try:
        await es_client.get_indices()
        logger.info("Elasticsearch connection verified")
    except Exception as e:
        logger.error(f"Elasticsearch connection failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ELK Chatbot API...")

app = FastAPI(
    title="ELK Chatbot API",
    description="Natural Language to Elasticsearch DSL Converter",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    index: str
    query: str
    max_results: Optional[int] = 10

class QueryResponse(BaseModel):
    dsl_query: Dict[str, Any]
    total_hits: int
    results: List[Dict[str, Any]]
    execution_time_ms: float
    fixes_applied: List[str] = []

# API Endpoints
@app.get("/")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "elasticsearch": "connected" if await _check_es_health() else "disconnected",
        "mistral": "connected" if mistral_client.check_availability() else "disconnected"
    }

@app.get("/indices")
async def list_indices():
    """Get all available indices"""
    return await es_client.get_indices()

@app.get("/mapping/{index}")
async def get_mapping(index: str):
    """Get mapping for specific index"""
    return await es_client.get_mapping(index)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query"""
    import time
    
    start_time = time.time()
    
    try:
        # Build prompt
        prompt = await prompt_builder.build_prompt(request.query, request.index)
        
        # Generate DSL
        raw_dsl = mistral_client.generate_dsl(prompt)
        
        # Apply corrections
        mapping = await es_client.get_mapping(request.index)
        corrector = DSLCorrector(mapping)
        corrected_dsl, fixes_applied, validation_errors = corrector.fix_all_issues(raw_dsl)
        
        if validation_errors:
            raise HTTPException(status_code=400, detail={
                "error": "Query validation failed",
                "details": validation_errors
            })
        
        # Execute query
        dsl_dict = json.loads(corrected_dsl)
        results = await es_client.execute_query(
            request.index, 
            dsl_dict, 
            request.max_results
        )
        
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            dsl_query=dsl_dict,
            total_hits=results['hits']['total']['value'],
            results=[hit['_source'] for hit in results['hits']['hits']],
            execution_time_ms=execution_time,
            fixes_applied=fixes_applied
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _check_es_health() -> bool:
    """Check Elasticsearch health"""
    try:
        await es_client.get_indices()
        return True
    except:
        return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
```

### Phase 5: Testing & Quality Assurance (Week 6)

#### 5.1 Test Suite Implementation
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

@pytest.fixture
def sample_query_request():
    return {
        "index": "kibana_sample_data_ecommerce",
        "query": "Show me orders from France",
        "max_results": 5
    }

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()

def test_list_indices():
    response = client.get("/indices")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_query_processing(sample_query_request):
    response = client.post("/query", json=sample_query_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "dsl_query" in data
    assert "total_hits" in data
    assert "results" in data
    assert "execution_time_ms" in data

# tests/test_dsl_corrector.py
from src.dsl_corrector import DSLCorrector

def test_date_math_correction():
    sample_mapping = {}  # Mock mapping
    corrector = DSLCorrector(sample_mapping)
    
    query_with_error = '{"query": {"range": {"date": {"gte": "now-1Y"}}}}'
    corrected, fixes, errors = corrector.fix_all_issues(query_with_error)
    
    assert "now-1y" in corrected
    assert len(fixes) > 0

def test_field_name_correction():
    # Test field name corrections
    pass
```

### Phase 6: Deployment & Documentation (Week 7)

#### 6.1 Production Deployment
```bash
# deployment/docker-compose.yml
version: '3.8'
services:
  elk-chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ES_HOST=${ES_HOST}
      - ES_USERNAME=${ES_USERNAME}
      - ES_PASSWORD=${ES_PASSWORD}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped

# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

#### 6.2 API Documentation
```python
# Add to main.py for auto-generated docs
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_favicon_url="/static/favicon.ico",
    )
```

### Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | Week 1 | Environment setup, configuration |
| Phase 2 | Week 2-3 | Core modules (ES client, Mistral integration) |
| Phase 3 | Week 4 | DSL correction system |
| Phase 4 | Week 5 | FastAPI application |
| Phase 5 | Week 6 | Testing suite |
| Phase 6 | Week 7 | Deployment & documentation |

### Success Metrics

1. **Functional Requirements**
   - 95% successful DSL generation from natural language
   - Average response time < 3 seconds
   - Support for 10+ query types

2. **Technical Requirements**
   - 99.9% API uptime
   - Proper error handling and logging
   - Comprehensive test coverage (>80%)

3. **User Experience**
   - Intuitive API interface
   - Clear error messages
   - Helpful query suggestions

### Risk Mitigation

1. **Mistral/Ollama Availability**
   - Implement health checks
   - Add fallback mechanisms
   - Cache frequent queries

2. **Elasticsearch Connectivity**
   - Connection pooling
   - Retry mechanisms
   - Circuit breaker pattern

3. **Query Accuracy**
   - Extensive testing with sample data
   - User feedback collection
   - Continuous prompt improvement

## 🔧 Technical Stack

### Backend Technologies
- **FastAPI**: Modern, fast web framework for building APIs
- **Elasticsearch**: Search and analytics engine
- **Mistral AI**: Large Language Model for query interpretation
- **Ollama**: Local LLM runtime environment
- **Python 3.8+**: Primary programming language

### Key Libraries
```python
fastapi>=0.104.0
elasticsearch>=8.0.0
requests>=2.31.0
python-dotenv>=1.0.0
uvicorn>=0.24.0
pydantic>=2.0.0
```


