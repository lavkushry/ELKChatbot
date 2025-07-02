# 🤖 ELK Chatbot API Documentation

## Overview

The ELK Chatbot API provides a comprehensive interface for converting natural language queries into Elasticsearch DSL (Domain Specific Language) and executing them against your Elasticsearch cluster. The API is built with FastAPI and includes interactive documentation through Swagger UI.

## 🚀 Getting Started

### Base URL
```
http://localhost:8001
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **OpenAPI JSON**: http://localhost:8001/openapi.json

## 📋 API Endpoints

### Health & Status

#### `GET /` - System Health Check
Check the overall health and status of all system components.

**Response:**
```json
{
  "elasticsearch": true,
  "mistral_ai": true,
  "overall_status": "healthy",
  "timestamp": "2025-07-02T14:21:06.944681"
}
```

**Status Codes:**
- `200` - System is healthy
- `503` - System is degraded (some components unavailable)

---

### Index Management

#### `GET /indices` - List Available Indices
Retrieve all available Elasticsearch indices with metadata.

**Response:**
```json
[
  {
    "name": "kibana_sample_data_ecommerce",
    "health": "green",
    "status": "open",
    "doc_count": 4675,
    "store_size": "4.8mb",
    "primary_shards": 1
  }
]
```

**Features:**
- Automatically filters system indices (starting with '.')
- Includes health status and document counts
- Shows storage size and shard information

#### `GET /indices/{index}/mapping` - Get Index Field Mapping
Retrieve detailed field mappings for a specific index.

**Parameters:**
- `index` (path) - Name of the Elasticsearch index

**Response:**
```json
{
  "index": "kibana_sample_data_ecommerce",
  "mapping": {
    "properties": {
      "customer_full_name": {
        "type": "text",
        "fields": {
          "keyword": {
            "type": "keyword"
          }
        }
      },
      "taxful_total_price": {
        "type": "float"
      }
    }
  }
}
```

---

### Query Processing

#### `POST /convert` - Convert Natural Language to DSL
Convert natural language query to Elasticsearch DSL without execution.

**Request Body:**
```json
{
  "index": "kibana_sample_data_ecommerce",
  "query": "Find customers named Mary with orders over $100"
}
```

**Response:**
```json
{
  "success": true,
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
  "validation_errors": null
}
```

**Use Cases:**
- Understanding query translation logic
- Testing queries before execution
- Building and debugging complex queries

#### `POST /search` - Execute Natural Language Query
Complete query processing pipeline: convert natural language to DSL and execute.

**Request Body:**
```json
{
  "index": "kibana_sample_data_ecommerce",
  "query": "Show me orders from France with total price greater than 100 euros",
  "max_results": 10
}
```

**Response:**
```json
{
  "success": true,
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
```

**Query Processing Steps:**
1. Validates target index exists
2. Retrieves index field mappings for context
3. Converts natural language to Elasticsearch DSL
4. Executes query against Elasticsearch
5. Returns results with metadata

#### `POST /validate` - Validate Elasticsearch DSL
Validate a DSL query against an index without executing it.

**Request Body:**
```json
{
  "index": "kibana_sample_data_ecommerce",
  "dsl_query": {
    "query": {
      "match": {"customer_name": "Mary"}
    }
  }
}
```

**Response:**
```json
{
  "valid": true,
  "error": null,
  "query": {
    "query": {
      "match": {"customer_name": "Mary"}
    }
  }
}
```

---

## 📝 Query Examples

### Basic Queries
```
"Find customers named John"
"Show products with price over 50"
"Get orders from last month"
```

### Advanced Queries
```
"Show me orders from France with total price greater than 100 euros"
"Find customers who bought products containing 'shirt' last month"
"Get top 5 most expensive orders from Germany"
"Show sales data for products in electronics category from 2023"
```

### Best Practices for Natural Language Queries

1. **Be Specific with Field Names**
   - ✅ "customer_full_name contains Mary"
   - ❌ "customer contains Mary"

2. **Include Units and Context**
   - ✅ "price greater than 100 euros"
   - ❌ "price > 100"

3. **Specify Date Ranges Clearly**
   - ✅ "orders from last 30 days"
   - ❌ "recent orders"

4. **Use Exact Field Values When Known**
   - ✅ "country_iso_code is FR"
   - ❌ "from France"

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ES_HOST` | Elasticsearch hostname | `localhost` | Yes |
| `ES_PORT` | Elasticsearch port | `9200` | Yes |
| `ES_USERNAME` | Elasticsearch username | - | Yes |
| `ES_PASSWORD` | Elasticsearch password | - | Yes |
| `ES_CA_CERT` | Path to CA certificate | - | No |

### Query Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_results` | Maximum results to return | `10` | `1-1000` |

---

## 🚨 Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `404` | Index not found | Verify index name exists |
| `400` | Invalid query | Check DSL syntax and field names |
| `500` | Mistral AI service error | Verify Ollama is running |
| `503` | Elasticsearch unavailable | Check ES connection and credentials |

### Error Response Format
```json
{
  "success": false,
  "error": "Index 'invalid_index' not found",
  "error_code": "INDEX_NOT_FOUND",
  "details": "The specified index does not exist in the cluster",
  "suggestions": [
    "Check the index name spelling",
    "Use GET /indices to see available indices"
  ]
}
```

---

## 🧪 Testing with cURL

### Health Check
```bash
curl -X GET "http://localhost:8001/" \
  -H "accept: application/json"
```

### List Indices
```bash
curl -X GET "http://localhost:8001/indices" \
  -H "accept: application/json"
```

### Convert Query
```bash
curl -X POST "http://localhost:8001/convert" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "index": "kibana_sample_data_ecommerce",
    "query": "Find customers from France with high-value orders"
  }'
```

### Execute Search
```bash
curl -X POST "http://localhost:8001/search" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "index": "kibana_sample_data_ecommerce",
    "query": "Show me top 5 customers by total spent",
    "max_results": 5
  }'
```

---

## 📊 Response Models

### QueryResponse
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Execution status |
| `dsl_query` | object | Generated Elasticsearch DSL |
| `total_hits` | integer | Total matching documents |
| `results` | array | Array of result documents |
| `execution_time_ms` | float | API processing time |
| `took_ms` | integer | Elasticsearch execution time |

### IndexInfo
| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Index name |
| `health` | string | Health status (green/yellow/red) |
| `status` | string | Index status (open/close) |
| `doc_count` | integer | Number of documents |
| `store_size` | string | Storage size |

---

## 🔍 Advanced Features

### Query Optimization
- Automatic field mapping analysis
- Smart error correction for common mistakes
- Query validation before execution
- Performance metrics tracking

### Security
- Environment-based configuration
- Secure credential management
- Connection verification
- Input validation and sanitization

### Monitoring
- Health status endpoints
- Performance metrics
- Error tracking and logging
- Circuit breaker patterns

---

## 🤝 Support

For issues, questions, or feature requests, please refer to:
- Interactive API documentation at `/docs`
- Health status endpoint for system diagnostics
- Application logs for detailed error information

## 📚 Additional Resources

- [Elasticsearch Query DSL Documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenAPI Specification](https://swagger.io/specification/)
