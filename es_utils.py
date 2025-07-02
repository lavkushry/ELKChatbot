"""
Elasticsearch utilities for the ELK Chatbot
Handles connection, querying, and mapping operations
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ElasticsearchClient:
    """Elasticsearch client with connection management and query operations"""
    
    def __init__(self):
        self.es = self._create_connection()
        self._test_connection()
    
    def _create_connection(self) -> Elasticsearch:
        """Create Elasticsearch connection using environment variables"""
        try:
            es = Elasticsearch(
                f"https://{os.getenv('ES_HOST', 'localhost')}:{os.getenv('ES_PORT', 9200)}",
                basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD')),
                ca_certs=os.getenv('ES_CA_CERT'),
                verify_certs=True,
                request_timeout=30,
                retry_on_timeout=True
            )
            return es
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch connection: {e}")
            raise
    
    def _test_connection(self):
        """Test Elasticsearch connection"""
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch: {info.body['version']['number']}")
        except Exception as e:
            logger.error(f"Elasticsearch connection test failed: {e}")
            raise ConnectionError(f"Cannot connect to Elasticsearch: {e}")
    
    def is_connected(self) -> bool:
        """Check if Elasticsearch is connected"""
        try:
            self.es.ping()
            return True
        except:
            return False
    
    def get_indices(self) -> List[Dict[str, Any]]:
        """Get list of available indices with basic stats"""
        try:
            # Get all indices
            indices_response = self.es.indices.get(index="*")
            
            # Get stats for indices
            stats_response = self.es.indices.stats(index="*")
            
            indices = []
            for index_name in indices_response.body.keys():
                if not index_name.startswith('.'):  # Skip system indices
                    stats = stats_response.body.get('indices', {}).get(index_name, {})
                    doc_count = stats.get('total', {}).get('docs', {}).get('count', 0)
                    store_size = stats.get('total', {}).get('store', {}).get('size_in_bytes', 0)
                    
                    # Convert bytes to human readable format
                    size_str = self._format_bytes(store_size)
                    
                    indices.append({
                        "name": index_name,
                        "doc_count": doc_count,
                        "store_size": size_str
                    })
            
            return sorted(indices, key=lambda x: x['name'])
            
        except Exception as e:
            logger.error(f"Error fetching indices: {e}")
            raise
    
    def get_mapping(self, index: str) -> Dict[str, Any]:
        """Get mapping for a specific index"""
        try:
            response = self.es.indices.get_mapping(index=index)
            return response.body
        except Exception as e:
            logger.error(f"Error getting mapping for {index}: {e}")
            raise
    
    def validate_query(self, index: str, dsl_query: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a DSL query without executing it"""
        try:
            self.es.indices.validate_query(index=index, body=dsl_query)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def execute_query(self, index: str, dsl_query: Dict[str, Any], max_results: int = 10) -> Dict[str, Any]:
        """Execute DSL query and return formatted results"""
        start_time = time.time()
        
        try:
            # Add size parameter to limit results
            if 'size' not in dsl_query:
                dsl_query['size'] = max_results
            
            # Execute query
            response = self.es.search(index=index, body=dsl_query)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Format results
            hits = response.body['hits']
            total_hits = hits['total']['value'] if isinstance(hits['total'], dict) else hits['total']
            
            formatted_hits = []
            for hit in hits['hits']:
                formatted_hit = {
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'source': hit['_source']
                }
                # Include highlights if available
                if 'highlight' in hit:
                    formatted_hit['highlight'] = hit['highlight']
                formatted_hits.append(formatted_hit)
            
            return {
                'total_hits': total_hits,
                'hits': formatted_hits,
                'execution_time_ms': round(execution_time, 2),
                'aggregations': response.body.get('aggregations', {})
            }
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_sample_documents(self, index: str, size: int = 5) -> List[Dict[str, Any]]:
        """Get sample documents from an index for context"""
        try:
            response = self.es.search(
                index=index,
                body={
                    "size": size,
                    "query": {"match_all": {}}
                }
            )
            
            return [hit['_source'] for hit in response.body['hits']['hits']]
            
        except Exception as e:
            logger.error(f"Error getting sample documents from {index}: {e}")
            raise
    
    def get_field_names(self, index: str) -> List[str]:
        """Extract all field names from index mapping"""
        try:
            mapping = self.get_mapping(index)
            fields = set()
            
            def extract_fields(obj, prefix=''):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'properties':
                            extract_fields(value, prefix)
                        elif isinstance(value, dict) and 'type' in value:
                            field_name = f"{prefix}.{key}" if prefix else key
                            fields.add(field_name)
                        elif isinstance(value, dict):
                            new_prefix = f"{prefix}.{key}" if prefix else key
                            extract_fields(value, new_prefix)
            
            # Extract from mapping
            for index_name, index_mapping in mapping.items():
                properties = index_mapping.get('mappings', {})
                extract_fields(properties)
            
            return sorted(list(fields))
            
        except Exception as e:
            logger.error(f"Error extracting fields from {index}: {e}")
            return []
    
    @staticmethod
    def _format_bytes(bytes_count: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f} PB"
