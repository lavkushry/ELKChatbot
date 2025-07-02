"""
Prompt builder for generating high-quality Elasticsearch DSL queries
Includes smart prompt engineering and DSL correction capabilities
"""

import json
import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PromptBuilder:
    """Builds optimized prompts for Mistral to generate Elasticsearch DSL"""
    
    def __init__(self):
        self.common_field_mappings = {
            # Common field name corrections for various datasets
            "country": ["geoip.country_iso_code", "country_code", "country_name"],
            "price": ["taxful_total_price", "total_amount", "amount", "cost"],
            "customer": ["customer_full_name", "customer_name", "name"],
            "product": ["products.product_name", "product_name", "title"],
            "quantity": ["total_quantity", "qty", "amount"],
            "date": ["order_date", "timestamp", "@timestamp", "created_at"],
            "category": ["category.keyword", "category", "product_category"]
        }
    
    def build_prompt(self, natural_query: str, mapping: Dict[str, Any]) -> str:
        """Build a comprehensive prompt for DSL generation"""
        
        # Extract available fields
        available_fields = self._extract_fields_from_mapping(mapping)
        
        # Get sample field examples
        field_examples = self._get_field_examples(available_fields)
        
        # Build the main prompt
        prompt = f"""You are an expert Elasticsearch Query DSL generator. Convert the natural language query into a proper Elasticsearch DSL JSON query.

IMPORTANT RULES:
1. Return ONLY valid JSON without any markdown formatting or explanations
2. Use exact field names from the mapping provided below
3. For date math, use lowercase letters: 'y' for years, 'd' for days, 'h' for hours, 'm' for minutes
4. For numeric values in range queries, don't use quotes around numbers
5. Use appropriate query types: match, term, range, bool, etc.
6. Include proper aggregations when asking for counts, sums, or grouping

AVAILABLE FIELDS:
{self._format_fields_for_prompt(available_fields)}

FIELD EXAMPLES:
{field_examples}

ELASTICSEARCH MAPPING STRUCTURE:
{json.dumps(mapping, indent=2)}

QUERY PATTERNS TO USE:
- For text search: {{"match": {{"field_name": "search_term"}}}}
- For exact matches: {{"term": {{"field_name.keyword": "exact_value"}}}}
- For numeric ranges: {{"range": {{"field_name": {{"gte": 100, "lte": 500}}}}}}
- For date ranges: {{"range": {{"date_field": {{"gte": "now-1y", "lte": "now"}}}}}}
- For counting: Include "size": 0 and proper aggregations
- For combining conditions: Use bool query with must, should, filter

NATURAL LANGUAGE QUERY: "{natural_query}"

Generate the Elasticsearch DSL query in JSON format:"""

        return prompt
    
    def _extract_fields_from_mapping(self, mapping: Dict[str, Any]) -> List[str]:
        """Extract all available fields from the mapping"""
        fields = set()
        
        def extract_fields_recursive(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'properties':
                        extract_fields_recursive(value, prefix)
                    elif isinstance(value, dict) and 'type' in value:
                        field_name = f"{prefix}.{key}" if prefix else key
                        fields.add(field_name)
                        # Also add keyword version if text field
                        if value.get('type') == 'text':
                            fields.add(f"{field_name}.keyword")
                    elif isinstance(value, dict):
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        extract_fields_recursive(value, new_prefix)
        
        for index_name, index_mapping in mapping.items():
            properties = index_mapping.get('mappings', {})
            extract_fields_recursive(properties)
        
        return sorted(list(fields))
    
    def _format_fields_for_prompt(self, fields: List[str]) -> str:
        """Format fields in a readable way for the prompt"""
        if not fields:
            return "No fields available"
        
        # Group fields by top-level category
        grouped = {}
        for field in fields:
            top_level = field.split('.')[0]
            if top_level not in grouped:
                grouped[top_level] = []
            grouped[top_level].append(field)
        
        formatted = []
        for category, field_list in sorted(grouped.items()):
            if len(field_list) <= 3:
                formatted.append(f"- {category}: {', '.join(field_list)}")
            else:
                main_fields = [f for f in field_list if not f.endswith('.keyword')][:3]
                formatted.append(f"- {category}: {', '.join(main_fields)}...")
        
        return '\n'.join(formatted)
    
    def _get_field_examples(self, fields: List[str]) -> str:
        """Generate examples of how to use common fields"""
        examples = []
        
        # Common patterns
        text_fields = [f for f in fields if not f.endswith('.keyword') and 'name' in f.lower()]
        if text_fields:
            examples.append(f"Text search: {{\"match\": {{\"{text_fields[0]}\": \"search term\"}}}}")
        
        keyword_fields = [f for f in fields if f.endswith('.keyword')]
        if keyword_fields:
            examples.append(f"Exact match: {{\"term\": {{\"{keyword_fields[0]}\": \"exact value\"}}}}")
        
        numeric_fields = [f for f in fields if any(x in f.lower() for x in ['price', 'amount', 'quantity', 'total'])]
        if numeric_fields:
            examples.append(f"Numeric range: {{\"range\": {{\"{numeric_fields[0]}\": {{\"gte\": 100}}}}}}")
        
        date_fields = [f for f in fields if any(x in f.lower() for x in ['date', 'time', 'created'])]
        if date_fields:
            examples.append(f"Date range: {{\"range\": {{\"{date_fields[0]}\": {{\"gte\": \"now-30d\"}}}}}}")
        
        return '\n'.join(examples) if examples else "No specific examples available"
    
    def clean_and_validate_dsl(self, raw_response: str) -> Dict[str, Any]:
        """Clean and validate the DSL response from Mistral"""
        logger.info("Cleaning and validating DSL response")
        
        # Remove markdown formatting
        cleaned = self._remove_markdown_formatting(raw_response)
        
        # Try to parse JSON
        try:
            dsl_query = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Mistral: {e}")
            # Try to extract JSON from text
            dsl_query = self._extract_json_from_text(cleaned)
            if not dsl_query:
                raise ValueError(f"Could not parse valid JSON from response: {cleaned[:200]}...")
        
        # Apply smart corrections
        corrected_dsl = self._apply_smart_corrections(dsl_query)
        
        return corrected_dsl
    
    def _remove_markdown_formatting(self, text: str) -> str:
        """Remove markdown code block formatting"""
        # Remove ```json or ``` blocks
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```', '', text)
        
        # Remove any explanatory text before/after JSON
        lines = text.strip().split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
                if stripped.endswith('}') and json_lines:
                    # Check if this might be the end of JSON
                    try:
                        json.loads('\n'.join(json_lines))
                        break
                    except:
                        continue
        
        if json_lines:
            return '\n'.join(json_lines)
        
        return text.strip()
    
    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Try to extract JSON from mixed text"""
        # Find JSON-like content
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except:
                continue
        
        return None
    
    def _apply_smart_corrections(self, dsl_query: Dict[str, Any]) -> Dict[str, Any]:
        """Apply smart corrections to the DSL query"""
        # Convert to string for regex operations
        query_str = json.dumps(dsl_query)
        
        # Fix date math units (Y -> y, etc.)
        date_fixes = {
            r'now-([0-9]+)Y': r'now-\1y',
            r'now-([0-9]+)D': r'now-\1d',
            r'now-([0-9]+)H': r'now-\1h',
            r'now-([0-9]+)MIN': r'now-\1m',
        }
        
        for wrong, correct in date_fixes.items():
            if re.search(wrong, query_str):
                query_str = re.sub(wrong, correct, query_str)
                logger.info(f"Fixed date math: {wrong} -> {correct}")
        
        # Fix quoted numbers in range queries
        number_fixes = [
            (r'"gte":\s*"([0-9]+\.?[0-9]*)"', r'"gte": \1'),
            (r'"lte":\s*"([0-9]+\.?[0-9]*)"', r'"lte": \1'),
            (r'"gt":\s*"([0-9]+\.?[0-9]*)"', r'"gt": \1'),
            (r'"lt":\s*"([0-9]+\.?[0-9]*)"', r'"lt": \1'),
        ]
        
        for pattern, replacement in number_fixes:
            if re.search(pattern, query_str):
                query_str = re.sub(pattern, replacement, query_str)
                logger.info("Fixed quoted numbers in range query")
        
        # Ensure query wrapper exists
        try:
            corrected_query = json.loads(query_str)
            if 'query' not in corrected_query and any(key in corrected_query for key in ['match', 'term', 'range', 'bool']):
                corrected_query = {'query': corrected_query}
                logger.info("Added missing 'query' wrapper")
            
            return corrected_query
        except json.JSONDecodeError:
            logger.error("Failed to parse corrected query")
            return dsl_query
    
    def suggest_field_corrections(self, query: str, available_fields: List[str]) -> Dict[str, str]:
        """Suggest field corrections based on common patterns"""
        suggestions = {}
        query_lower = query.lower()
        
        for common_term, possible_fields in self.common_field_mappings.items():
            if common_term in query_lower:
                matching_fields = [f for f in available_fields if any(pf in f for pf in possible_fields)]
                if matching_fields:
                    suggestions[common_term] = matching_fields[0]
        
        return suggestions
