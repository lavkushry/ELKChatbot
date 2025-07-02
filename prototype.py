import os
import json
import re
import requests
import time
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# Setup Elasticsearch connection
es = Elasticsearch(
    f"https://{os.getenv('ES_HOST')}:{os.getenv('ES_PORT')}",
    basic_auth=(os.getenv('ES_USERNAME'), os.getenv('ES_PASSWORD')),
    ca_certs=os.getenv('ES_CA_CERT'),
    verify_certs=True
)

class ElasticsearchDSLCorrector:
    """
    Smart DSL corrector that fixes common LLM-generated Elasticsearch query issues
    """
    
    def __init__(self, index_name):
        self.index_name = index_name
        self.mapping = self.get_index_mapping()
        self.available_fields = self.extract_available_fields()
    
    def get_index_mapping(self):
        """Get the mapping for the index"""
        try:
            return es.indices.get_mapping(index=self.index_name).body
        except Exception as e:
            print(f"❌ Error getting mapping: {e}")
            return {}
    
    def extract_available_fields(self):
        """Extract all available fields from the mapping"""
        fields = set()
        try:
            mapping = self.mapping.get(self.index_name, {}).get('mappings', {})
            
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
            
            extract_fields(mapping)
        except Exception as e:
            print(f"Warning: Could not extract fields from mapping: {e}")
        
        return fields
    
    def fix_date_math_units(self, dsl_query):
        """Fix invalid date math units"""
        fixes = []
        
        # Fix uppercase Y to lowercase y (years)
        if re.search(r'now-[0-9]+Y', dsl_query):
            dsl_query = re.sub(r'now-([0-9]+)Y', r'now-\1y', dsl_query)
            fixes.append("Fixed date math: 'Y' → 'y' (year unit)")
        
        # Fix other potential date math issues
        date_unit_fixes = {
            r'now-([0-9]+)D': r'now-\1d',  # Day
            r'now-([0-9]+)H': r'now-\1h',  # Hour
            r'now-([0-9]+)MIN': r'now-\1m',  # Minutes
            r'now-([0-9]+)S': r'now-\1s',   # Seconds
        }
        
        for wrong_pattern, correct_pattern in date_unit_fixes.items():
            if re.search(wrong_pattern, dsl_query):
                dsl_query = re.sub(wrong_pattern, correct_pattern, dsl_query)
                fixes.append(f"Fixed date math unit: {wrong_pattern} → {correct_pattern}")
        
        return dsl_query, fixes
    
    def fix_field_names(self, dsl_query):
        """Fix common field name mistakes"""
        fixes = []
        
        # Common field name corrections for Kibana sample ecommerce data
        field_corrections = {
            # Country fields
            '"country"': '"geoip.country_iso_code"',
            '"country_code"': '"geoip.country_iso_code"',
            '"country_name"': '"geoip.country_iso_code"',
            
            # Price/amount fields
            '"total_amount"': '"taxful_total_price"',
            '"price"': '"taxful_total_price"',
            '"amount"': '"taxful_total_price"',
            '"total_price"': '"taxful_total_price"',
            '"cost"': '"taxful_total_price"',
            
            # Customer fields
            '"customer_name"': '"customer_full_name"',
            '"customer"': '"customer_full_name"',
            '"name"': '"customer_full_name"',
            
            # Product fields
            '"product"': '"products.product_name"',
            '"product_name"': '"products.product_name"',
            
            # Quantity fields
            '"quantity"': '"total_quantity"',
            '"qty"': '"total_quantity"',
            
            # Date fields
            '"date"': '"order_date"',
            '"timestamp"': '"order_date"',
        }
        
        for wrong_field, correct_field in field_corrections.items():
            if wrong_field in dsl_query and correct_field.strip('"') in self.available_fields:
                dsl_query = dsl_query.replace(wrong_field, correct_field)
                fixes.append(f"Fixed field name: {wrong_field} → {correct_field}")
        
        return dsl_query, fixes
    
    def fix_range_queries(self, dsl_query):
        """Fix range query syntax issues"""
        fixes = []
        
        # Fix numeric values that are incorrectly quoted
        numeric_range_fixes = [
            (r'"gte":\s*"([0-9]+\.?[0-9]*)"', r'"gte": \1'),
            (r'"lte":\s*"([0-9]+\.?[0-9]*)"', r'"lte": \1'),
            (r'"gt":\s*"([0-9]+\.?[0-9]*)"', r'"gt": \1'),
            (r'"lt":\s*"([0-9]+\.?[0-9]*)"', r'"lt": \1'),
        ]
        
        for pattern, replacement in numeric_range_fixes:
            if re.search(pattern, dsl_query):
                dsl_query = re.sub(pattern, replacement, dsl_query)
                fixes.append("Fixed numeric range query (removed quotes from numbers)")
        
        return dsl_query, fixes
    
    def fix_boolean_structure(self, dsl_query):
        """Fix boolean query structure issues"""
        fixes = []
        
        try:
            parsed = json.loads(dsl_query)
            
            # Ensure query wrapper exists
            if 'query' not in parsed:
                parsed = {'query': parsed}
                fixes.append("Added missing 'query' wrapper")
            
            # Convert back to string
            dsl_query = json.dumps(parsed, indent=2)
            
        except json.JSONDecodeError:
            # If JSON is malformed, we'll handle this in validation
            pass
        
        return dsl_query, fixes
    
    def validate_and_suggest_fields(self, dsl_query):
        """Validate field names and suggest corrections"""
        suggestions = []
        
        # Extract field names from the query
        field_pattern = r'"([a-zA-Z_][a-zA-Z0-9_.]*)":\s*\{'
        found_fields = re.findall(field_pattern, dsl_query)
        
        for field in found_fields:
            if field not in self.available_fields:
                # Find similar fields
                similar_fields = [f for f in self.available_fields if field.lower() in f.lower() or f.lower() in field.lower()]
                if similar_fields:
                    suggestions.append(f"Field '{field}' not found. Similar fields: {', '.join(similar_fields[:3])}")
        
        return suggestions
    
    def fix_all_issues(self, dsl_query):
        """Apply all fixes to the DSL query"""
        print("🔧 Running comprehensive DSL fixes...")
        
        original_query = dsl_query
        all_fixes = []
        
        # Apply all fixes
        dsl_query, fixes = self.fix_date_math_units(dsl_query)
        all_fixes.extend(fixes)
        
        dsl_query, fixes = self.fix_field_names(dsl_query)
        all_fixes.extend(fixes)
        
        dsl_query, fixes = self.fix_range_queries(dsl_query)
        all_fixes.extend(fixes)
        
        dsl_query, fixes = self.fix_boolean_structure(dsl_query)
        all_fixes.extend(fixes)
        
        # Validate and provide suggestions
        suggestions = self.validate_and_suggest_fields(dsl_query)
        
        # Report results
        if all_fixes:
            print("✅ Applied fixes:")
            for fix in all_fixes:
                print(f"   • {fix}")
        
        if suggestions:
            print("💡 Suggestions:")
            for suggestion in suggestions:
                print(f"   • {suggestion}")
        
        if not all_fixes and not suggestions:
            print("✅ No issues found in DSL query")
        
        return dsl_query

# Initialize corrector
corrector = None

def get_dsl_from_mistral_with_correction(prompt, index_name):
    """Get DSL from Mistral and apply smart corrections"""
    global corrector
    
    if corrector is None:
        corrector = ElasticsearchDSLCorrector(index_name)
    
    # Get raw response from Mistral (using the ultra patient method)
    raw_dsl = get_dsl_from_mistral_ultra_patient(prompt)
    
    # Apply smart corrections
    corrected_dsl = corrector.fix_all_issues(raw_dsl)
    
    return corrected_dsl

# ... (rest of the functions from ultra_patient version)

def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def build_prompt(nl_query, mapping):
    return f"""You are a helpful assistant that translates natural language queries into Elasticsearch DSL.

Here is the Elasticsearch mapping:
{json.dumps(mapping, indent=2)}

Important guidelines:
1. Use lowercase 'y' for years in date math (not 'Y')
2. Use exact field names from the mapping
3. For numeric ranges, don't quote the numbers
4. Return only valid JSON without markdown formatting

Translate the following natural language query into Elasticsearch DSL:

"{nl_query}"

Return only the JSON DSL query."""

def clean_json_from_mistral_response(response_text):
    match = re.search(r"```(?:json)?(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
    except:
        pass
    
    return response_text.strip()

def get_dsl_from_mistral_ultra_patient(prompt):
    base_timeout = 600
    max_timeout = 3600
    attempt = 1
    
    while True:
        current_timeout = min(base_timeout * attempt, max_timeout)
        
        try:
            print(f"🔄 Attempt {attempt} - Timeout: {current_timeout//60} minutes")
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_ctx": 4096,
                        "num_predict": 2048
                    }
                },
                timeout=current_timeout
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ Response received after {elapsed_time:.1f} seconds")
            
            if response.status_code == 200:
                return clean_json_from_mistral_response(response.json()["response"])
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout after {current_timeout} seconds")
            
        except requests.exceptions.ConnectionError as e:
            print(f"🔌 Connection error: {e}")
            if not check_ollama_status():
                input("Press Enter when Ollama is running again...")
                continue
        
        time.sleep(min(30, attempt * 5))
        attempt += 1
        
        if attempt % 5 == 0:
            response = input(f"Tried {attempt} times. Continue? (y/n): ").lower()
            if response != 'y':
                raise Exception("User chose to stop")

def run_query_with_smart_retry(dsl_query, index_name):
    """Run query with smart error handling and auto-retry with fixes"""
    try:
        parsed_query = json.loads(dsl_query)
        result = es.search(index=index_name, body=parsed_query)
        print("✅ Query executed successfully")
        return result
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return None
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Elasticsearch error: {error_msg}")
        
        # Try to fix specific errors
        if "unit [Y] not supported" in error_msg:
            print("🔧 Applying emergency date math fix...")
            fixed_query = re.sub(r'now-([0-9]+)Y', r'now-\1y', dsl_query)
            if fixed_query != dsl_query:
                print("✅ Applied fix, retrying...")
                return run_query_with_smart_retry(fixed_query, index_name)
        
        return None

# Main workflow
if __name__ == "__main__":
    index_name = "kibana_sample_data_ecommerce"
    user_query = "How Many products in Men's Clothing category"
    
    print("🚀 Smart DSL Auto-Corrector for Elasticsearch")
    print("🧠 This version automatically fixes common LLM mistakes!")
    print("=" * 60)
    
    # Check Ollama
    if not check_ollama_status():
        print("❌ Ollama not running. Please start: ollama serve")
        exit(1)
    
    print("✅ Ollama is running")
    
    # Initialize corrector
    corrector = ElasticsearchDSLCorrector(index_name)
    print(f"📋 Loaded mapping for {index_name}")
    print(f"🔍 Found {len(corrector.available_fields)} available fields")
    
    # Build prompt
    mapping = corrector.mapping
    prompt = build_prompt(user_query, mapping)
    
    # Get and correct DSL
    print(f"\n📨 Processing query: '{user_query}'")
    try:
        dsl_query = get_dsl_from_mistral_with_correction(prompt, index_name)
        print(f"\n🎉 Final corrected DSL:")
        print("=" * 50)
        print(json.dumps(json.loads(dsl_query), indent=2))
        print("=" * 50)
    except Exception as e:
        print(f"❌ Failed: {e}")
        exit(1)
    
    # Execute query
    print("\n🔎 Executing query...")
    results = run_query_with_smart_retry(dsl_query, index_name)
    
    if results:
        total_hits = results['hits']['total']['value']
        print(f"\n🎯 SUCCESS! Found {total_hits} results")
        
        if total_hits > 0:
            print("\n📋 Sample results:")
            for i, hit in enumerate(results['hits']['hits'][:3], 1):
                print(f"\n--- Result {i} ---")
                print(json.dumps(hit['_source'], indent=2))
    else:
        print("❌ Query execution failed")
    
    print("\n✨ Process completed with smart corrections!")