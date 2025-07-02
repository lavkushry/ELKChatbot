"""
Test script for ELK Chatbot FastAPI
Run this after starting the server to test basic functionality
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['message']}")
            print(f"   - Elasticsearch: {data.get('elasticsearch', 'Unknown')}")
            print(f"   - Mistral: {data.get('mistral', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_indices_endpoint():
    """Test the indices listing endpoint"""
    print("\n📋 Testing indices endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/indices")
        if response.status_code == 200:
            indices = response.json()
            print(f"✅ Found {len(indices)} indices:")
            for idx in indices[:3]:  # Show first 3
                print(f"   - {idx['name']}: {idx['doc_count']} docs, {idx['store_size']}")
            return True
        else:
            print(f"❌ Indices endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Indices endpoint error: {e}")
        return False

def test_mapping_endpoint():
    """Test the mapping endpoint"""
    print("\n🗺️  Testing mapping endpoint...")
    try:
        # Use a common index name
        index_name = "kibana_sample_data_ecommerce"
        response = requests.get(f"{BASE_URL}/mapping/{index_name}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Mapping retrieved for {index_name}")
            return True
        elif response.status_code == 404:
            print(f"⚠️  Index {index_name} not found (this is OK if you don't have sample data)")
            return True
        else:
            print(f"❌ Mapping endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Mapping endpoint error: {e}")
        return False

def test_query_endpoint():
    """Test the main query endpoint"""
    print("\n🔍 Testing query endpoint...")
    try:
        test_query = {
            "index": "kibana_sample_data_ecommerce",
            "query": "show me 5 orders",
            "max_results": 5
        }
        
        print(f"   Query: '{test_query['query']}'")
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=test_query,
            timeout=60  # 1 minute timeout for Mistral
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Query successful!")
            print(f"   - Total hits: {data['total_hits']}")
            print(f"   - Results returned: {len(data['results'])}")
            print(f"   - Execution time: {data['execution_time_ms']}ms")
            print(f"   - Generated DSL: {json.dumps(data['dsl_query'], indent=2)[:200]}...")
            return True
        elif response.status_code == 404:
            print(f"⚠️  Index not found (add sample data: kibana_sample_data_ecommerce)")
            return True
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Query timed out (Mistral might be slow or not responding)")
        return False
    except Exception as e:
        print(f"❌ Query error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 ELK Chatbot API Test Suite")
    print("=" * 50)
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    tests = [
        test_health_check,
        test_indices_endpoint,
        test_mapping_endpoint,
        test_query_endpoint
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    print(f"\n📖 API Documentation: {BASE_URL}/docs")
    print(f"📚 Alternative docs: {BASE_URL}/redoc")

if __name__ == "__main__":
    main()
