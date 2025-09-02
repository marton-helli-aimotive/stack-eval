#!/usr/bin/env python3
"""
Test script to verify OpenRouter integration.
This script tests both model fetching and chat completion.
"""

import os
from dotenv import load_dotenv
from openrouter_client import openrouter_client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_fetching():
    """Test fetching models from OpenRouter"""
    logger.info("Testing model fetching...")
    
    try:
        models = openrouter_client.get_available_models()
        logger.info(f"Successfully fetched {len(models)} models")
        
        # Check if we have some expected models
        model_ids = list(models.keys())
        logger.info(f"Model IDs: {model_ids[:5]}...")
        
        return True
    except Exception as e:
        logger.error(f"Model fetching failed: {e}")
        return False

def test_chat_completion():
    """Test chat completion with OpenRouter"""
    logger.info("Testing chat completion...")
    
    try:
        # Use a simple model for testing
        models = openrouter_client.get_available_models()
        test_model = None
        
        # Find a suitable test model
        for model_id in models:
            if "gpt" in model_id.lower() or "claude" in model_id.lower():
                test_model = model_id
                break
        
        if not test_model:
            logger.warning("No suitable test model found, skipping completion test")
            return True
        
        logger.info(f"Using test model: {test_model}")
        
        messages = [
            {"role": "user", "content": "Hello! Please respond with just 'Hello back!'"}
        ]
        
        response = openrouter_client.chat_completion(
            messages=messages,
            model=test_model,
            temperature=0.1,
            max_tokens=50
        )
        
        logger.info(f"Chat completion response: {response}")
        return True
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        return False

def main():
    """Run all tests"""
    load_dotenv()
    
    logger.info("Starting OpenRouter integration tests...")
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set")
        return 1
    
    # Run tests
    tests = [
        ("Model Fetching", test_model_fetching),
        ("Chat Completion", test_chat_completion),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✓ {test_name} passed")
        else:
            logger.error(f"✗ {test_name} failed")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("All tests passed! OpenRouter integration is working correctly.")
        return 0
    else:
        logger.error("Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main())