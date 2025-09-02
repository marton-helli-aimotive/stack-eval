#!/usr/bin/env python3
"""
Utility script to refresh the OpenRouter models cache.
Run this script to fetch the latest available models from OpenRouter API.
"""

import os
from dotenv import load_dotenv
from openrouter_client import openrouter_client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Refresh the OpenRouter models cache"""
    load_dotenv()
    
    try:
        logger.info("Fetching latest models from OpenRouter API...")
        models = openrouter_client.get_available_models(force_refresh=True)
        logger.info(f"Successfully cached {len(models)} models")
        
        # Print some model examples
        logger.info("Available models include:")
        for i, (model_id, config) in enumerate(list(models.items())[:10]):
            if not model_id.startswith("_"):  # Skip cache metadata
                name = config.get("model_metadata", {}).get("name", model_id)
                provider = config.get("model_metadata", {}).get("provider", "Unknown")
                logger.info(f"  {model_id}: {name} ({provider})")
        
        if len(models) > 10:
            logger.info(f"  ... and {len(models) - 10} more models")
            
    except Exception as e:
        logger.error(f"Failed to refresh models: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())