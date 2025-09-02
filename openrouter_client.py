import os
import json
import requests
import yaml
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """Client for interacting with OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key missing. Set OPENROUTER_API_KEY to enable it.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.models_cache_file = "config/openrouter_models_cache.yml"
        self.cache_duration = timedelta(hours=24)  # Cache models for 24 hours
        try:
            self.request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
        except Exception:
            self.request_timeout = 120.0

        # Optional but recommended headers per OpenRouter guidance
        # See https://openrouter.ai/docs for details
        self.app_title = os.getenv("OPENROUTER_APP_TITLE", "Stack Eval Dashboard")
        self.http_referer = (
            os.getenv("OPENROUTER_HTTP_REFERER")
            or os.getenv("SITE_URL")
            or os.getenv("STREAMLIT_SERVER_URL")
            or "http://localhost"
        )

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": self.http_referer,
            "X-Title": self.app_title,
        }
        return headers
        
    def get_available_models(self, force_refresh: bool = False) -> Dict:
        """
        Get available models from OpenRouter API or cache.
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            Dictionary of available models in the same format as the original llms.yml
        """
        if not force_refresh and self._is_cache_valid():
            return self._load_cached_models()
        
        try:
            models = self._fetch_models_from_api()
            self._cache_models(models)
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from OpenRouter API: {e}")
            if self._is_cache_valid():
                logger.info("Using cached models due to API failure")
                return self._load_cached_models()
            else:
                raise
    
    def _fetch_models_from_api(self) -> Dict:
        """Fetch models from OpenRouter API"""
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set; cannot fetch models from API.")
        headers = self._build_headers()
        
        response = requests.get(
            f"{self.base_url}/models",
            headers=headers,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        
        data = response.json()
        return self._transform_models_data(data)
    
    def _transform_models_data(self, api_data: Dict) -> Dict:
        """Transform OpenRouter API response to match the original llms.yml format"""
        models = {}
        
        for model_info in api_data.get("data", []):
            model_id = model_info.get("id", "")
            if not model_id:
                continue
                
            # Create a model configuration similar to the original format
            model_config = {
                "custom_llm_provider": "openrouter",
                "sample_parameters": {
                    "temperature": 0.01,
                    "max_tokens": 2048,
                    "seed": 42
                },
                "model_parameters": {
                    "model": model_id,
                    "rate_limit": self._estimate_rate_limit(model_info)
                },
                "model_metadata": {
                    "name": model_info.get("name", model_id),
                    "provider": model_info.get("pricing", {}).get("prompt", "Unknown"),
                    "context_length": model_info.get("context_length", 0),
                    "description": model_info.get("description", "")
                }
            }
            
            # Add JSON response format for models that support it
            if "json" in model_id.lower() or "json" in model_info.get("name", "").lower():
                model_config["sample_parameters"]["response_format"] = {"type": "json_object"}
                models[f"{model_id}-json"] = model_config.copy()
            
            models[model_id] = model_config
        
        # Add some default configurations
        models["default"] = models.get("openai/gpt-4o", models.get("anthropic/claude-3-5-sonnet", list(models.values())[0] if models else {}))
        models["default-json"] = models.get("openai/gpt-4o-json", models.get("anthropic/claude-3-5-sonnet", list(models.values())[0] if models else {}))
        
        return models
    
    def _estimate_rate_limit(self, model_info: Dict) -> int:
        """Estimate rate limit based on model pricing and context length"""
        # This is a rough estimation - you might want to adjust based on your needs
        context_length = model_info.get("context_length", 100000)
        if context_length > 100000:
            return 50  # High context models
        elif context_length > 50000:
            return 100  # Medium context models
        else:
            return 200  # Standard models
    
    def _is_cache_valid(self) -> bool:
        """Check if the cached models are still valid"""
        if not os.path.exists(self.models_cache_file):
            return False
        
        try:
            with open(self.models_cache_file, 'r') as f:
                cache_data = yaml.safe_load(f)
                cache_time = datetime.fromisoformat(cache_data.get("_cache_time", "1970-01-01T00:00:00"))
                return datetime.now() - cache_time < self.cache_duration
        except Exception:
            return False
    
    def _load_cached_models(self) -> Dict:
        """Load models from cache"""
        try:
            with open(self.models_cache_file, 'r') as f:
                cache_data = yaml.safe_load(f)
                # Remove cache metadata
                cache_data.pop("_cache_time", None)
                return cache_data
        except Exception as e:
            logger.error(f"Failed to load cached models: {e}")
            return {}
    
    def _cache_models(self, models: Dict):
        """Cache models to file"""
        try:
            os.makedirs(os.path.dirname(self.models_cache_file), exist_ok=True)
            cache_data = models.copy()
            cache_data["_cache_time"] = datetime.now().isoformat()
            
            with open(self.models_cache_file, 'w') as f:
                yaml.dump(cache_data, f, default_flow_style=False)
            
            logger.info(f"Cached {len(models)} models to {self.models_cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache models: {e}")
    
    def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        **kwargs
    ) -> str:
        """
        Generate chat completion using OpenRouter API
        
        Args:
            messages: List of message dictionaries with role and content
            model: Model ID to use
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Generated completion text
        """
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set; cannot perform chat completion.")
        headers = self._build_headers()
        
        data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        try:
            timeout_value = float(kwargs.pop("timeout", self.request_timeout))
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout_value,
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            # Try to include error payload for easier debugging
            try:
                err_json = e.response.json()
                err_msg = err_json.get("error") or err_json.get("message") or err_json
            except Exception:
                err_msg = getattr(e.response, "text", str(e))
            status = getattr(e.response, "status_code", "?")
            logger.error(f"OpenRouter API HTTP {status}: {err_msg}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format from OpenRouter API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in chat completion: {e}")
            raise

# Global instance
openrouter_client = OpenRouterClient()