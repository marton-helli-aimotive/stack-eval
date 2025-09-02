import os
import json
import requests
import yaml
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


class OpenWebUIClient:
    """Client for interacting with an OpenAI-compatible Open WebUI (Ollama) API"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        # API key and base URL are read from environment by default
        self.api_key = api_key or os.getenv("OPENWEBUI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Do not raise at import time; some users may not configure this provider
            logger.warning("OpenWebUI API key missing. Set OPENWEBUI_API_KEY in your environment to enable it.")

        self.base_url = (base_url or os.getenv("OPENWEBUI_BASE_URL") or "https://chat.aimotive.com/ollama/v1").rstrip("/")
        self.models_cache_file = "config/openwebui_models_cache.yml"
        self.cache_duration = timedelta(hours=24)

    def is_enabled(self) -> bool:
        return self.api_key is not None and len(self.api_key) > 0

    def get_available_models(self, force_refresh: bool = False) -> Dict:
        """
        Get available models from Open WebUI API or cache.

        Returns a dict keyed by provider-prefixed model ids: "openwebui/{model_id}"
        with values matching the same shape expected elsewhere in the app.
        """
        if not force_refresh and self._is_cache_valid():
            return self._load_cached_models()

        try:
            models = self._fetch_models_from_api()
            self._cache_models(models)
            return models
        except Exception as e:
            logger.error(f"Failed to fetch models from OpenWebUI API: {e}")
            if self._is_cache_valid():
                logger.info("Using cached OpenWebUI models due to API failure")
                return self._load_cached_models()
            else:
                # If not enabled or not reachable, return empty set rather than raising
                return {}

    def _fetch_models_from_api(self) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Try OpenAI-compatible /models endpoint
        response = requests.get(f"{self.base_url}/models", headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return self._transform_models_data(data)

    def _transform_models_data(self, api_data: Dict) -> Dict:
        """Transform Open WebUI models response to our internal llms.yml-like format"""
        models: Dict[str, Dict] = {}

        # Normalize possible shapes
        raw_models: List[Dict] = []
        if isinstance(api_data, dict):
            if "data" in api_data and isinstance(api_data["data"], list):
                raw_models = api_data["data"]
            elif "models" in api_data and isinstance(api_data["models"], list):
                raw_models = api_data["models"]
            elif "objects" in api_data and isinstance(api_data["objects"], list):
                raw_models = api_data["objects"]
        elif isinstance(api_data, list):
            raw_models = api_data

        for item in raw_models:
            model_id = item.get("id") or item.get("name") or item.get("model")
            if not model_id:
                continue

            key = f"openwebui/{model_id}"
            model_config = {
                "custom_llm_provider": "openwebui",
                "sample_parameters": {
                    "temperature": 0.01,
                    "max_tokens": 2048,
                    "seed": 42,
                },
                "model_parameters": {
                    # For chat completions we will send just the raw model id
                    "model": model_id,
                    # Conservative default; adjust per deployment if needed
                    "rate_limit": 120,
                },
                "model_metadata": {
                    "name": model_id,
                    "provider": "openwebui",
                    "context_length": item.get("context_length", 0),
                    "description": item.get("description", ""),
                },
            }

            models[key] = model_config

        # Provide defaults if available
        if models:
            first_key = sorted(models.keys())[0]
            models.setdefault("openwebui/default", models[first_key])
            models.setdefault("openwebui/default-json", models[first_key])

        return models

    def _is_cache_valid(self) -> bool:
        if not os.path.exists(self.models_cache_file):
            return False
        try:
            with open(self.models_cache_file, "r", encoding="utf-8") as f:
                cache_data = yaml.safe_load(f) or {}
                cache_time = datetime.fromisoformat(cache_data.get("_cache_time", "1970-01-01T00:00:00"))
                return datetime.now() - cache_time < self.cache_duration
        except Exception:
            return False

    def _load_cached_models(self) -> Dict:
        try:
            with open(self.models_cache_file, "r", encoding="utf-8") as f:
                cache_data = yaml.safe_load(f) or {}
                cache_data.pop("_cache_time", None)
                return cache_data
        except Exception as e:
            logger.error(f"Failed to load OpenWebUI cached models: {e}")
            return {}

    def _cache_models(self, models: Dict):
        try:
            os.makedirs(os.path.dirname(self.models_cache_file), exist_ok=True)
            cache_data = models.copy()
            cache_data["_cache_time"] = datetime.now().isoformat()
            with open(self.models_cache_file, "w", encoding="utf-8") as f:
                yaml.dump(cache_data, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Cached {len(models)} OpenWebUI models to {self.models_cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache OpenWebUI models: {e}")

    def chat_completion(
        self,
        messages: List[Dict],
        model: str,
        **kwargs,
    ) -> str:
        """
        Generate chat completion using OpenAI-compatible /chat/completions endpoint on Open WebUI.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {"model": model, "messages": messages, **kwargs}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            # OpenAI-compatible shape
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWebUI API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected response format from OpenWebUI API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in OpenWebUI chat completion: {e}")
            raise


# Global instance
openwebui_client = OpenWebUIClient()


