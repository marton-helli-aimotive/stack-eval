import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import json_repair
import pandas as pd
import yaml
from limits import RateLimitItemPerMinute
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from tqdm import tqdm

from openrouter_client import openrouter_client
from openwebui_client import openwebui_client

with open("config/prompts.yml", "r") as file:
    PROMPTS = dict(yaml.safe_load(file))

# Load models from providers (with caching)
_llms_openrouter = {}
_llms_openwebui = {}

try:
    _llms_openrouter = openrouter_client.get_available_models()
except Exception as e:
    logging.warning(f"Failed to load models from OpenRouter, using fallback: {e}")
    try:
        with open("config/llms.yml", "r") as file:
            _llms_openrouter = dict(yaml.safe_load(file))
    except Exception:
        _llms_openrouter = {}

try:
    _llms_openwebui = openwebui_client.get_available_models()
except Exception as e:
    logging.warning(f"Failed to load models from OpenWebUI: {e}")
    _llms_openwebui = {}

# Merge provider model maps. OpenWebUI keys are prefixed with 'openwebui/'.
LLMS = {**_llms_openrouter, **_llms_openwebui}
TASKS = {
    "stack-eval-mini": "data/stack-eval-mini.jsonl",
    "stack-unseen": "data/stack-unseen-2.jsonl",
    "stack-eval": "data/stack-eval.jsonl",
}
_limiter = MovingWindowRateLimiter(MemoryStorage())


def format_prompt(prompt: dict, variables: dict = None) -> list[dict]:
    """
    Formats the prompt based on the provided template and variables.

    Args:
        prompt (dict): The prompt dictionary containing the system and user prompt templates.
        variables (dict, optional): The dictionary containing the variable values to replace in the user message. Defaults to None.

    Returns:
        list[dict]: A list of formatted prompt messages with roles and content.

    Raises:
        ValueError: If there is a mismatch between the prompt requirements and provided variables.
    """
    if ("variables" not in prompt) ^ (variables is None):
        raise ValueError(
            f"Mismatch between prompt variables and provided variables. Expected: {prompt.get('variables')}, got: {variables}"
        )

    messages = []
    if "system" in prompt:
        messages.append({"role": "system", "content": prompt["system"]})

    if "user" in prompt:
        if not variables:
            messages.append({"role": "user", "content": prompt["user"]})
        else:
            user_prompt = prompt["user"]
            for key in variables:
                user_prompt = user_prompt.replace(
                    "{{" + key + "}}", str(variables[key])
                )
            messages.append({"role": "user", "content": user_prompt})

    if len(messages) == 0:
        raise ValueError("Prompt must contain either 'system' or 'user' key.")

    return messages


def batch_format_prompt(
    prompt: dict, data: pd.DataFrame | list[dict]
) -> list[list[dict]]:
    """
    Formats the prompt based on the provided template and variables on a batch of data.

    Args:
        prompt (dict): The prompt to be formatted.
        data (pd.DataFrame | list[dict]): The data to be used for formatting the prompt. It can be either a pandas DataFrame or a list of dictionaries.

    Returns:
        list[list[dict]]: A list of formatted prompts, in OpenAI format.
    """
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    return [format_prompt(prompt, record) for record in data]


def parse_json(json_str: str, expected_keys: list | set, default_value=None) -> dict:
    """
    Loads a JSON string from the given text and returns a dictionary.

    Args:
        json_str (str): The text to search for a JSON string.
        expected_keys (list | set): The expected keys in the JSON string.
        default_value (any): The default value to use for missing keys.

    Returns:
        dict: The dictionary loaded from the JSON string.
    """
    if isinstance(expected_keys, list):
        expected_keys = set(expected_keys)

    try:
        loaded = json_repair.loads(json_str)
        if not isinstance(loaded, dict):
            # Not an object -> cannot extract expected keys
            return {}
        data = loaded
    except (ValueError, TypeError) as e:
        print(f"Error parsing JSON: {e}")
        return {}

    return {key: data.get(key, default_value) for key in expected_keys}


def batch_parse_json(
    json_strs: list[str] | str, expected_keys: list | set, default_value=None
) -> list[dict]:
    """
    Parses a list of JSON strings and returns a list of dictionaries containing the values of the expected keys.

    Args:
        json_strs (list[str]): A list of JSON strings to parse.
        expected_keys (list | set): A list of keys whose values are expected to be present in the JSON.
        default_value (any): The default value to use for missing keys.

    Returns:
        list[dict]: A list of dictionaries containing the values of the expected keys for each JSON string.
    """
    if isinstance(json_strs, str):
        json_strs = [json_strs]
    return [parse_json(j, expected_keys, default_value) for j in json_strs]


def rate_limiter(func: callable, rate: int):
    """
    Decorator function to rate limit the execution of the decorated function.

    Args:
        func (callable): The function to be rate limited.
        rate (int): The rate limit in requests per minute.

    Returns:
        callable: The decorated function.
    """
    rate_limit = RateLimitItemPerMinute(rate)

    def wrapper(*args, **kwargs):
        try:
            while not _limiter.hit(rate_limit, func.__name__):
                time.sleep(0.1)
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return e

    return wrapper


def batch_executor(
    func: callable, func_args: list[dict], rate: int = None, verbose: bool = True
):
    """
    Execute a function in parallel for a list of arguments using a thread pool executor.

    Args:
        func (callable): The function to be executed.
        func_args (list[dict]): A list of dictionaries containing the arguments for each function call.
        rate (int, optional): The rate limit for making API requests. Defaults to None.
        verbose (bool, optional): Whether to display progress information. Defaults to True.

    Returns:
        list: A list of results from each function call.
    """
    if rate:
        func = rate_limiter(func, rate)

    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(func, **args): i for i, args in enumerate(func_args)}
        for future in tqdm(
            as_completed(futures), total=len(futures), disable=not verbose
        ):
            try:
                results.append((futures[future], future.result()))
            except Exception as e:
                results.append((futures[future], e))
    results = [result[1] for result in sorted(results, key=lambda x: x[0])]
    return results


def completion(
    messages: list[list[dict]] | list[dict],
    model: str,
    custom_llm_provider: str,
    rate_limit: int = None,
    verbose: bool = True,
    **kwargs,
) -> list[str]:
    """
    Generate completions for a list of messages using OpenRouter API.

    Args:
        messages (list[list[dict]] | list[dict]): A list of messages to generate completions for.
        model (str): The language model to use for generating completions.
        custom_llm_provider (str): The custom language model provider to use for generating completions.
        rate_limit (int, optional): The rate limit for making API requests. Defaults to None.
        verbose (bool, optional): Whether to display progress information. Defaults to True.

    Returns:
        list[str]: One completion string per input message (always a list).
    """
    if isinstance(messages[0], dict):
        messages = [messages]

    # Select provider client and model id mapping
    if custom_llm_provider == "openrouter":
        func = openrouter_client.chat_completion
        model_id = model
    elif custom_llm_provider == "openwebui":
        func = openwebui_client.chat_completion
        model_id = model
    else:
        # Unknown provider: try routing through OpenRouter with provider/model path
        func = openrouter_client.chat_completion
        model_id = f"{custom_llm_provider}/{model}"

    func_args = [{"messages": m, "model": model_id, **kwargs} for m in messages]
    responses = batch_executor(
        func=func, func_args=func_args, rate=rate_limit, verbose=verbose,
    )
    
    # Providers return strings directly
    completions = [r if isinstance(r, str) else "" for r in responses]
    # Always return a list, even for a single input
    return completions


def setup_logger(name: str):
    """
    Set up a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    # Prevent duplicate handlers on Streamlit reruns
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
