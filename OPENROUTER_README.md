# OpenRouter Integration

This codebase has been refactored to use the OpenRouter API instead of a static list of LLMs. OpenRouter provides access to a wide variety of language models from different providers through a single API.

## Setup

### 1. Get OpenRouter API Key

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Set the environment variable:

```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

Or create a `.env` file:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## How It Works

### Model Discovery

Instead of using the static `config/llms.yml` file, the system now:

1. **Fetches models dynamically** from OpenRouter's `/models` endpoint
2. **Caches the results** in `config/openrouter_models_cache.yml` for 24 hours
3. **Falls back gracefully** to the original config if OpenRouter is unavailable

### Model Format

The OpenRouter models are transformed to match the original format:

```yaml
openai/gpt-4o:
  custom_llm_provider: openrouter
  sample_parameters:
    temperature: 0.01
    max_tokens: 2048
    seed: 42
  model_parameters:
    model: openai/gpt-4o
    rate_limit: 140
  model_metadata:
    name: GPT-4o
    provider: OpenAI
    context_length: 128000
    description: "GPT-4o model..."
```

### API Integration

All inference and evaluation now use OpenRouter's chat completions API:

```python
# Instead of litellm, we use OpenRouter directly
response = openrouter_client.chat_completion(
    messages=messages,
    model="openai/gpt-4o",
    temperature=0.01,
    max_tokens=2048
)
```

## Usage

### Running Inference

```bash
python inference.py --task stack-unseen --model openai/gpt-4o
```

### Running Evaluation

```bash
python evaluation.py --task stack-unseen --evaluatee openai/gpt-4o --evaluator anthropic/claude-3-5-sonnet
```

### Refreshing Models Cache

To manually refresh the models cache:

```bash
python refresh_models.py
```

### Testing Integration

To test that everything works:

```bash
python test_openrouter.py
```

## Available Models

The system automatically discovers models from providers like:

- **OpenAI**: GPT-4o, GPT-4o Mini, GPT-4 Turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Google**: Gemini Pro, Gemini Flash
- **Meta**: Llama 3.1 8B, Llama 3.1 70B
- **Mistral**: Mistral 7B, Mixtral 8x7B
- And many more...

## Benefits

1. **Always up-to-date**: Access to the latest models as they become available
2. **Provider diversity**: Choose from multiple AI providers
3. **Cost optimization**: Compare pricing across providers
4. **Unified API**: Single interface for all models
5. **Automatic fallback**: Graceful degradation if OpenRouter is unavailable

## Configuration

### Cache Settings

The models cache is stored in `config/openrouter_models_cache.yml` and is valid for 24 hours by default. You can modify this in `openrouter_client.py`:

```python
self.cache_duration = timedelta(hours=24)  # Change this value
```

### Rate Limiting

Rate limits are automatically estimated based on model context length and pricing. You can customize this logic in the `_estimate_rate_limit` method.

## Troubleshooting

### Common Issues

1. **API Key Not Set**: Ensure `OPENROUTER_API_KEY` environment variable is set
2. **Rate Limiting**: Check your OpenRouter account limits
3. **Model Not Found**: Some models may be temporarily unavailable
4. **Cache Issues**: Run `python refresh_models.py` to clear cache

### Fallback Behavior

If OpenRouter is unavailable, the system will:
1. Try to use cached models
2. Fall back to the original `config/llms.yml`
3. Log warnings about the fallback

### Logging

Enable debug logging to see detailed API interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Migration Notes

- The original `config/llms.yml` is kept as a fallback
- All existing scripts (`inference.py`, `evaluation.py`) work without changes
- Model names now use the format `provider/model-id` (e.g., `openai/gpt-4o`)
- The `custom_llm_provider` field is set to `"openrouter"` for all models