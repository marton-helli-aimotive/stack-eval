# StackEval: Benchmarking LLMs in Coding Assistance
~ **[Nidhish Shah](https://www.linkedin.com/in/nidhish-s-shah/), [Zulkuf Genc](https://www.linkedin.com/in/zulkufgenc/), [Dogu Araci](https://www.linkedin.com/in/dogutanaraci5522b5a3/)**

We present two comprehensive benchmarks to evaluate the performance of language models in coding assistance tasks, covering code writing, debugging, code review, and conceptual understanding. Our main contribution includes two curated datasets: StackEval, a large-scale benchmark derived from Stack Overflow questions, and StackUnseen, a dynamic benchmark featuring the most recent Stack Overflow content. These benchmarks offer novel insights into the capabilities and limitations of LLMs, particularly in handling new and emerging content. Additionally, we assess LLMs' proficiency as judges for coding tasks using a curated, human-annotated dataset, exploring their evaluation capabilities and potential biases, including whether they favor their own generated solutions. Our findings underscore the potential of these benchmarks to advance LLM development and application in coding assistance.

### StackEval
The StackEval benchmark can be downloaded [here](./data/stack-eval.jsonl). The dataset contains `925` questions sampled between January 2018 - September 2023, and filtered to remove links and image. The following columns are present in the dataset:
- `questionId`: A unique ID
- `question`: The sampled StackOverflow question.
- `answer`: The accepted answer, containing at least 1 upvote.
- `questionMetadata`: A dictionary containing the following keys:
    - `type`: One of `implemention`, `conceptual`, `debugging`, and `optimisation`.
    - `level`: One of `beginner`, `intermediate`, `advanced`.
    - `tag`: The programming language of the question, eg - `python`, `cpp`, `java`, etc.

### StackUnseen
The StackEval-Recent benchmark can be downloaded at the following links. The dataset is dynamic, and thus will be continuously updated:
- [September 2023 - March 2024](./data/stack-unseen-1.jsonl)
- [March 2024 - May 2024](./data/stack-unseen-2.jsonl)

The following columns are present in the dataset.
- `questionId`: A unique ID
- `question`: The sampled StackOverflow question.
- `answer`: The accepted answer, containing at least 1 upvote.
- `questionMetadata`: A dictionary containing the following keys:
    - `type`: One of `implemention`, `conceptual`, `debugging`, `optimisation`, `version`.
    - `level`: One of `beginner`, `intermediate`, `advanced`.
    - `tag`: The programming language of the question, eg - `python`, `cpp`, `java`, etc.

### LLM-as-a-Judge
A small sample of the LLM-as-a-Judge benchmark can be downloaded [here](./data/llm-as-judge.jsonl). Due to the large effort involved in data-labelling, the entire benchmark will be released once the next iteration of the benchmark is curated. This is done to keep the benchmark private, and protect the integrity of the evaluations. The following columns are present in the dataset:
- `Id`: A unique ID
- `Question`: The sampled StackOverflow question.
- `Answer`: The accepted answer, containing at least 1 upvote.
- `Completion`: The LLM-generated answer to be evaluated.
- `Model`: The LLM used for answer generation.
- `Level`: One of `Beginner`, `Intermediate`, `Advanced`.
- `Type`: One of `Implemention`, `Conceptual`, `Debugging`, and `Optimisation`.
- `Acceptance`: Human-label for whether the generated answer is acceptable compared to the accepted answer.
- `Evaluation`: Human reasoning or the acceptance label.

### Inference & Evaluation
We provide our inference and evaluation code along with the corresponding prompts. The system now uses **OpenRouter API** to dynamically fetch available models from multiple AI providers. To run inference/evaluations:

1. **Set up OpenRouter API key** in your `.env` file (see [.env.example](.env.example))
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Run inference/evaluation** using the model IDs from OpenRouter

**Available Models**: The system automatically discovers models from providers like OpenAI, Anthropic, Google, Meta, Mistral, and many more. See [OpenRouter Integration Guide](OPENROUTER_README.md) for details.

Inference can be run with the following command:
```bash
# Using OpenRouter models (recommended)
python3 inference.py -t stack-eval -m openai/gpt-4o
python3 inference.py -t stack-unseen -m anthropic/claude-3-5-sonnet

# Legacy models (if OpenRouter is unavailable)
python3 inference.py -t stack-eval -m gpt-4-turbo-2024-04-09
```

Evaluation can be run with the following command:
```bash
# Using OpenRouter models (recommended)
python3 evaluation.py -t stack-eval -e openai/gpt-4o -j anthropic/claude-3-5-sonnet -p eval-cot-ref
python3 evaluation.py -t stack-unseen -e openai/gpt-4o-mini -j openai/gpt-4o -p eval-cot-ref

# Legacy models (if OpenRouter is unavailable)
python3 evaluation.py -t stack-eval -e claude-3-5-sonnet -j gpt-4-turbo-2024-04-09 -p eval-cot-ref
```

This will generate scores for each question on a `0-3` scale. A completion is said to be acceptable, if it's score is greater than or equal to `2`.

**Note**: Model names now use the format `provider/model-id` (e.g., `openai/gpt-4o`, `anthropic/claude-3-5-sonnet`).

### Dashboard
We provide a comprehensive Streamlit dashboard that combines coding tasks exploration with live evaluation score tracking. The dashboard can be launched with the following command:
```
streamlit run dashboard.py
```

**Features:**
- **Live Evaluation Scores**: Real-time display of evaluation metrics based on selected models and tasks
- **Model Selection**: Choose from cached OpenRouter models for both inferencer/evaluatee and judge/evaluator roles
- **Task Selection**: Switch between stack-eval and stack-unseen tasks
- **Coding tasks**: Browse, filter, and search through the coding tasks dataset
- **Interactive Charts**: Visualize task distributions by type, complexity, and programming language
- **Random task Generator**: Get random tasks for practice or testing

**Model Selection:**
The dashboard automatically loads available models from `config/openrouter_models_cache.yml` and provides:
- **Inferencer/Evaluatee Model**: The model that generates solutions to be evaluated
- **Judge/Evaluator Model**: The model that evaluates the generated solutions
- **Task Selection**: Choose between stack-eval and stack-unseen evaluation tasks

**Live Updates:**
Evaluation scores update automatically based on the output folder structure and JSONL files, providing real-time insights into model performance.

