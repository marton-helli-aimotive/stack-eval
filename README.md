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
We provide our inference and evaluation code along with the corresponding prompts. To run inference/evaluations, ensure that the API key for the model are defined in a `.env` file. Please refer to the list of llms in the [config file](./config/llms.yml) for available LLMs.

Inference can be run with the following command:
```
python3 inference.py -t stack-eval -m gpt-4-turbo-2024-04-09
python3 inference.py -t TASK_TYPE -m MODEL_ID
```
Evaluation can be run with the following command:
```
python3 evaluation.py -t stack-eval -e claude-3.5-sonnet -j gpt-4-turbo-2024-04-09 -p eval-cot-ref
python3 evaluation.py -t TASK_TYPE -e EVALUATEE -j EVALUATOR -p PROMPT_NAME
```
This will generate scores for each question on a `0-3` scale. A completion is said to be acceptable, if it's score is greater than or equal to `2`.

### Dashboard
We also provided a streamlit app to view and analyze the evaluation results, which can be launched with the following command:
```
streamlit run dashboard.py
```
Ensure that the `STACK_EVAL_DIR` and `STACK_UNSEEN_DIR` constants in [dashboard.py](./dashboard.py) are correctly pointed to the evaluation output directory, and at least one evaluation is present.

---

### Contact

Please contact Nidhish Shah `nidhish.shah[at]prosus[dot]com`, Zulkuf Genc `zulkuf.genc[at]prosus[dot]com` and Dogu Araci `dogu.araci[at]prosus[dot]com` about any StackEval related issues and questions.
