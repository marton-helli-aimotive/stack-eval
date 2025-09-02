import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils import (
    LLMS,
    PROMPTS,
    TASKS,
    batch_format_prompt,
    batch_parse_json,
    completion,
    setup_logger,
)
import os

logger = setup_logger(__name__)


def main(task: str, prompt_name: str, evaluatee_model: str, evaluator_model: str):
    prompt = PROMPTS[prompt_name]
    input_path = f"output/{task}/inf/{evaluatee_model}.jsonl"
    if not os.path.exists(input_path):
        raise ValueError(
            f"Inference for {evaluatee_model} has not been run on task {task}."
        )

    evaluatee_data = pd.read_json(input_path, lines=True)
    messages = batch_format_prompt(PROMPTS[prompt_name], evaluatee_data)
    logger.info("Loaded %s messages for inference on task %s.", len(messages), task)

    if evaluator_model not in LLMS:
        raise ValueError(f"Model {evaluator_model} not found in the configuration.")
    logger.info("Launching batch completion using model %s...", evaluator_model)

    model = LLMS[evaluator_model]
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    completions = completion(
        messages,
        custom_llm_provider=model['custom_llm_provider'],
        **model['model_parameters'], **model['sample_parameters'],
        num_retries=3, timeout=request_timeout
    )
    parsed_completions = batch_parse_json(completions, expected_keys=prompt["output"])
    parsed_completions_df = pd.DataFrame(parsed_completions)

    empty_evaluations = parsed_completions_df.isna().any(axis=1).sum()
    if empty_evaluations > 0:
        logger.warning("Empty evaluations found: %s.", empty_evaluations)
        parsed_completions_df = _fillna(parsed_completions_df)
    evaluatee_data = pd.concat([evaluatee_data, parsed_completions_df], axis=1)

    output_path = f"output/{task}/evl/{evaluator_model}/{prompt_name}/{evaluatee_model}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    evaluatee_data.to_json(output_path, lines=True, orient="records")

    logger.info("Saved evaluations to %s", output_path)


def _fillna(df: pd.DataFrame):
    for c in df.columns:
        if c.endswith("Score"):
            df[c] = df[c].fillna(0).astype(int)
        if c.endswith("Evaluation"):
            df[c] = df[c].fillna("")
    return df


if __name__ == "__main__":
    import argparse

    #fmt: off
    parser = argparse.ArgumentParser(description="Run evaluation on a task using a specified model.")
    parser.add_argument("--task", "-t", required=True, type=str, choices=TASKS.keys(), help="The task to run evaluation on.")
    parser.add_argument("--prompt", "-p", default="eval-cot-ref", type=str, choices=[p for p in PROMPTS.keys() if p.startswith("eval")], help="The prompt to use for evaluation.")
    parser.add_argument("--evaluatee", "-e", required=True, type=str, choices=LLMS.keys(), help="The model whose output will be evaluated.")
    parser.add_argument("--evaluator", "-j", default="gpt-4-turbo-2024-04-09", type=str, choices=LLMS.keys(), help="The model to use for evaluation.")
    args = parser.parse_args()
    #fmt: on

    main(args.task, args.prompt, args.evaluatee, args.evaluator)
