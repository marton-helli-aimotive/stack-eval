import os
from typing import List, Optional, Set

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
def _filter_by_metadata(df: pd.DataFrame,
                        types: Optional[List[str]] = None,
                        levels: Optional[List[str]] = None,
                        tags: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    mask = pd.Series([True] * len(df))
    if types:
        mask &= df["questionMetadata"].apply(lambda m: m.get("type") in set(types))
    if levels:
        mask &= df["questionMetadata"].apply(lambda m: m.get("level") in set(levels))
    if tags:
        mask &= df["questionMetadata"].apply(lambda m: m.get("tag") in set(tags))
    return df[mask]


def _load_existing_eval_ids(output_path: str) -> Set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_json(output_path, lines=True)
        if "questionId" not in existing.columns:
            return set()
        # Consider evaluated if acceptabilityScore present (any type) and not NaN
        if "acceptabilityScore" in existing.columns:
            has_score = existing["acceptabilityScore"].notna()
            return set(existing.loc[has_score, "questionId"].astype(str).tolist())
        return set(existing["questionId"].astype(str).tolist())
    except Exception as e:
        logger.warning(f"Failed to read existing evaluation file {output_path}: {e}")
        return set()


def main(task: str,
         prompt_name: str,
         evaluatee_model: str,
         evaluator_model: str,
         types: Optional[List[str]] = None,
         levels: Optional[List[str]] = None,
         tags: Optional[List[str]] = None,
         skip_existing: bool = True):
    prompt = PROMPTS[prompt_name]
    input_path = f"output/{task}/inf/{evaluatee_model}.jsonl"
    if not os.path.exists(input_path):
        raise ValueError(
            f"Inference for {evaluatee_model} has not been run on task {task}."
        )

    evaluatee_all = pd.read_json(input_path, lines=True)
    evaluatee_data = _filter_by_metadata(evaluatee_all, types=types, levels=levels, tags=tags).reset_index(drop=True)
    if evaluatee_data.empty:
        logger.info("No tasks match the provided filters. Nothing to evaluate.")
        return

    output_path = f"output/{task}/evl/{evaluator_model}/{prompt_name}/{evaluatee_model}.jsonl"
    selected_ids: Set[str] = set(evaluatee_data["questionId"].astype(str).tolist())
    already_evaluated_ids: Set[str] = _load_existing_eval_ids(output_path) if skip_existing else set()
    to_run_ids = list(selected_ids - already_evaluated_ids)

    if skip_existing and len(to_run_ids) == 0:
        logger.info("All %d selected tasks already evaluated by %s for %s. Skipping.", len(selected_ids), evaluator_model, prompt_name)
        return

    run_df = evaluatee_data[evaluatee_data["questionId"].astype(str).isin(to_run_ids)] if to_run_ids else evaluatee_data

    messages = batch_format_prompt(PROMPTS[prompt_name], run_df)
    logger.info("Loaded %s messages for evaluation on task %s (selected=%d, remaining=%d).",
                len(messages), task, len(selected_ids), len(run_df))

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
    run_df = pd.concat([run_df, parsed_completions_df], axis=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        try:
            existing = pd.read_json(output_path, lines=True)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, run_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["questionId"], keep="last")
        combined.to_json(output_path, lines=True, orient="records")
    else:
        run_df.to_json(output_path, lines=True, orient="records")

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
    parser.add_argument("--types", "-T", nargs="*", default=None, help="Optional list of task types to include (space-separated).")
    parser.add_argument("--levels", "-L", nargs="*", default=None, help="Optional list of task levels to include (space-separated).")
    parser.add_argument("--tags", "-G", nargs="*", default=None, help="Optional list of programming language tags to include (space-separated).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip tasks already evaluated; re-run them.")
    args = parser.parse_args()
    #fmt: on

    main(
        args.task,
        args.prompt,
        args.evaluatee,
        args.evaluator,
        types=args.types,
        levels=args.levels,
        tags=args.tags,
        skip_existing=not args.no_skip_existing,
    )
