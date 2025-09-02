import os
from typing import List, Optional, Set

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils import LLMS, PROMPTS, TASKS, batch_format_prompt, completion, setup_logger
import os

logger = setup_logger(__name__)

def _filter_by_metadata(df: pd.DataFrame,
                        types: Optional[List[str]] = None,
                        levels: Optional[List[str]] = None,
                        tags: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filter a raw tasks dataframe (with questionMetadata column) by optional
    type/level/tag lists. If a list is empty/None, that dimension is not filtered.
    """
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


def _load_existing_inference_ids(output_path: str) -> Set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        existing = pd.read_json(output_path, lines=True)
        if "questionId" not in existing.columns:
            return set()
        # Consider a task solved if completion is a non-empty string
        solved = existing[existing.get("completion", "").astype(str).str.len() > 0]
        return set(solved["questionId"].astype(str).tolist())
    except Exception as e:
        logger.warning(f"Failed to read existing inference file {output_path}: {e}")
        return set()


def main(task: str,
         model_name: str,
         types: Optional[List[str]] = None,
         levels: Optional[List[str]] = None,
         tags: Optional[List[str]] = None,
         skip_existing: bool = True):
    """
    Run inference for the specified dataset/model. Optionally restrict to a subset
    of tasks by type/level/tag, and skip tasks already solved in the output file.
    """
    raw = pd.read_json(TASKS[task], lines=True)
    data = _filter_by_metadata(raw, types=types, levels=levels, tags=tags).reset_index(drop=True)
    if data.empty:
        logger.info("No tasks match the provided filters. Nothing to do.")
        return

    if model_name not in LLMS:
        raise ValueError(f"Model {model_name} not found in the configuration.")
    model = LLMS[model_name]

    output_path = f"output/{task}/inf/{model_name}.jsonl"

    # Optionally skip tasks already solved
    already_solved_ids: Set[str] = _load_existing_inference_ids(output_path) if skip_existing else set()
    subset_ids: Set[str] = set(data["questionId"].astype(str).tolist())
    to_run_ids = list(subset_ids - already_solved_ids)

    if skip_existing and len(to_run_ids) == 0:
        logger.info("All %d selected tasks already have completions for %s. Skipping.", len(subset_ids), model_name)
        return

    run_df = data[data["questionId"].astype(str).isin(to_run_ids)] if to_run_ids else data

    prompt = PROMPTS["stack-inf"]
    messages = batch_format_prompt(prompt, run_df)
    logger.info("Loaded %s messages for inference on task %s (selected=%d, remaining=%d).",
                len(messages), task, len(subset_ids), len(run_df))

    logger.info("Launching batch completion using model %s...", model_name)
    # Allow override timeout via env
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    completions = completion(
        messages,
        custom_llm_provider=model['custom_llm_provider'],
        **model['model_parameters'], **model['sample_parameters'],
        num_retries=3, timeout=request_timeout
    )
    run_df = run_df.copy()
    run_df["completion"] = completions
    empty_completions = (pd.Series(completions).astype(str).str.len() == 0).sum()
    if empty_completions > 0:
        logger.warning("Found %s empty completions.", empty_completions)
    run_df["model"] = [model["model_parameters"]["model"] for _ in range(len(run_df))]
    run_df["modelMetadata"] = [model["model_metadata"] for _ in range(len(run_df))]

    # Merge with any existing outputs (keep previous rows for other IDs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        try:
            existing = pd.read_json(output_path, lines=True)
        except Exception:
            existing = pd.DataFrame()
        combined = pd.concat([existing, run_df], ignore_index=True)
        # De-duplicate on questionId, prefer latest results
        combined = combined.drop_duplicates(subset=["questionId"], keep="last")
        combined.to_json(output_path, lines=True, orient="records")
    else:
        run_df.to_json(output_path, lines=True, orient="records")
    logger.info("Saved completions to %s", output_path)


if __name__ == "__main__":
    import argparse

    #fmt: off
    parser = argparse.ArgumentParser(description="Run inference on a task using a specified model.")
    parser.add_argument("--task", "-t", required=True, type=str, choices=TASKS.keys(), help="The task to run inference on.")
    parser.add_argument("--model", "-m", required=True, type=str, choices=LLMS.keys(), help="The model to use for inference.")
    parser.add_argument("--types", "-T", nargs="*", default=None, help="Optional list of task types to include (space-separated).")
    parser.add_argument("--levels", "-L", nargs="*", default=None, help="Optional list of task levels to include (space-separated).")
    parser.add_argument("--tags", "-G", nargs="*", default=None, help="Optional list of programming language tags to include (space-separated).")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not skip tasks that already have completions; re-generate.")
    args = parser.parse_args()
    #fmt: on

    main(
        args.task,
        args.model,
        types=args.types,
        levels=args.levels,
        tags=args.tags,
        skip_existing=not args.no_skip_existing,
    )
