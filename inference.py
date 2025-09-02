import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from utils import LLMS, PROMPTS, TASKS, batch_format_prompt, completion, setup_logger
import os

logger = setup_logger(__name__)


def main(task: str, model_name: str):
    data = pd.read_json(TASKS[task], lines=True)
    prompt = PROMPTS["stack-inf"]
    messages = batch_format_prompt(prompt, data)
    logger.info("Loaded %s messages for inference on task %s.", len(messages), task)

    if model_name not in LLMS:
        raise ValueError(f"Model {model_name} not found in the configuration.")
    model = LLMS[model_name]
    logger.info("Launching batch completion using model %s...", model_name)

    # Allow override timeout via env
    request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "120"))
    data["completion"] = completion(
        messages,
        custom_llm_provider=model['custom_llm_provider'],
        **model['model_parameters'], **model['sample_parameters'],
        num_retries=3, timeout=request_timeout
    )
    empty_completions = (data["completion"].str.len() == 0).sum()
    if empty_completions > 0:
        logger.warning("Found %s empty completions.", empty_completions)
    data["model"] = [model["model_parameters"]["model"] for _ in range(len(data))]
    data["modelMetadata"] = [model["model_metadata"] for _ in range(len(data))]

    output_path = f"output/{task}/inf/{model_name}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_json(output_path, lines=True, orient="records")
    logger.info("Saved completions to %s", output_path)


if __name__ == "__main__":
    import argparse

    #fmt: off
    parser = argparse.ArgumentParser(description="Run inference on a task using a specified model.")
    parser.add_argument("--task", "-t", required=True, type=str, choices=TASKS.keys(), help="The task to run inference on.")
    parser.add_argument( "--model", "-m", required=True, type=str, choices=LLMS.keys(), help="The model to use for inference.")
    args = parser.parse_args()
    #fmt: on

    main(args.task, args.model)
