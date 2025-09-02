import streamlit as st
import pandas as pd
import altair as alt
import json
import yaml
import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import modules so their logs print to the original terminal when called directly
import inference as inference_module
import evaluation as evaluation_module

load_dotenv()

from utils import setup_logger  # noqa: E402

logger = setup_logger(__name__)



# Page configuration
st.set_page_config(
    page_title="Coding Tasks & Evaluation Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_openrouter_models_cache() -> Dict[str, Any]:
    """
    Load the cached OpenRouter models from YAML file.
    """
    try:
        with open("config/openrouter_models_cache.yml", 'r', encoding='utf-8') as file:
            models = yaml.safe_load(file)
        return models
    except Exception as e:
        logger.error(f"Error loading OpenRouter models cache: {e}")
        return {}


@st.cache_data
def load_openwebui_models_cache() -> Dict[str, Any]:
    """
    Load the cached OpenWebUI models from YAML file.
    """
    try:
        with open("config/openwebui_models_cache.yml", 'r', encoding='utf-8') as file:
            models = yaml.safe_load(file)
        return models
    except Exception as e:
        logger.error(f"Error loading OpenWebUI models cache: {e}")
        return {}

@st.cache_data(ttl=5)
def load_evaluation_data(path: str) -> pd.DataFrame:
    """
    Load the evaluation data from the given path.
    
    Args:
        path: The path to the evaluation data file.
        
    Returns:
        DataFrame with evaluation data
    """
    # Check if the path is a file or directory
    if os.path.isfile(path):
        # Single file
        files = [path]
    else:
        # Directory - search recursively for .jsonl files
        files = glob.glob(f"{path}/**/*.jsonl", recursive=True)
    
    evals = []
    for file in files:
        try:
            df = pd.read_json(file, lines=True)[['questionId', 'questionMetadata', 'model', 'modelMetadata', 'acceptabilityScore']]
            # Coerce None to empty dicts before .get usage
            df['questionMetadata'] = df['questionMetadata'].apply(lambda v: v if isinstance(v, dict) and v is not None else {})
            df['modelMetadata'] = df['modelMetadata'].apply(lambda v: v if isinstance(v, dict) and v is not None else {})
            # Acceptance flag
            df['acceptance'] = (df['acceptabilityScore'] >= 2).astype(int)
            
            # Extract metadata fields safely
            df['type'] = df['questionMetadata'].apply(lambda x: x.get('type', ''))
            df['level'] = df['questionMetadata'].apply(lambda x: x.get('level', ''))
            df['tag'] = df['questionMetadata'].apply(lambda x: x.get('tag', ''))
            
            # Extract model metadata safely
            df['name'] = df['modelMetadata'].apply(lambda x: x.get('name', ''))
            df['provider'] = df['modelMetadata'].apply(lambda x: x.get('provider', ''))
            
            # Drop rows without a valid questionId (can occur if a partial/invalid line is saved)
            df = df.dropna(subset=['questionId'])
            # Drop the original metadata columns
            df = df.drop(['questionMetadata', 'modelMetadata', 'acceptabilityScore'], axis=1)
            evals.append(df)
        except Exception as e:
            # During active runs files may be incomplete; avoid spamming warnings
            if st.session_state.get("is_running", False):
                logger.debug(f"Error loading file {file}: {e}")
            else:
                logger.warning(f"Error loading file {file}: {e}")
            continue
    
    if len(evals) == 0:
        if st.session_state.get("is_running", False):
            logger.debug("No evaluation data found in %s.", path)
        else:
            logger.warning("No evaluation data found in %s.", path)
        return pd.DataFrame()
    
    if st.session_state.get("is_running", False):
        logger.debug("Loaded %s evaluation data from %s.", len(evals), path)
    else:
        logger.info("Loaded %s evaluation data from %s.", len(evals), path)
    return pd.concat(evals).reset_index(drop=True)



@st.cache_data
def load_coding_tasks(file_path: str) -> pd.DataFrame:
    """
    Load coding tasks from JSONL file and convert to DataFrame.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame with coding tasks
    """
    tasks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    task = json.loads(line)
                    # Extract metadata
                    metadata = task.get('questionMetadata', {})
                    tasks.append({
                        'questionId': task.get('questionId', ''),
                        'question': task.get('question', ''),
                        'answer': task.get('answer', ''),
                        'type': metadata.get('type', ''),
                        'level': metadata.get('level', ''),
                        'tag': metadata.get('tag', '')
                    })
        
        df = pd.DataFrame(tasks)
        # Clean up empty values
        df = df.replace('', pd.NA).dropna(subset=['question', 'type', 'level', 'tag'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_task_display(task: Dict[str, Any], include_answer: bool = True) -> str:
    """
    Format a task for human-readable display.
    
    Args:
        task: Dictionary containing task data
        
    Returns:
        Formatted string representation
    """
    formatted = f"""
## Task ID: {task['questionId']}

**Type:** {task['type'].title()}
**Complexity:** {task['level'].title()}
**Language:** {task['tag'].title()}

### Question:
{task['question']}
"""
    if include_answer:
        formatted += f"""

### Answer:
{task['answer']}
"""
    return formatted

def get_evaluation_score(task: str, inferencer_model: str, judge_model: str, 
                        selected_types: List[str] = None, selected_levels: List[str] = None, 
                        selected_tags: List[str] = None) -> Dict[str, Any]:
    """
    Get the evaluation score for the selected combination, optionally filtered by task criteria.
    
    Args:
        task: The selected task (stack-eval or stack-unseen)
        inferencer_model: The selected inferencer/evaluatee model
        judge_model: The selected judge/evaluator model
        selected_types: List of selected question types to filter by
        selected_levels: List of selected question levels to filter by
        selected_tags: List of selected question tags to filter by
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Construct the expected path based on the actual file structure
    # Path format: output/{task}/evl/{judge_model}/eval-cot-ref/{inferencer_model}.jsonl
    base_path = _evaluation_output_path(task, inferencer_model, judge_model)
    
    # Try to load evaluation data
    try:
        df = load_evaluation_data(base_path)
        if not df.empty:
            # Apply filters if provided
            if selected_types and selected_levels and selected_tags:
                filtered_df = df[
                    (df['type'].isin(selected_types)) &
                    (df['level'].isin(selected_levels)) &
                    (df['tag'].isin(selected_tags))
                ]
            else:
                filtered_df = df
            
            # Calculate acceptance rate for filtered data
            total_questions = len(filtered_df)
            accepted_questions = filtered_df['acceptance'].sum()
            acceptance_rate = (accepted_questions / total_questions) * 100 if total_questions > 0 else 0
            
            return {
                'total_questions': total_questions,
                'accepted_questions': accepted_questions,
                'acceptance_rate': acceptance_rate,
                'data_loaded': True,
                'filtered': selected_types and selected_levels and selected_tags
            }
    except Exception as e:
        logger.warning(f"Could not load evaluation data for {base_path}: {e}")
    
    return {
        'total_questions': 0,
        'accepted_questions': 0,
        'acceptance_rate': 0,
        'data_loaded': False,
        'filtered': False
    }


def _inference_output_path(task: str, inferencer_model: str) -> str:
    return f"output/{task}/inf/{inferencer_model}.jsonl"


def _evaluation_output_path(task: str, inferencer_model: str, judge_model: str, prompt_name: str = "eval-cot-ref") -> str:
    return f"output/{task}/evl/{judge_model}/{prompt_name}/{inferencer_model}.jsonl"

@st.cache_data(ttl=5)
def load_inference_answers(task: str, inferencer_model: str) -> Dict[str, str]:
    """
    Load a mapping from questionId to the model's completion (answer) for a given
    task and inferencer model. Returns an empty dict if file is missing.
    """
    answers: Dict[str, str] = {}
    if not inferencer_model:
        return answers
    path = _inference_output_path(task, inferencer_model)
    if not os.path.exists(path):
        return answers
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("questionId")
                    comp = rec.get("completion")
                    if qid is not None:
                        # Store even empty strings to mark presence; caller can check emptiness
                        answers[qid] = comp if isinstance(comp, str) else ""
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"Failed to load inference answers from {path}: {e}")
    return answers

@st.cache_data(ttl=5)
def load_evaluation_results(task: str, inferencer_model: str, judge_model: str, prompt_name: str = "eval-cot-ref") -> Dict[str, Dict[str, Any]]:
    """
    Load mapping: questionId -> { "acceptabilityScore": int | None, "acceptabilityEvaluation": str | None }
    for a given dataset, inferencer model and judge model.
    """
    results: Dict[str, Dict[str, Any]] = {}
    if not (inferencer_model and judge_model):
        return results
    path = _evaluation_output_path(task, inferencer_model, judge_model, prompt_name)
    if not os.path.exists(path):
        return results
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    qid = rec.get("questionId")
                    score = rec.get("acceptabilityScore")
                    eval_txt = rec.get("acceptabilityEvaluation")
                    results[qid] = {
                        "acceptabilityScore": int(score) if isinstance(score, (int, float, str)) and str(score).isdigit() else None,
                        "acceptabilityEvaluation": eval_txt if isinstance(eval_txt, str) else None,
                    }
                except Exception:
                    continue
    except Exception as e:
        logger.debug(f"Failed to load evaluation results from {path}: {e}")
    return results

@st.cache_data(ttl=5)
def list_evaluated_models(task: str, judge_model: str, prompt_name: str = "eval-cot-ref") -> List[str]:
    """
    List all inferencer models that have been evaluated for a given dataset and judge.

    Returns model ids in the same format used elsewhere.
    """
    base_dir = f"output/{task}/evl/{judge_model}/{prompt_name}"
    if not os.path.exists(base_dir):
        return []

    files = glob.glob(f"{base_dir}/**/*.jsonl", recursive=True)
    models = set()
    for file in files:
        try:
            rel = os.path.relpath(file, base_dir)
            if rel.endswith(".jsonl"):
                model_id = rel[: -len(".jsonl")].replace(os.sep, "/")
                models.add(model_id)
        except Exception:
            continue
    return sorted(models)

@st.cache_data(ttl=5)
def load_inference_metadata(task: str, inferencer_model: str) -> Dict[str, Any]:
    """
    Attempt to load basic inference metadata for an inferencer model.
    Looks for context length, provider/name from modelMetadata, and any optional
    cost/duration fields if present in the inference outputs. If not present, returns None values.
    """
    path = _inference_output_path(task, inferencer_model)
    meta = {"context_length": None, "provider": None, "name": None, "est_cost": None, "duration_sec": None}
    if not os.path.exists(path):
        return meta
    try:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if first_line:
            record = json.loads(first_line)
            mm = record.get("modelMetadata", {}) or {}
            meta["context_length"] = mm.get("context_length")
            meta["name"] = mm.get("name")
            # Prefer deriving provider from model id to avoid cache inconsistencies
            meta["provider"] = (inferencer_model.split("/")[0] if "/" in inferencer_model else mm.get("provider"))
            # Optional usage/cost/duration fields if present
            # (Not currently emitted by inference, but future-compatible.)
            for k in ["est_cost", "cost", "estimated_cost"]:
                if k in record:
                    meta["est_cost"] = record.get(k)
                    break
            for k in ["duration", "elapsed", "latency_seconds", "time", "elapsed_sec"]:
                if k in record:
                    try:
                        meta["duration_sec"] = float(record.get(k))
                    except Exception:
                        meta["duration_sec"] = None
                    break
    except Exception as e:
        logger.debug(f"Could not load inference metadata for {inferencer_model}: {e}")
    return meta

def main():
    st.title("Coding Tasks & Evaluation Dashboard")
    st.markdown("Filter and explore coding tasks by type, complexity, and programming language, plus view live evaluation scores.")
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False
    # Top-of-page status notice while long-running tasks execute
    top_notice = st.empty()
    if st.session_state.get("is_running", False):
        top_notice.info("⏳ Inference/Evaluation is running. Check the terminal for detailed progress logs.")
    
    # Load model caches
    openrouter_models = load_openrouter_models_cache()
    openwebui_models = load_openwebui_models_cache()
    
    # Top left controls for LLM selection and task
    st.sidebar.header("🎯 Task Selection")
    
    # Task selector
    task_options = ["stack-eval-mini", "stack-eval", "stack-unseen"]
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=task_options,
        index=task_options.index("stack-eval"),
        help="Choose between stack-eval and stack-unseen tasks",
        disabled=st.session_state.get("is_running", False)
    )
    # Clear any previously selected random task when dataset changes
    if "prev_selected_task" not in st.session_state:
        st.session_state["prev_selected_task"] = selected_dataset
    elif st.session_state["prev_selected_task"] != selected_dataset:
        st.session_state["prev_selected_task"] = selected_dataset
        if "random_task" in st.session_state:
            del st.session_state["random_task"]
    
    # Load coding tasks data based on selected task (needed for filters)
    if selected_dataset == "stack-eval":
        data_files = ["data/stack-eval.jsonl"]
    elif selected_dataset == "stack-eval-mini":
        data_files = ["data/stack-eval-mini.jsonl"]
    else:  # stack-unseen
        data_files = sorted(glob.glob("data/stack-unseen*.jsonl"))

    dfs = []
    for path in data_files:
        if os.path.exists(path):
            part = load_coding_tasks(path)
            if not part.empty:
                dfs.append(part)
    df = pd.concat(dfs).reset_index(drop=True) if len(dfs) > 0 else pd.DataFrame()
    
    if df.empty:
        st.error("No coding tasks data loaded. Please check the file path.")
        return
    
    # Task filters directly under Task selection
    st.sidebar.header("🔍 Filters")
    
    # Question type filter
    task_types = sorted(df['type'].unique())
    selected_types = st.sidebar.multiselect(
        "Task Type",
        options=task_types,
        default=task_types,
        help="Select one or more question types to filter by",
        disabled=st.session_state.get("is_running", False)
    )
    
    # Complexity level filter
    complexity_levels = sorted(df['level'].unique())
    selected_levels = st.sidebar.multiselect(
        "Complexity Level",
        options=complexity_levels,
        default=complexity_levels,
        help="Select one or more complexity levels to filter by",
        disabled=st.session_state.get("is_running", False)
    )
    
    # Programming language filter
    programming_languages = sorted(df['tag'].unique())
    selected_languages = st.sidebar.multiselect(
        "Programming Language",
        options=programming_languages,
        default=programming_languages,
        help="Select one or more programming languages to filter by",
        disabled=st.session_state.get("is_running", False)
    )
    
    # Apply filters
    if selected_types and selected_levels and selected_languages:
        filtered_df = df[
            (df['type'].isin(selected_types)) &
            (df['level'].isin(selected_levels)) &
            (df['tag'].isin(selected_languages))
        ]
    else:
        filtered_df = df
    
    # Separator before model selection
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Model Selection")
    
    # API selectors
    api_options = ["OpenRouter", "OpenWebUI by aiMotive"]

    def get_default_api_index() -> int:
        # Prefer OpenWebUI by default to avoid external dependency pitfalls
        return 1 if len(api_options) > 1 else 0
    
    def get_models_for_api(api_name: str) -> List[str]:
        cache = openrouter_models if api_name == "OpenRouter" else openwebui_models
        models = []
        if cache:
            for model_id in cache.keys():
                if model_id not in ['_cache_time', 'default', 'default-json']:
                    models.append(model_id)
        models.sort()
        return models
    
    def get_default_model_index(api_name: str, models: List[str]) -> int:
        """Get the index of the preferred default model for the given API"""
        if api_name == "OpenRouter":
            # Prefer qwen/qwen3-coder-30b-a3b-instruct if available
            preferred = "qwen/qwen3-coder-30b-a3b-instruct"
        else:  # OpenWebUI
            # Prefer openwebui/qwen3-coder:30b if available
            preferred = "openwebui/qwen3-coder:30b"
        
        try:
            return models.index(preferred)
        except ValueError:
            return 0  # Fallback to first model
    
    # Inferencer container
    with st.sidebar.container():
        st.markdown("**🧠 Solver**")
        inferencer_api = st.selectbox(
            "API",
            options=api_options,
            index=get_default_api_index(),
            help="Choose which API/provider to use for the inferencer",
            disabled=st.session_state.get("is_running", False)
        )
        
        inferencer_models = get_models_for_api(inferencer_api)
        if inferencer_models:
            inferencer_model = st.selectbox(
                "Model",
                options=inferencer_models,
                index=get_default_model_index(inferencer_api, inferencer_models),
                help="Select the model that generates solutions to be evaluated",
                disabled=st.session_state.get("is_running", False)
            )
        else:
            st.warning(
                "No models found for the selected Inferencer API. Ensure the respective cache YAML exists."
            )
            inferencer_model = None
    
    # Judge container
    with st.sidebar.container():
        st.markdown("**⚖️ Judge**")
        judge_api = st.selectbox(
            "API",
            options=api_options,
            index=get_default_api_index(),
            help="Choose which API/provider to use for the judge",
            disabled=st.session_state.get("is_running", False)
        )
        
        judge_models = get_models_for_api(judge_api)
        if judge_models:
            judge_model = st.selectbox(
                "Model",
                options=judge_models,
                index=get_default_model_index(judge_api, judge_models),
                help="Select the model that evaluates the generated solutions",
                disabled=st.session_state.get("is_running", False)
            )
        else:
            st.warning(
                "No models found for the selected Judge API. Ensure the respective cache YAML exists."
            )
            judge_model = None
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📋 Coding Tasks", "🎲 Random Task", "🏆 Leaderboard"])
    
    # Tab 1: Coding Tasks
    with tab1:                
        # Live status and evaluation (updated with filters)
        subset_qids = set(filtered_df['questionId'].astype(str).tolist()) if not filtered_df.empty else set()
        answers_map = load_inference_answers(selected_dataset, inferencer_model) if inferencer_model else {}
        eval_map = load_evaluation_results(selected_dataset, inferencer_model, judge_model) if (inferencer_model and judge_model) else {}
        have_inf_ids = {str(qid) for qid, comp in answers_map.items() if isinstance(comp, str) and len(comp) > 0}
        have_eval_ids = {str(qid) for qid, rec in eval_map.items() if rec and rec.get('acceptabilityScore') is not None}
        missing_inference_ids = list(subset_qids - have_inf_ids)
        missing_eval_ids = list((subset_qids & have_inf_ids) - have_eval_ids)

        # Show inference-only status if judge not selected
        if inferencer_model and not judge_model:
            subset_total = len(subset_qids)
            inf_count = len(subset_qids & have_inf_ids)
            if subset_total == 0:
                st.warning("No tasks match the current filters.")
            else:
                if inf_count == 0:
                    st.warning(f"No inference results for current filters using {inferencer_model}.")
                elif inf_count < subset_total:
                    st.warning(f"Inference partial: {inf_count}/{subset_total} tasks have model answers.")
                else:
                    st.success(f"Inference complete: {inf_count}/{subset_total} tasks have model answers.")

        # Live evaluation and inference status synced with filters (3-state messaging)
        if inferencer_model and judge_model:
            st.subheader("📊 Live Evaluation Score")

            subset_total = len(subset_qids)
            inf_count = len(subset_qids & have_inf_ids)
            eval_count = len(subset_qids & have_eval_ids)

            if subset_total == 0:
                st.warning("No tasks match the current filters.")
            else:
                # Inference status (3 states)
                if inf_count == 0:
                    st.warning(f"No inference results for current filters using {inferencer_model}.")
                elif inf_count < subset_total:
                    st.warning(f"Inference partial: {inf_count}/{subset_total} tasks have model answers.")
                else:
                    st.success(f"Inference complete: {inf_count}/{subset_total} tasks have model answers.")

                # Evaluation status (3 states)
                if eval_count == 0:
                    st.warning(f"No evaluation results for current filters using {judge_model}.")
                elif eval_count < subset_total:
                    st.warning(f"Evaluation partial: {eval_count}/{subset_total} tasks have judge scores.")
                else:
                    st.success(f"Evaluation complete: {eval_count}/{subset_total} tasks have judge scores.")

                # Show metrics only when evaluation fully covers the current subset
                if eval_count == subset_total and subset_total > 0:
                    evaluation_score = get_evaluation_score(
                        selected_dataset, inferencer_model, judge_model,
                        selected_types, selected_levels, selected_languages
                    )

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Task", selected_dataset.replace("-", " ").title())
                    with col2:
                        st.metric("Total Questions", evaluation_score['total_questions'])
                    with col3:
                        st.metric("Accepted Solutions", evaluation_score['accepted_questions'])
                    with col4:
                        st.metric("Acceptance Rate", f"{evaluation_score['acceptance_rate']:.1f}%")
        
        # Action buttons for running inference/evaluation
            st.markdown("---")
            col_run1, col_run2 = st.columns(2)
            with col_run1:
                run_inf_disabled = (
                    st.session_state.get("is_running", False)
                    or inferencer_model is None
                    or len(missing_inference_ids) == 0
                )
                if st.button("🧠 Solve tasks", disabled=run_inf_disabled):
                    try:
                        st.session_state["is_running"] = True
                        # Call directly so logs appear in the terminal running Streamlit
                        with st.spinner("Running inference... check the terminal for detailed progress."):
                            inference_module.main(
                                selected_dataset,
                                inferencer_model,
                                types=selected_types,
                                levels=selected_levels,
                                tags=selected_languages,
                                skip_existing=True,
                            )
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                    finally:
                        try:
                            load_inference_answers.clear()
                        except Exception:
                            pass
                        st.session_state["is_running"] = False
                        st.rerun()
            with col_run2:
                run_eval_disabled = (
                    st.session_state.get("is_running", False)
                    or inferencer_model is None
                    or judge_model is None
                    or len(missing_eval_ids) == 0
                )
                if st.button("⚖️ Judge solutions", disabled=run_eval_disabled):
                    try:
                        st.session_state["is_running"] = True
                        # Use default prompt name used across the app
                        with st.spinner("Running evaluation... check the terminal for detailed progress."):
                            evaluation_module.main(
                                selected_dataset,
                                "eval-cot-ref",
                                inferencer_model,
                                judge_model,
                                types=selected_types,
                                levels=selected_levels,
                                tags=selected_languages,
                                skip_existing=True,
                            )
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
                    finally:
                        # Clear cached eval data so Live score refreshes
                        try:
                            load_evaluation_data.clear()
                        except Exception:
                            pass
                        try:
                            load_evaluation_results.clear()
                        except Exception:
                            pass
                        st.session_state["is_running"] = False
                        st.rerun()
        
        # Display statistics
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Total Tasks", len(df))
        
        with col_stats2:
            st.metric("Filtered Tasks", len(filtered_df))
        
        with col_stats3:
            if len(filtered_df) > 0:
                percentage = (len(filtered_df) / len(df)) * 100
                st.metric("Filter Coverage", f"{percentage:.1f}%")
            else:
                st.metric("Filter Coverage", "0%")
        
        # Distribution charts
        if len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("📈 Distribution by Category")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Build status per questionId based on evaluation score
                def _status_for_qid(qid: str) -> str:
                    if not (inferencer_model and judge_model and eval_map):
                        return "Not evaluated"
                    rec = eval_map.get(qid)
                    if not rec or rec.get("acceptabilityScore") is None:
                        return "Not evaluated"
                    try:
                        s = int(rec.get("acceptabilityScore"))
                    except Exception:
                        return "Not evaluated"
                    labels = {
                        0: "Completely Unacceptable",
                        1: "Useful but Unacceptable",
                        2: "Acceptable",
                        3: "Optimal",
                    }
                    return labels.get(s, "Not evaluated")

                vis_df = filtered_df.copy()
                vis_df["qid"] = vis_df["questionId"].astype(str)
                vis_df["status"] = vis_df["qid"].apply(_status_for_qid)

                def _stacked_chart(df: pd.DataFrame, col: str, title: str):
                    agg = df.groupby([col, "status"]).size().reset_index(name="count")
                    domain = [
                        "Completely Unacceptable",
                        "Useful but Unacceptable",
                        "Acceptable",
                        "Optimal",
                        "Not evaluated",
                    ]
                    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
                    chart = alt.Chart(agg).mark_bar().encode(
                        x=alt.X(f"{col}:N", sort='-y', title=title),
                        y=alt.Y("count:Q", title="Count"),
                        color=alt.Color("status:N", scale=alt.Scale(domain=domain, range=colors), legend=alt.Legend(title="Status")),
                        tooltip=[col, "status", "count"],
                    )
                    st.altair_chart(chart, use_container_width=True)

                _stacked_chart(vis_df, "type", "Question Type")
                st.caption("Distribution by Question Type (color-coded by evaluation status)")
            
            with col_chart2:
                _stacked_chart(vis_df, "level", "Complexity Level")
                st.caption("Distribution by Complexity Level (color-coded by evaluation status)")
            
            # Programming language distribution
            st.markdown("---")
            st.subheader("🔤 Programming Language Distribution")
            _stacked_chart(vis_df, "tag", "Programming Language")
        
        # Task list
        if len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("📋 Task List")
            # Search functionality
            search_term = st.text_input("🔍 Search in questions:", placeholder="Enter keywords to search...", disabled=st.session_state.get("is_running", False))
            
            if search_term:
                search_filtered = filtered_df[
                    filtered_df['question'].str.contains(search_term, case=False, na=False) |
                    filtered_df['answer'].str.contains(search_term, case=False, na=False)
                ]
                display_df = search_filtered
                st.info(f"Found {len(search_filtered)} tasks matching '{search_term}'")
            else:
                display_df = filtered_df
            
            # Display tasks in a table
            st.dataframe(
                display_df[['questionId', 'type', 'level', 'tag', 'question']].head(50),
                width="stretch",
                column_config={
                    'questionId': st.column_config.TextColumn('ID', width=100),
                    'type': st.column_config.TextColumn('Type', width=80),
                    'level': st.column_config.TextColumn('Level', width=80),
                    'tag': st.column_config.TextColumn('Language', width=100),
                    'question': st.column_config.TextColumn('Question', width=400)
                }
            )
            
            if len(display_df) > 50:
                st.info(f"Showing first 50 of {len(display_df)} tasks. Use search to find specific tasks.")
        else:
            st.info("No tasks match the current filters. Try adjusting your filter criteria.")
    
    # Tab 2: Random Task
    with tab2:        
        if len(filtered_df) > 0:
            if st.button("🎯 Pick Random Task", type="primary", disabled=st.session_state.get("is_running", False)):
                random_task = filtered_df.sample(n=1).iloc[0]
                st.session_state.random_task = random_task
            
            # Display random task if available
            if 'random_task' in st.session_state:
                task = st.session_state.random_task
                st.markdown("**Selected Task:**")
                st.info(f"**{task['type'].title()}** - **{task['level'].title()}** - **{task['tag'].title()}**")
                
                # Show question preview
                question_preview = task['question'][:200] + "..." if len(task['question']) > 200 else task['question']
                st.text_area("Question Preview:", question_preview, height=100, disabled=True)
        else:
            st.warning("No tasks match the current filters.")
        
        if 'random_task' in st.session_state:
            task = st.session_state.random_task
            st.markdown("---")
            st.markdown("## 📖 Full Task Display")
            # Show question (without embedding the reference answer twice)
            st.markdown(format_task_display(task, include_answer=False))
            # Answers side-by-side
            st.markdown("### 🧪 Answers")
            col_ref, col_model = st.columns(2)
            with col_ref:
                st.markdown("**Reference Answer**")
                st.text_area("Reference", task['answer'], height=300, disabled=True)
            with col_model:
                st.markdown("**Model Answer**")
                model_ans = ""
                if inferencer_model:
                    answers_map = load_inference_answers(selected_dataset, inferencer_model)
                    model_ans = answers_map.get(task['questionId'], "")
                if not model_ans:
                    st.info("No model answer available for this selection.")
                st.text_area("Model", model_ans or "", height=300, disabled=True)
                # Color-coded evaluation score (if available)
                eval_map = load_evaluation_results(selected_dataset, inferencer_model, judge_model)
                q_eval = eval_map.get(task['questionId']) if eval_map else None
                score = q_eval.get("acceptabilityScore") if q_eval else None
                if score is not None:
                    labels = {
                        0: "Score: 0 - Completely Unacceptable",
                        1: "Score: 1 - Useful but Unacceptable",
                        2: "Score: 2 - Acceptable",
                        3: "Score: 3 - Optimal",
                    }
                    colors = {
                        0: "#e74c3c",  # red
                        1: "#e67e22",  # orange
                        2: "#f1c40f",  # yellow
                        3: "#2ecc71",  # green
                    }
                    bg = colors.get(int(score), "#bdc3c7")
                    txt = labels.get(int(score), f"Score: {score}")
                    st.markdown(
                        f"<div style='margin-top:8px;padding:10px;border-radius:6px;background:{bg};color:white;font-weight:600;'>{txt}</div>",
                        unsafe_allow_html=True,
                    )
                    if q_eval and q_eval.get("acceptabilityEvaluation"):
                        with st.expander("Why this score?"):
                            st.write(q_eval["acceptabilityEvaluation"])
                else:
                    st.caption("No judge score available for this selection.")
    
    # Tab 3: Leaderboard
    with tab3:
        st.subheader("🏆 Evaluated Models Leaderboard")
        if not judge_model:
            st.info("Select a judge model in the sidebar to view the leaderboard.")
        else:
            evaluated_models = list_evaluated_models(selected_dataset, judge_model, "eval-cot-ref")
            if len(evaluated_models) == 0:
                st.warning("No evaluated models found for the selected dataset and judge.")
            else:
                rows = []
                hidden_partial = 0
                subset_qids_lead = set(filtered_df['questionId'].astype(str).tolist()) if not filtered_df.empty else set()
                for model_id in evaluated_models:
                    # Include only models that have complete evaluation for current subset
                    eval_map_model = load_evaluation_results(selected_dataset, model_id, judge_model)
                    have_eval_ids_model = {qid for qid, rec in eval_map_model.items() if rec and rec.get('acceptabilityScore') is not None}
                    covered = subset_qids_lead <= have_eval_ids_model if subset_qids_lead else False
                    if not covered:
                        if len(subset_qids_lead & have_eval_ids_model) > 0:
                            hidden_partial += 1
                        continue

                    score = get_evaluation_score(
                        selected_dataset,
                        model_id,
                        judge_model,
                        selected_types,
                        selected_levels,
                        selected_languages,
                    )
                    meta = load_inference_metadata(selected_dataset, model_id)
                    rows.append({
                        "Model": model_id,
                        "Provider": (model_id.split("/")[0] if "/" in model_id else ""),
                        "Acceptance Rate (%)": score["acceptance_rate"],
                        "Accepted": score["accepted_questions"],
                        "Total": score["total_questions"],
                        "Context Length": meta.get("context_length"),
                        "Est. Cost ($)": meta.get("est_cost"),
                        "Est. Duration (s)": meta.get("duration_sec"),
                    })

                if len(rows) == 0:
                    st.warning("No evaluation data matched the current filters.")
                else:
                    leaderboard_df = pd.DataFrame(rows)
                    leaderboard_df = leaderboard_df.sort_values(by=["Acceptance Rate (%)", "Accepted"], ascending=[False, False])

                    st.dataframe(
                        leaderboard_df,
                        width="stretch",
                        column_config={
                            "Model": st.column_config.TextColumn("Inferencer Model", width=300),
                            "Provider": st.column_config.TextColumn("Provider", width=120),
                            "Acceptance Rate (%)": st.column_config.NumberColumn("Acceptance Rate (%)", format="%.1f"),
                            "Accepted": st.column_config.NumberColumn("Accepted"),
                            "Total": st.column_config.NumberColumn("Total"),
                            "Context Length": st.column_config.NumberColumn("Context Length"),
                            "Est. Cost ($)": st.column_config.NumberColumn("Est. Cost ($)", format="$%.4f"),
                            "Est. Duration (s)": st.column_config.NumberColumn("Est. Duration (s)", format="%.2f"),
                        },
                    )

                    st.caption("Leaderboard reflects the currently selected Dataset, Judge and filters.")
                    if hidden_partial > 0:
                        st.warning(f"{hidden_partial} model(s) have partial results for this selection and are hidden.")

    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use the sidebar filters to narrow down tasks by type, complexity, and programming language.")
    st.markdown("🎲 **Random Task:** Click the button to get a random task from the filtered results.")
    st.markdown("📊 **Live Updates:** The evaluation scores update automatically based on the output folder structure.")

if __name__ == "__main__":
    main()
