import streamlit as st
import pandas as pd
import json
import random
import yaml
import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from utils import setup_logger

logger = setup_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Coding Problems & Evaluation Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_models_cache() -> Dict[str, Any]:
    """
    Load the cached OpenRouter models from YAML file.
    
    Returns:
        Dictionary containing model information
    """
    try:
        with open("config/openrouter_models_cache.yml", 'r', encoding='utf-8') as file:
            models = yaml.safe_load(file)
        return models
    except Exception as e:
        logger.error(f"Error loading models cache: {e}")
        return {}

@st.cache_data()
def load_evaluation_data(path: str) -> pd.DataFrame:
    """
    Load the evaluation data from the given path.
    
    Args:
        path: The path to the evaluation data.
        
    Returns:
        DataFrame with evaluation data
    """
    # Search recursively for .jsonl files in subdirectories
    files = glob.glob(f"{path}/**/*.jsonl", recursive=True)
    evals = []
    for file in files:
        try:
            df = pd.read_json(file, lines=True)[['questionId', 'questionMetadata', 'model', 'modelMetadata', 'acceptabilityScore']]
            df['acceptance'] = (df['acceptabilityScore'] >= 2).astype(int)
            questionMetadata = pd.json_normalize(df['questionMetadata'])
            modelMetadata = pd.json_normalize(df['modelMetadata'])
            df = pd.concat([df, questionMetadata, modelMetadata], axis=1).drop(['questionMetadata', 'modelMetadata', 'acceptabilityScore'], axis=1)
            evals.append(df)
        except Exception as e:
            logger.warning(f"Error loading file {file}: {e}")
            continue
    
    if len(evals) == 0:
        logger.warning("No evaluation data found in %s.", path)
        return pd.DataFrame()
    
    logger.info("Loaded %s evaluation data from %s.", len(evals), path)
    return pd.concat(evals).reset_index(drop=True)

def get_filtered_leaderboard(df: pd.DataFrame, types: list[str], levels: list[str], tags: list[str]) -> pd.DataFrame:
    """
    From the given dataframe, filter the rows which contain any of the given types, levels and tags.
    
    Args:
        df: The dataframe to filter
        types: The list of types to filter
        levels: The list of levels to filter
        tags: The list of tags to filter
        
    Returns:
        The filtered dataframe.
    """
    filtered_df = df[(df['type'].isin(types)) & (df['level'].isin(levels)) & (df['tag'].isin(tags))]
    return filtered_df.groupby(['name', 'model', 'provider']).agg({'acceptance': 'mean'}).sort_values(by='acceptance', ascending=False).reset_index()

def render_evaluation_tab(df: pd.DataFrame, type_key: str, level_key: str, tag_key: str):
    """
    Render the evaluation tab with the given dataframe and keys.
    
    Args:
        df: The dataframe to render
        type_key: The key for the question type multiselect
        level_key: The key for the question level multiselect
        tag_key: The key for the question tag multiselect
    """
    # Select all unique options by default
    unique_types = df['type'].unique()
    unique_levels = df['level'].unique()
    unique_tags = df['tag'].unique()
    
    tags = st.multiselect("Select Question Tag", options=unique_tags, default=unique_tags, key=tag_key)
    col1, col2 = st.columns(2)
    with col1:
        types = st.multiselect("Select Question Type", options=unique_types, default=unique_types, key=type_key)
    with col2:
        levels = st.multiselect("Select Question Level", options=unique_levels, default=unique_levels, key=level_key)
    
    if types and levels and tags:
        filtered_df = get_filtered_leaderboard(df, types, levels, tags)
        # Make the leaderboard take up necessary height
        st.dataframe(filtered_df, width="stretch", height=(len(filtered_df) + 1) * 35 + 3)
    else:
        st.warning("Please select at least one type, level, and tag.")

@st.cache_data
def load_coding_problems(file_path: str) -> pd.DataFrame:
    """
    Load coding problems from JSONL file and convert to DataFrame.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        DataFrame with coding problems
    """
    problems = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    problem = json.loads(line)
                    # Extract metadata
                    metadata = problem.get('questionMetadata', {})
                    problems.append({
                        'questionId': problem.get('questionId', ''),
                        'question': problem.get('question', ''),
                        'answer': problem.get('answer', ''),
                        'type': metadata.get('type', ''),
                        'level': metadata.get('level', ''),
                        'tag': metadata.get('tag', '')
                    })
        
        df = pd.DataFrame(problems)
        # Clean up empty values
        df = df.replace('', pd.NA).dropna(subset=['question', 'type', 'level', 'tag'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def format_problem_display(problem: Dict[str, Any]) -> str:
    """
    Format a problem for human-readable display.
    
    Args:
        problem: Dictionary containing problem data
        
    Returns:
        Formatted string representation
    """
    formatted = f"""
## Problem ID: {problem['questionId']}

**Type:** {problem['type'].title()}
**Complexity:** {problem['level'].title()}
**Language:** {problem['tag'].title()}

### Question:
{problem['question']}

### Answer:
{problem['answer']}
"""
    return formatted

def get_evaluation_score(task: str, inferencer_model: str, judge_model: str) -> Dict[str, Any]:
    """
    Get the evaluation score for the selected combination.
    
    Args:
        task: The selected task (stack-eval or stack-unseen)
        inferencer_model: The selected inferencer/evaluatee model
        judge_model: The selected judge/evaluator model
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Construct the expected path based on the dashboard.py logic
    if task == "stack-eval":
        base_path = f"output/stack-eval/evl/{inferencer_model}/eval-cot-ref"
    else:  # stack-unseen
        base_path = f"output/stack-unseen/evl/{inferencer_model}/eval-cot-ref"
    
    # Try to load evaluation data
    try:
        df = load_evaluation_data(base_path)
        if not df.empty:
            # Calculate overall acceptance rate
            total_questions = len(df)
            accepted_questions = df['acceptance'].sum()
            acceptance_rate = (accepted_questions / total_questions) * 100 if total_questions > 0 else 0
            
            return {
                'total_questions': total_questions,
                'accepted_questions': accepted_questions,
                'acceptance_rate': acceptance_rate,
                'data_loaded': True
            }
    except Exception as e:
        logger.warning(f"Could not load evaluation data for {base_path}: {e}")
    
    return {
        'total_questions': 0,
        'accepted_questions': 0,
        'acceptance_rate': 0,
        'data_loaded': False
    }

def main():
    st.title("💻 Coding Problems & Evaluation Dashboard")
    st.markdown("Filter and explore coding problems by type, complexity, and programming language, plus view live evaluation scores.")
    
    # Load models cache
    models_cache = load_models_cache()
    
    # Top left controls for LLM selection and task
    st.sidebar.header("🤖 Model & Task Selection")
    
    # Task selector
    task_options = ["stack-eval", "stack-unseen"]
    selected_task = st.sidebar.selectbox(
        "Select Task",
        options=task_options,
        index=0,
        help="Choose between stack-eval and stack-unseen tasks"
    )
    
    # Get available models from cache
    available_models = []
    if models_cache:
        for model_id, model_info in models_cache.items():
            if model_id not in ['_cache_time', 'default', 'default-json']:
                model_name = model_info.get('model_metadata', {}).get('name', model_id)
                available_models.append((model_id, model_name))
    
    # Sort models by name for better UX
    available_models.sort(key=lambda x: x[1])
    
    # Model selectors
    if available_models:
        model_choices = [f"{name} ({id})" for id, name in available_models]
        
        selected_inferencer = st.sidebar.selectbox(
            "Inferencer/Evaluatee Model",
            options=model_choices,
            index=0 if model_choices else None,
            help="Select the model that generates solutions to be evaluated"
        )
        
        selected_judge = st.sidebar.selectbox(
            "Judge/Evaluator Model",
            options=model_choices,
            index=0 if model_choices else None,
            help="Select the model that evaluates the generated solutions"
        )
        
        # Extract model IDs from selections
        inferencer_model = selected_inferencer.split(" (")[-1].rstrip(")") if selected_inferencer else ""
        judge_model = selected_judge.split(" (")[-1].rstrip(")") if selected_judge else ""
        
        # Get evaluation score
        if inferencer_model and judge_model:
            evaluation_score = get_evaluation_score(selected_task, inferencer_model, judge_model)
            
            # Display evaluation score at the top
            st.header("📊 Live Evaluation Score")
            
            if evaluation_score['data_loaded']:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Task", selected_task.replace("-", " ").title())
                
                with col2:
                    st.metric("Total Questions", evaluation_score['total_questions'])
                
                with col3:
                    st.metric("Accepted Solutions", evaluation_score['accepted_questions'])
                
                with col4:
                    st.metric("Acceptance Rate", f"{evaluation_score['acceptance_rate']:.1f}%")
                
                # Progress bar for acceptance rate
                st.progress(evaluation_score['acceptance_rate'] / 100)
                
            else:
                st.warning(f"No evaluation data found for the selected combination. Check if the output directory exists: output/{selected_task}/evl/{inferencer_model}/eval-cot-ref")
    else:
        st.sidebar.warning("No models found in cache. Please check the openrouter_models_cache.yml file.")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📋 Coding Problems", "📊 Evaluation Dashboard", "🎲 Random Problem"])
    
    # Load coding problems data
    data_file = "data/stack-eval.jsonl"
    df = load_coding_problems(data_file)
    
    if df.empty:
        st.error("No coding problems data loaded. Please check the file path.")
        return
    
    # Tab 1: Coding Problems
    with tab1:
        st.header("📋 Problem List")
        
        # Sidebar filters for coding problems
        st.sidebar.header("🔍 Problem Filters")
        
        # Question type filter
        question_types = sorted(df['type'].unique())
        selected_types = st.sidebar.multiselect(
            "Question Type",
            options=question_types,
            default=question_types,
            help="Select one or more question types to filter by"
        )
        
        # Complexity level filter
        complexity_levels = sorted(df['level'].unique())
        selected_levels = st.sidebar.multiselect(
            "Complexity Level",
            options=complexity_levels,
            default=complexity_levels,
            help="Select one or more complexity levels to filter by"
        )
        
        # Programming language filter
        programming_languages = sorted(df['tag'].unique())
        selected_languages = st.sidebar.multiselect(
            "Programming Language",
            options=programming_languages,
            default=programming_languages,
            help="Select one or more programming languages to filter by"
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
        
        # Display filter summary
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Total Problems:** {len(df)}")
        st.sidebar.markdown(f"**Filtered Problems:** {len(filtered_df)}")
        
        # Display statistics
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.metric("Total Problems", len(df))
        
        with col_stats2:
            st.metric("Filtered Problems", len(filtered_df))
        
        with col_stats3:
            if len(filtered_df) > 0:
                percentage = (len(filtered_df) / len(df)) * 100
                st.metric("Filter Coverage", f"{percentage:.1f}%")
            else:
                st.metric("Filter Coverage", "0%")
        
        # Distribution charts
        if len(filtered_df) > 0:
            st.subheader("📈 Distribution by Category")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                type_counts = filtered_df['type'].value_counts()
                st.bar_chart(type_counts)
                st.caption("Distribution by Question Type")
            
            with col_chart2:
                level_counts = filtered_df['level'].value_counts()
                st.bar_chart(level_counts)
                st.caption("Distribution by Complexity Level")
            
            # Programming language distribution
            st.subheader("🔤 Programming Language Distribution")
            lang_counts = filtered_df['tag'].value_counts()
            st.bar_chart(lang_counts)
        
        # Problem list
        if len(filtered_df) > 0:
            # Search functionality
            search_term = st.text_input("🔍 Search in questions:", placeholder="Enter keywords to search...")
            
            if search_term:
                search_filtered = filtered_df[
                    filtered_df['question'].str.contains(search_term, case=False, na=False) |
                    filtered_df['answer'].str.contains(search_term, case=False, na=False)
                ]
                display_df = search_filtered
                st.info(f"Found {len(search_filtered)} problems matching '{search_term}'")
            else:
                display_df = filtered_df
            
            # Display problems in a table
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
                st.info(f"Showing first 50 of {len(display_df)} problems. Use search to find specific problems.")
        else:
            st.info("No problems match the current filters. Try adjusting your filter criteria.")
    
    # Tab 2: Evaluation Dashboard
    with tab2:
        st.header("📊 Evaluation Dashboard")
        
        if inferencer_model and judge_model:
            # Load evaluation data for both tasks
            stack_eval_df = load_evaluation_data(f"output/stack-eval/evl/{inferencer_model}/eval-cot-ref")
            stack_unseen_df = load_evaluation_data(f"output/stack-unseen/evl/{inferencer_model}/eval-cot-ref")
            
            eval_tab1, eval_tab2 = st.tabs(["Stack Eval", "Stack Unseen"])
            
            if not stack_eval_df.empty:
                with eval_tab1:
                    render_evaluation_tab(stack_eval_df, "eval_types", "eval_levels", "eval_tags")
            else:
                with eval_tab1:
                    st.warning("No Stack Eval data found for the selected model combination.")
            
            if not stack_unseen_df.empty:
                with eval_tab2:
                    render_evaluation_tab(stack_unseen_df, "unseen_types", "unseen_levels", "unseen_tags")
            else:
                with eval_tab2:
                    st.warning("No Stack Unseen data found for the selected model combination.")
        else:
            st.info("Please select both inferencer and judge models to view evaluation data.")
    
    # Tab 3: Random Problem
    with tab3:
        st.header("🎲 Random Problem")
        
        if len(filtered_df) > 0:
            if st.button("🎯 Pick Random Problem", type="primary"):
                random_problem = filtered_df.sample(n=1).iloc[0]
                st.session_state.random_problem = random_problem
            
            # Display random problem if available
            if 'random_problem' in st.session_state:
                problem = st.session_state.random_problem
                st.markdown("**Selected Problem:**")
                st.info(f"**{problem['type'].title()}** - **{problem['level'].title()}** - **{problem['tag'].title()}**")
                
                # Show question preview
                question_preview = problem['question'][:200] + "..." if len(problem['question']) > 200 else problem['question']
                st.text_area("Question Preview:", question_preview, height=100, disabled=True)
                
                if st.button("👁️ View Full Problem"):
                    st.session_state.show_full_problem = True
        else:
            st.warning("No problems match the current filters.")
        
        # Full problem display
        if 'show_full_problem' in st.session_state and st.session_state.show_full_problem:
            if 'random_problem' in st.session_state:
                problem = st.session_state.random_problem
                st.markdown("---")
                st.markdown("## 📖 Full Problem Display")
                st.markdown(format_problem_display(problem))
                
                if st.button("❌ Close Full View"):
                    st.session_state.show_full_problem = False
                    st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use the sidebar filters to narrow down problems by type, complexity, and programming language.")
    st.markdown("🎲 **Random Problem:** Click the button to get a random problem from the filtered results.")
    st.markdown("📊 **Live Updates:** The evaluation scores update automatically based on the output folder structure.")

if __name__ == "__main__":
    main()
