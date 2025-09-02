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
            df['acceptance'] = (df['acceptabilityScore'] >= 2).astype(int)
            
            # Extract metadata fields
            df['type'] = df['questionMetadata'].apply(lambda x: x.get('type', ''))
            df['level'] = df['questionMetadata'].apply(lambda x: x.get('level', ''))
            df['tag'] = df['questionMetadata'].apply(lambda x: x.get('tag', ''))
            
            # Extract model metadata
            df['name'] = df['modelMetadata'].apply(lambda x: x.get('name', ''))
            df['provider'] = df['modelMetadata'].apply(lambda x: x.get('provider', ''))
            
            # Drop the original metadata columns
            df = df.drop(['questionMetadata', 'modelMetadata', 'acceptabilityScore'], axis=1)
            evals.append(df)
        except Exception as e:
            logger.warning(f"Error loading file {file}: {e}")
            continue
    
    if len(evals) == 0:
        logger.warning("No evaluation data found in %s.", path)
        return pd.DataFrame()
    
    logger.info("Loaded %s evaluation data from %s.", len(evals), path)
    return pd.concat(evals).reset_index(drop=True)



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

def get_evaluation_score(task: str, inferencer_model: str, judge_model: str, 
                        selected_types: List[str] = None, selected_levels: List[str] = None, 
                        selected_tags: List[str] = None) -> Dict[str, Any]:
    """
    Get the evaluation score for the selected combination, optionally filtered by problem criteria.
    
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
    if task == "stack-eval":
        base_path = f"output/stack-eval/evl/{judge_model}/eval-cot-ref/{inferencer_model}.jsonl"
    else:  # stack-unseen
        base_path = f"output/stack-unseen/evl/{judge_model}/eval-cot-ref/{inferencer_model}.jsonl"
    
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
                available_models.append(model_id)
    
    # Sort models by ID for better UX
    available_models.sort()
    
    # Model selectors - display just the identifier
    if available_models:
        selected_inferencer = st.sidebar.selectbox(
            "Inferencer/Evaluatee Model",
            options=available_models,
            index=0 if available_models else None,
            help="Select the model that generates solutions to be evaluated"
        )
        
        selected_judge = st.sidebar.selectbox(
            "Judge/Evaluator Model",
            options=available_models,
            index=0 if available_models else None,
            help="Select the model that evaluates the generated solutions"
        )
        
        # Use the selected models directly
        inferencer_model = selected_inferencer
        judge_model = selected_judge
        
        # Evaluation score will be calculated after filters are applied
    else:
        st.sidebar.warning("No models found in cache. Please check the openrouter_models_cache.yml file.")
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📋 Coding Problems", "🎲 Random Problem"])
    
    # Load coding problems data
    data_file = "data/stack-eval.jsonl"
    df = load_coding_problems(data_file)
    
    if df.empty:
        st.error("No coding problems data loaded. Please check the file path.")
        return
    
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
    
    # Tab 1: Coding Problems
    with tab1:                
        # Live Evaluation Score (updated with filters)
        if inferencer_model and judge_model:
            st.subheader("📊 Live Evaluation Score")
            
            # Get evaluation score with current filters
            evaluation_score = get_evaluation_score(
                selected_task, inferencer_model, judge_model,
                selected_types, selected_levels, selected_languages
            )
            
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
                
                # Show the path being used
                st.info(f"📁 Data loaded from: `output/{selected_task}/evl/{judge_model}/eval-cot-ref/{inferencer_model}.jsonl`")
                
            else:
                st.warning(f"No evaluation data found for the selected combination: {selected_task}, {inferencer_model} and {judge_model}.")
        
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
            st.markdown("---")
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
            st.markdown("---")
            st.subheader("🔤 Programming Language Distribution")
            lang_counts = filtered_df['tag'].value_counts()
            st.bar_chart(lang_counts)
        
        # Problem list
        if len(filtered_df) > 0:
            st.markdown("---")
            st.subheader("📋 Problem List")
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
    
    # Tab 2: Random Problem
    with tab2:        
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
