import streamlit as st
import pandas as pd
import json
import random
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="Coding Problems Dashboard",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def main():
    st.title("💻 Coding Problems Dashboard")
    st.markdown("Filter and explore coding problems by type, complexity, and programming language.")
    
    # Load data
    data_file = "data/stack-eval.jsonl"
    df = load_coding_problems(data_file)
    
    if df.empty:
        st.error("No data loaded. Please check the file path.")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
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
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Problem Statistics")
        
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
    
    with col2:
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
    
    # Problem list
    st.header("📋 Problem List")
    
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
            use_container_width=True,
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
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip:** Use the sidebar filters to narrow down problems by type, complexity, and programming language.")
    st.markdown("🎲 **Random Problem:** Click the button to get a random problem from the filtered results.")

if __name__ == "__main__":
    main()
