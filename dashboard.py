import streamlit as st
import pandas as pd
import glob
from utils import setup_logger

STACK_EVAL_DIR = "output/stack-eval/evl/gpt-4-turbo-2024-04-09/eval-cot-ref"
STACK_UNSEEN_DIR = "output/stack-unseen/evl/gpt-4-turbo-2024-04-09/eval-cot-ref"
logger = setup_logger("dashboard")

st.set_page_config(layout="wide")

@st.cache_data()
def load_data(path: str):
    """
    Load the evaluation data from the given path.
    
    @param path: The path to the evaluation data.
    @return: The loaded evaluation data.
    """
    files = glob.glob(f"{path}/*.jsonl")
    evals = []
    for file in files:
        df = pd.read_json(file, lines=True)[['questionId', 'questionMetadata', 'model', 'modelMetadata', 'acceptabilityScore']]
        df['acceptance'] = (df['acceptabilityScore'] >= 2).astype(int)
        questionMetadata = pd.json_normalize(df['questionMetadata'])
        modelMetadata = pd.json_normalize(df['modelMetadata'])
        df = pd.concat([df, questionMetadata, modelMetadata], axis=1).drop(['questionMetadata', 'modelMetadata', 'acceptabilityScore'], axis=1)
        evals.append(df)
    if len(evals) == 0:
        logger.warning("No evaluation data found in %s.", path)
        return pd.DataFrame()
    logger.info("Loaded %s evaluation data from %s.", len(evals), path)
    return pd.concat(evals).reset_index(drop=True)

def get_filtered_leaderboard(df: pd.DataFrame, types: list[str], levels: list[str], tags: list[str]):
    """
    From the given dataframe, filter the rows which contain any of the given types, levels and tags.
    
    @param df: The dataframe to filter
    @param types: The list of types to filter
    @param levels: The list of levels to filter
    @param tags: The list of tags to filter
    @return: The filtered dataframe.
    """
    filtered_df = df[(df['type'].isin(types)) & (df['level'].isin(levels)) & (df['tag'].isin(tags))]
    return filtered_df.groupby(['name', 'model', 'provider']).agg({'acceptance': 'mean'}).sort_values(by='acceptance', ascending=False).reset_index()

def render_tab(df: pd.DataFrame, type_key: str, level_key: str, tag_key: str):
    """
    Render the given tab with the given dataframe and keys.
    
    @param df: The dataframe to render
    @param type_key: The key for the question type multiselect
    @param level_key: The key for the question level multiselect
    @param tag_key: The key for the question tag multiselect
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
        # Make the leaderboard take up necessary height: https://discuss.streamlit.io/t/st-dataframe-controlling-the-height-threshold-for-scolling/31769/4
        st.dataframe(filtered_df, use_container_width=True, height=(len(filtered_df) + 1) * 35 + 3)
    else:
        st.warning("Please select at least one type, level, and tag.")


def main():
    stack_eval_df = load_data(STACK_EVAL_DIR)
    stack_unseen_df = load_data(STACK_UNSEEN_DIR)
    
    st.title("Stack Evaluation Dashboard")
    
    tab1, tab2 = st.tabs(["Stack Eval", "Stack Unseen"])
    if not stack_eval_df.empty:
        with tab1:
            render_tab(stack_eval_df, "eval_types", "eval_levels", "eval_tags")
    if not stack_unseen_df.empty:
        with tab2:
            render_tab(stack_unseen_df, "unseen_types", "unseen_levels", "unseen_tags")

if __name__ == "__main__":
    main()

        