from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ast

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load data
CSV_FILE = "data/news_chunks_w_umap.csv"
ALL_TOPICS_COL = "all_topics"
SOURCE_COL = "source_type"

def load_data():
    df = pd.read_csv(CSV_FILE)
    # reading topics col as a list object
    if df[ALL_TOPICS_COL].dtype == object:
        df[ALL_TOPICS_COL] = df[ALL_TOPICS_COL].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

df = load_data()

# function to get subtopic distribution as a list of dictionaries

def get_subtopic_distribution(drill_path=None):
    if not drill_path:
        # Broadest topic distribution
        subdf = df[df[ALL_TOPICS_COL].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        subdf['label'] = subdf[ALL_TOPICS_COL].apply(lambda x: x[-1])
        group = subdf.groupby([SOURCE_COL, 'label']).size().reset_index(name='count')
        group['source_total'] = group.groupby(SOURCE_COL)['count'].transform('sum')
        group['proportion'] = group['count'] / group['source_total']
        # Sort by total count (descending)
        order = group.groupby('label')['count'].sum().sort_values(ascending=False).index
        group['label'] = pd.Categorical(group['label'], categories=order, ordered=True)
        group = group.sort_values(['label', SOURCE_COL])
        # Return as list of dicts
        return group.to_dict(orient='records')
    
    # Otherwise, drilldown
    if not isinstance(drill_path, (list, tuple)):
        raise ValueError("drill_path must be a list or None")
    if len(drill_path) < 1:
        raise ValueError("drill_path must be non-empty or None for broadest topic distribution.")
    
    def matches_drill_path(all_topics):
        if not isinstance(all_topics, list):
            return False
        if len(all_topics) < len(drill_path):
            return False
        return all_topics[-len(drill_path):] == drill_path

    subset = df[df[ALL_TOPICS_COL].apply(matches_drill_path)].copy()
    if subset.empty:
        return []
    # Get next subtopic up the hierarchy
    def get_next_subtopic(all_topics):
        idx = len(all_topics) - len(drill_path)
        if idx > 0:
            return all_topics[idx - 1]
        return None

    subset['label'] = subset[ALL_TOPICS_COL].apply(get_next_subtopic)
    subset = subset[subset['label'].notnull() & (subset['label'] != "")]
    if subset.empty:
        return []
    group = subset.groupby([SOURCE_COL, 'label']).size().reset_index(name='count')
    group['source_total'] = group.groupby(SOURCE_COL)['count'].transform('sum')
    group['proportion'] = group['count'] / group['source_total']
    order = group.groupby('label')['count'].sum().sort_values(ascending=False).index
    group['label'] = pd.Categorical(group['label'], categories=order, ordered=True)
    group = group.sort_values(['label', SOURCE_COL])
    return group.to_dict(orient='records')

# --- API endpoints ---

@app.get("/api/topics/")
def get_broad_topics():
    """
    Get distribution of broadest topics.
    """
    return {"topics": get_subtopic_distribution()}

@app.get("/api/topics/drilldown/")
def get_subtopics(path: str = Query("")):
    """
    Get subtopic distribution for a drill path.
    path: comma-separated string, e.g. "society,fundamental rights"
    """
    drill_path = [p.strip() for p in path.split(",") if p.strip()]
    if not drill_path:
        # If empty, return broadest
        return {"topics": get_subtopic_distribution()}
    return {"topics": get_subtopic_distribution(drill_path)}