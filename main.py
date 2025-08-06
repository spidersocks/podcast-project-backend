from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ast
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Topic Distribution API ---
CSV_FILE = "data/news_chunks_w_umap.csv"
ALL_TOPICS_COL = "all_topics"
SOURCE_COL = "source_type"

def load_news_df():
    df = pd.read_csv(CSV_FILE)
    if df[ALL_TOPICS_COL].dtype == object:
        df[ALL_TOPICS_COL] = df[ALL_TOPICS_COL].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

news_df = load_news_df()

def get_subtopic_distribution(drill_path=None):
    if not drill_path:
        subdf = news_df[news_df[ALL_TOPICS_COL].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
        subdf['label'] = subdf[ALL_TOPICS_COL].apply(lambda x: x[-1])
        group = subdf.groupby([SOURCE_COL, 'label']).size().reset_index(name='count')
        group['source_total'] = group.groupby(SOURCE_COL)['count'].transform('sum')
        group['proportion'] = group['count'] / group['source_total']
        order = group.groupby('label')['count'].sum().sort_values(ascending=False).index
        group['label'] = pd.Categorical(group['label'], categories=order, ordered=True)
        group = group.sort_values(['label', SOURCE_COL])
        return group.to_dict(orient='records')
    if not isinstance(drill_path, (list, tuple)):
        raise ValueError("drill_path must be a list or None")
    if len(drill_path) < 1:
        raise ValueError("drill_path must be non-empty or None for broadest topic distribution.")
    drill_path = list(reversed(drill_path))
    def matches_drill_path(all_topics):
        if not isinstance(all_topics, list):
            return False
        if len(all_topics) < len(drill_path):
            return False
        return all_topics[-len(drill_path):] == drill_path
    subset = news_df[news_df[ALL_TOPICS_COL].apply(matches_drill_path)].copy()
    if subset.empty:
        return []
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

@app.get("/api/topics/")
def get_broad_topics():
    return {"topics": get_subtopic_distribution()}

PATH_SEPARATOR = "||"

@app.get("/api/topics/drilldown/")
def get_subtopics(path: str = Query("")):
    drill_path = [p.strip() for p in path.split(PATH_SEPARATOR) if p.strip()]
    if not drill_path:
        return {"topics": get_subtopic_distribution()}
    return {"topics": get_subtopic_distribution(drill_path)}

# --- Stance chart API ---

STANCE_Z_CSV = "data/stance_z_agg.csv"

def load_stance_z_df():
    return pd.read_csv(STANCE_Z_CSV)

@app.get("/api/stance/zscores/")
def get_stance_z_data(
    topics: Optional[str] = Query(None, description="Comma-separated list of topics to filter (optional)")
):
    """
    Returns avg z-score of stance per topic per source_type.
    """
    df = load_stance_z_df()
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
        df = df[df["topic"].isin(topic_list)]
    return {"data": df.to_dict(orient="records")}


# --- Sentiment Word Cloud API ---

TOPWORDS_PARQUET = "data/topwords_by_topic.parquet"
AVG_SENTIMENT_CSV = "data/avg_sentiment_by_source_topic.csv"

def load_topwords_df():
    df = pd.read_parquet(TOPWORDS_PARQUET)
    # Ensure 'top_words' column is a list (if stored as string, parse with ast.literal_eval)
    if df['top_words'].dtype == object:
        df['top_words'] = df['top_words'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and not isinstance(x, list) else x
        )
    return df

def load_avg_sentiment_df():
    df = pd.read_csv(AVG_SENTIMENT_CSV)
    # Fill missing columns if necessary for consistency
    if "quantile_sentiment_scaled" not in df.columns:
        df["quantile_sentiment_scaled"] = 0.0
    return df

topwords_df = load_topwords_df()
avg_sentiment_df = load_avg_sentiment_df()

@app.get("/api/wordcloud/topwords/")
def get_topwords(
    source_type: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    topic: Optional[str] = Query(None)
):
    df = topwords_df.copy()
    if source_type:
        df = df[df["source_type"] == source_type]
    if source_name:
        df = df[df["source_name"] == source_name]
    if topic:
        df = df[df["topic"] == topic]
    # Convert top_words to strings for JSON serialization if needed
    return {"data": [
        {
            "source_type": row["source_type"],
            "source_name": row["source_name"],
            "topic": row["topic"],
            "top_words": list(row["top_words"])
        }
        for _, row in df.iterrows()
    ]}

@app.get("/api/wordcloud/sentiment/")
def get_sentiment(
    source_type: Optional[str] = Query(None),
    source_name: Optional[str] = Query(None),
    topic: Optional[str] = Query(None)
):
    df = avg_sentiment_df.copy()
    if source_type:
        df = df[df["source_type"] == source_type]
    if source_name:
        df = df[df["source_name"] == source_name]
    if topic:
        df = df[df["topic"] == topic]
    return {"data": df.to_dict(orient="records")}

# --- END ---