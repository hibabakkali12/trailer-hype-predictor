import pandas as pd
from textblob import TextBlob
import time
COMMENTS_FILE="data/trailer_comments.csv"
YT_STATS_FILE="data/movies_with_yt_stats.csv"
COMMENT_ANALYSIS_FILE="data/comments_with_sentiment.csv"
FINAL_OUTPUT="data/movies_with_sentiment.csv"
def score_comment(text):
    if not isinstance(text,str) or text.strip()=="":
        return 0.0, 0.0
    blob=TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity
def label_sentiment(polarity):
    if polarity>0.05:
        return "positive"
    elif polarity<-0.05:
        return "negative"
    else:
        return "neutral"    
if __name__=="__main__":
    df=pd.read_csv(COMMENTS_FILE)
    total=len(df)
    print(f"Loaded {total} comments for sentiment analysis.")
    df=df.dropna(subset=["comment"]).reset_index(drop=True)
    print("Scoring sentiment for each comment")
    polarities=[]
    subjectivities=[]
    labels=[]
    for i, row in df.iterrows():
        polarity, subjectivity=score_comment(row["comment"])
        polarities.append(polarity)
        subjectivities.append(subjectivity)
        labels.append(label_sentiment(polarity))
        if (i+1)%500==0:
            print(f"Processed {i+1}/{total} comments.")
    df["polarity"]=polarities
    df["subjectivity"]=subjectivities
    df["sentiment_label"]=labels
    df.to_csv(COMMENT_ANALYSIS_FILE, index=False)

    print("Grouping data at the movie level:")
    movie_sentiment=(df.groupby("movie_id").agg(
        avg_sentiment=("polarity","mean"),
        avg_subjectivity=("subjectivity","mean"),
        n_comments_analyzed=("polarity","count"),
        positive_comments_pct=("polarity", lambda x: (x>0.05).mean()*100),
        negative_comments_pct=("polarity", lambda x: (x<-0.05).mean()*100),
        neutral_comments_pct=("polarity", lambda x: (x.between(-0.05, 0.05)).mean() * 100))).reset_index()
    movie_sentiment = movie_sentiment.round(4)
    print("Loading youtube statistics data")
    yt_df=pd.read_csv(YT_STATS_FILE)
    final_df=yt_df.merge(movie_sentiment,left_on="id", right_on="movie_id", how="left")
    before = len(final_df)
    final_df = final_df.dropna(subset=["avg_sentiment"]).reset_index(drop=True)
    after = len(final_df)
    print(f"Dropped {before - after} movies with no sentiment data.")
    print(f"Final dataset: {after} movies.")

    import numpy as np
    final_df["log_revenue"] = np.log10(final_df["revenue"])
    final_df["log_views"]   = np.log10(final_df["view_count"].clip(lower=1))
    final_df["log_likes"]   = np.log10(final_df["like_count"].clip(lower=1))
    final_df.to_csv(FINAL_OUTPUT, index=False)
