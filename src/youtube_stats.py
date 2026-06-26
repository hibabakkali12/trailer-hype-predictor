import requests
import time
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")
YT_VIDEO_URL="https://www.googleapis.com/youtube/v3/videos"
BATCH_SIZE=50
INPUT_FILE  = "movies_with_trailers.csv"   # created in step 4
OUTPUT_FILE = "movies_with_yt_stats.csv"
def get_stats_batch(video_keys):
    params={
        "key":YOUTUBE_API_KEY,
        "id":",".join(video_keys),
        "part":"statistics"
    }
    response=requests.get(YT_VIDEO_URL, params=params)
    if response.status_code!=200:
        print(f"Failed to get stats for video keys {video_keys} with status code {response.status_code}")
        return {}
    data=response.json()
    items=data.get("items",[])
    result={}
    for item in items:
        vid_id=item.get("id")
        stats=item.get("statistics",{})
        result[vid_id]={
            "view_count": int(stats.get("viewCount", 0)),
            "like_count":int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0) or 0)
        }
    return result

if __name__=="__main__":
    df = pd.read_csv(INPUT_FILE)
    total = len(df)
    print(f"Loaded {total} movies with trailer keys.")
    all_keys = df["youtube_key"].dropna().tolist()
    all_stats = {}
    for batch_start in range(0, len(all_keys), BATCH_SIZE):
        batch = all_keys[batch_start : batch_start + BATCH_SIZE]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_keys) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Fetching batch {batch_num}/{total_batches}")
        batch_stats = get_stats_batch(batch)
        all_stats.update(batch_stats)
        time.sleep(0.5)
    view_counts      = []
    like_counts      = []
    comment_counts   = []
    engagement_rates = []
    for _, row in df.iterrows():
        key   = row["youtube_key"]
        stats = all_stats.get(key, {})
        views    = stats.get("view_count",    0)
        likes    = stats.get("like_count",    0)
        comments = stats.get("comment_count", 0)
        engagement = (likes / views) if views > 0 else 0
        view_counts.append(views)
        like_counts.append(likes)
        comment_counts.append(comments)
        engagement_rates.append(round(engagement, 6))

    df["view_count"]      = view_counts
    df["like_count"]      = like_counts
    df["comment_count"]   = comment_counts
    df["engagement_rate"] = engagement_rates

    before = len(df)
    df = df[df["view_count"] > 0].reset_index(drop=True)
    after  = len(df)
    print(f"Dropped {before - after} rows with 0 views.")

    df.to_csv(OUTPUT_FILE, index=False)

  