"""
Downloads up to 100 top comments per trailer from the YouTube Data API.
Movies with disabled comments are skipped gracefully. Each comment is saved
as its own row linked back to the movie. Saves to trailer_comments.csv.
"""

import requests
import time
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")
INPUT="movies_with_yt_stats.csv"
OUTPUT="trailer_comments.csv"
YT_COMMENTS_URL="https://www.googleapis.com/youtube/v3/commentThreads"
MAX_COMMENTS=100
RESULTS_PER_PAGE=100
def get_comments(video_key, max_comments=MAX_COMMENTS):
    comments=[]
    page_token=None
    while len(comments)<max_comments:
        params={
            "key":YOUTUBE_API_KEY,
            "videoId":video_key,
            "part":"snippet",
            "order":"relevance",
            "maxResults":RESULTS_PER_PAGE,
            "textFormat":"plainText"

        }

        if page_token:
            params["pageToken"] = page_token
        response=requests.get(YT_COMMENTS_URL, params=params)
        if response.status_code==403:
            print("Access forbidden. Possibly comments are disabled for this video.")
            return None
        if response.status_code!=200:
            print(f"error getting comments with status code {response.status_code}")
            break
        data=response.json()
        items=data.get("items",[])
        for item in items:
            top_comment=(item.get("snippet",{}).get("topLevelComment", {}).get("snippet", {}).get("textDisplay"))
            if top_comment:
                cleaned=" ".join(top_comment.split())
                comments.append(cleaned)
            if len(comments)>=max_comments:
                break
        page_token=data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)
    return comments
if __name__=="__main__":
    df=pd.read_csv(INPUT)
    total=len(df)
    disabled_count = 0
    found_count = 0
    all_comments=[]
    for i, row in df.iterrows():
        video_key=row.get("youtube_key")
        title=row["title"]
        id=row["id"]
        print(f"[{i+1}/{total}] {title}")
        if pd.isna(video_key):
            print(f"No trailer key for movie ID {row['id']}. Skipping comments.")
            continue
        comments=get_comments(video_key)
        if comments is None:
            print(f"Comments are disabled.")
            disabled_count += 1
            continue
        if len(comments) == 0:
            print(f"No comments found.")
            continue
        print(f"Collected {len(comments)} comments.")
        found_count += 1

        for comment in comments:
            all_comments.append({
                "movie_id":id,
                "title":title,
                "youtube_key":video_key,
                "comment":comment
            })
        time.sleep(0.2)
    comments_df=pd.DataFrame(all_comments)
    comments_df.to_csv(OUTPUT, index=False)

