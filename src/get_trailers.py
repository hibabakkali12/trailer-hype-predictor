import requests
import time
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
TMDB_API_KEY=os.getenv("TMDB_API_KEY")
def get_trailer_key(id):
     url = f"https://api.themoviedb.org/3/movie/{id}/videos"
     params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
    }
 
     response = requests.get(url, params=params)
 
     if response.status_code != 200:
        return None
 
     data     = response.json()
     videos   = data.get("results", [])
     for v in videos:
        if (v.get("site") == "YouTube"
                and v.get("type") == "Trailer"
                and v.get("official") is True):
            return v["key"]
     for v in videos:
        if v.get("site") == "YouTube" and v.get("type") == "Trailer":
            return v["key"]
     for v in videos:
        if v.get("site") == "YouTube":
            return v["key"]
     return None
input="movies_details.csv"
output = "movies_with_trailers.csv" 
if __name__ == "__main__":
    df=pd.read_csv(input)
    total = len(df)
    print(f"Loaded {total} movies")
    trailer_keys=[]
    trailer_urls=[]
    found_count=0
    missing_count=0
    for i, row in df.iterrows():
        id=row["id"]
        title=row["title"]
        print(f"[{i+1}/{total}] {title}")
        key=get_trailer_key(id)
        if key:
            found_count+=1
            url = f"https://www.youtube.com/watch?v={key}"
            trailer_keys.append(key)
            trailer_urls.append(url)
        else:
            missing_count+=1
            trailer_keys.append(None)
            trailer_urls.append(None)
        time.sleep(0.1)
    df["youtube_key"] = trailer_keys
    df["trailer_url"] = trailer_urls
    df_with_trailers = df.dropna(subset=["youtube_key"]).reset_index(drop=True)
    df_with_trailers.to_csv(output, index=False)


