"""
Takes the movie IDs from movies.json and fetches full details for each one
from the TMDB API — including box office revenue, budget, runtime, and genres.
Movies with missing revenue data are filtered out. Saves to movies_details.csv.
"""

import requests
import json
import time
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
TMDB_API_KEY=os.getenv("TMDB_API_KEY")
INPUT_FILE="movies.json"
OUTPUT_FILE="movies_details.csv"
min_revenue=1_000_000
def get_movie_details(movie_id):
    url=f"https://api.themoviedb.org/3/movie/{movie_id}"
    params={
        "api_key":TMDB_API_KEY,
        "language":"en-US"
    }
    response=requests.get(url,params=params)
    if response.status_code!=200:
        print(f"Failed to get details for movie ID {movie_id} with status code {response.status_code}")
        return None
    return response.json()
def extract_fields(movie):
    return{
        "id": movie.get("id"),
        "title":movie.get("title", "UNKNOWN"),
         "release_date":   movie.get("release_date", ""),
        "revenue":        movie.get("revenue", 0),
        "budget":         movie.get("budget", 0),
        "vote_average":   movie.get("vote_average", 0),
        "vote_count":     movie.get("vote_count", 0),
        "popularity":     movie.get("popularity", 0),
        "runtime":        movie.get("runtime", 0),
        "original_language": movie.get("original_language", ""),
        "genres": ", ".join([g["name"] for g in movie.get("genres", [])]),
    }
if __name__=="__main__":

 with open(INPUT_FILE, "r") as f:
        movies=json.load(f)
        total=len(movies)
        print(f"found {total} movies")
        rows=[]
        for i,movie in enumerate(movies,start=1):
           movie_id=movie["id"]
           movie_title=movie["title"]
           print(f"[{i}/{total}] Fetching: {movie_title} (ID: {movie_id})")
           raw_data=get_movie_details(movie_id)
           if raw_data is None:
               continue
           clean_data=extract_fields(raw_data)
           if clean_data["revenue"] < min_revenue:
               continue
           rows.append(clean_data)
           time.sleep(0.1)
        df=pd.DataFrame(rows)
        df=df.sort_values(by="revenue", ascending=False).reset_index(drop=True)
        df.to_csv(OUTPUT_FILE, index=False)
