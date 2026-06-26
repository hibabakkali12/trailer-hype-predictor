"""
Queries the TMDB Discover API to collect a list of popular movies
released in the last two years. Saves movie IDs and titles to movies.json.
This is the starting point of the pipeline — all other scripts depend on this file.
"""

import requests
import json
import time
from datetime import date, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()
TMDB_API_KEY=os.getenv("TMDB_API_KEY")
NUM_PAGES=25
def get_one_page(page_number,date_from,date_to):
    url="https://api.themoviedb.org/3/discover/movie"
    params={
        "api_key":TMDB_API_KEY,
        "sort_by":"popularity.desc",
        "primary_release_date.gte":date_from,
        "primary_release_date.lte":date_to,
        "include_adult":"false",
        "page":page_number
    }
    response=requests.get(url, params=params)
    if response.status_code!=200:
        print(f"Page: {page_number} failed with status code {response.status_code}")
        print(f"+{response.text}")
        return []
    data=response.json()
    return data.get("results", [])


if __name__=="__main__":
    today=date.today()
    two_years_ago=today-timedelta(days=2*365)
    date_from_str=two_years_ago.isoformat()
    date_to_str=today.isoformat()
    print(f"Movies released between {date_from_str} and {date_to_str}")
    all_movies=[]
    for page in range(1, NUM_PAGES+1):
        print(f"Page {page} of {NUM_PAGES}")
        movies=get_one_page(page, date_from_str, date_to_str)
        all_movies.extend(movies)
        time.sleep(0.3)
        simplified_list=[{"id":m["id"], "title":m.get("title","UNKNOWN")}
                         for m in all_movies]
        with open("movies.json","w") as f:
            json.dump(simplified_list, f, indent=2)

        



       
