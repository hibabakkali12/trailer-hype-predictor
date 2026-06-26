# Do Trailer Reactions Predict Box Office Success?

---

## The Question

When a movie trailer drops on YouTube, thousands of people immediately flood
the comments. Some are excited. Some are skeptical. Some are already buying
tickets. Can we measure that early reaction — the views, the likes, the tone
of the comments — and use it to predict how much money the movie will make?

That's what this project set out to answer.

---

## What I Built

A complete data pipeline that:

1. Collects movie data from TMDB (box office revenue, budget, ratings)
2. Finds the official YouTube trailer for each movie
3. Downloads trailer statistics (views, likes, comment count)
4. Scrapes up to 100 comments per trailer
5. Runs sentiment analysis on every comment using TextBlob
6. Aggregates everything into a single dataset
7. Builds and evaluates three predictive models

**Final dataset: 203 movies, 16,736 trailer comments, released between 2024 and 2026**

---

## Tools & Technologies

| Tool | What I used it for |
|---|---|
| Python | everything |
| TMDB API | movie details and trailer links |
| YouTube Data API v3 | video statistics and comments |
| TextBlob | sentiment analysis on comments |
| Pandas | data cleaning and manipulation |
| Scikit-learn | machine learning models |
| Matplotlib / Seaborn | all visualizations |

---

## The Data Pipeline

```
TMDB Discover API
"give me popular movies from the last 2 years" (25 pages)
        |
500 movie IDs collected

TMDB Details API (one call per movie)
revenue, budget, runtime, genres, ratings
        |
146 movies kept (others had no revenue data)

TMDB Videos API
official YouTube trailer key per movie
        |
~130 movies with trailers

YouTube Data API — video statistics
views, likes, comment count, engagement rate
        |
YouTube Data API — comments
up to 100 comments per trailer
16,736 comments collected

TextBlob sentiment analysis
polarity score per comment (-1 to +1)
        |
aggregated to movie level:
avg sentiment, % positive, % negative

Final dataset: 203 movies, one row per movie
```

---

## Key Findings

### 1. Budget is the dominant predictor of revenue

With a correlation of r = 0.59 and a feature importance of 0.621 in the
Random Forest model, production budget explains more of the variance in
box office revenue than all YouTube metrics combined. This makes intuitive
sense — studios spend more on films they expect to perform.

### 2. Trailer views have a weak but real signal

Trailer views correlated with revenue at r = 0.19 (p = 0.007), which is
statistically significant but not practically strong. A movie can go viral
on YouTube and still underperform at the box office (and vice versa).

### 3. Sentiment barely predicts revenue

Average comment sentiment showed almost no correlation with revenue
(r = 0.05, p = 0.473). The % of positive comments had a slightly stronger
signal (r = 0.15, p = 0.033) but it was still weak.

One reason: almost all trailer comment sections skew positive regardless
of how the movie eventually performs. Excited fans comment early; critics
arrive later or not at all. TextBlob also struggles with internet slang,
sarcasm, and short comments like "W movie" or "This is gonna be fire."

### 4. When combined, these features do have predictive power

| Model | R² | RMSE |
|---|---|---|
| Baseline (predict mean) | 0.000 | 0.766 |
| Linear Regression | 0.318 | 0.627 |
| Random Forest | 0.478 | 0.549 |

The Random Forest explains **47.8% of the variance** in box office revenue
and reduces prediction error by 28% compared to simply guessing the average.
That is meaningful — but it also means 52% of the variance remains unexplained
by the features collected here.

---

## What the Model Gets Right and Wrong

**Gets right:** mid-range studio films with normal marketing budgets and
typical audience reactions. These cluster tightly around the diagonal in
the actual vs predicted chart.

**Gets wrong:**
- **Demon Slayer** — massively underpredicted. The model had no way to account
  for its devoted fanbase and the cultural weight of the anime community.
- **Wake Up Dead Man** — overpredicted. Strong engagement metrics but a
  niche audience that didn't translate to wide box office.
- **Krishnavataram** — underpredicted. Indian cinema operates on different
  box office dynamics that the model was not trained to recognise.

These misses point to what is missing from the model: franchise loyalty,
cultural context, release timing, competition, and critic reviews.

---

## Limitations

**TextBlob is a blunt instrument.** It was designed for product reviews and
formal text. YouTube comments are full of slang, caps lock, memes, and
sarcasm that TextBlob misreads. A fine-tuned model like VADER or a
transformer-based classifier would perform better.

**Only 100 comments per trailer.** The most popular trailers have hundreds
of thousands of comments. 100 is a sample, not a census — and the YouTube
API returns top comments by default, which are already biased toward
highly liked and therefore positive comments.

**Revenue data is incomplete.** TMDB only has box office revenue for films
with wide theatrical releases. Streaming-first films, limited releases, and
many foreign films were excluded, which skews the dataset toward big studio
blockbusters.

**Correlation is not causation.** High-budget films get bigger marketing
pushes, which drives more trailer views, which correlates with revenue —
but it is the budget driving both, not the views driving the revenue.

---

## If I Were to Continue This

- Use VADER instead of TextBlob for sentiment (better suited to social media text)
- Collect comments at multiple time windows (day 1, week 1, month 1 after release)
- Add critic review scores from Rotten Tomatoes or Metacritic
- Include release timing features (summer vs winter, competition that weekend)
- Try gradient boosting (XGBoost) for better predictive performance
- Build a small web app: input a trailer URL, get a revenue estimate

---

## Conclusion

YouTube trailer reactions contain a real but weak signal about box office
performance. Views, likes, and positive sentiment all trend in the right
direction, but none of them are strong enough to be reliable predictors
on their own. Budget remains the single strongest predictor — studios
that spend more, earn more.

The most honest answer to the original question is: trailer reactions tell
you something, but not enough on their own. A model built purely on YouTube
metrics would be a poor forecasting tool. A model that combines these signals
with budget, franchise history, and release timing would likely do
considerably better — and that is the natural next step for this research.

---

## Project Structure

```
trailer_project/
|
|-- src/
|   |-- get_movies.py          collect movie IDs from TMDB
|   |-- get_details.py         collect full movie details
|   |-- get_trailers.py        find YouTube trailer URLs
|   |-- youtube_stats.py       collect views, likes, comment count
|   |-- get_comments.py        download comment text
|   |-- sentiment_analysis.py  sentiment analysis and dataset merge
|   |-- visualization.py       exploratory data analysis charts
|   `-- predictive_models.py   machine learning models and evaluation
|
|-- data/
|   |-- movies.json
|   |-- movies_details.csv
|   |-- movies_with_trailers.csv
|   |-- movies_with_yt_stats.csv
|   |-- trailer_comments.csv
|   |-- comments_with_sentiment.csv
|   `-- movies_with_sentiment.csv
|
|-- charts/
|   |-- 01_revenue_distribution.png
|   |-- 02_views_vs_revenue.png
|   |-- 03_sentiment_vs_revenue.png
|   |-- 04_sentiment_breakdown_vs_revenue.png
|   |-- 05_correlation_heatmap.png
|   |-- 06_top15_movies.png
|   |-- 07_model_comparison.png
|   |-- 08_feature_importance.png
|   `-- 09_actual_vs_predicted.png
|
|-- config.py                  API keys (not committed to version control)
|-- .env.example               template for required environment variables
|-- requirements.txt
`-- README.md
```

---

## How to Reproduce

```bash
# Install dependencies
pip install -r requirements.txt

# Download TextBlob language data
python -m textblob.download_corpora

# Add your API keys to config.py

# Run scripts in order from the src/ folder
python src/get_movies.py
python src/get_details.py
python src/get_trailers.py
python src/youtube_stats.py
python src/get_comments.py
python src/sentiment_analysis.py
python src/visualization.py
python src/predictive_models.py
```

Two API keys are required: TMDB (free at themoviedb.org) and YouTube Data API v3
(free via Google Cloud Console). Both have generous free tiers sufficient for
this project.