import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
INPUT_FILE  = "data/movies_with_sentiment.csv"
CHARTS_DIR  = "charts/"
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIG_W, FIG_H = 10, 6

def save(fig, filename):
    path=CHARTS_DIR + filename
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)

def millions_formatter(x, pos):
    if x>=1e9:
        return f"${x/1e9:.1f}B"
    elif x>=1e6:
        return f"${x/1e6:.1f}M"
    else:
        return f"${x:.0f}"
if __name__=="__main__":
    df=pd.read_csv(INPUT_FILE)
    fig, axes=plt.subplots(1,2, figsize=(FIG_W*1.2, FIG_H))
    axes[0].hist(df["revenue"]/1e6,bins=30,color="#4C72B0", edgecolor="white")
    axes[0].set_title("Revenue Distribution (Raw)")
    axes[0].set_xlabel("Revenue($M)")
    axes[0].set_ylabel("Number of Movies")

    axes[1].hist(df["log_revenue"], bins=30, color="#55A868", edgecolor="white")
    axes[1].set_title("Revenue Distribution (Log Scale)")
    axes[1].set_xlabel("log10(Revenue)")
    axes[1].set_ylabel("Number of Movies")
    fig.suptitle("Revenue — Raw vs Log Transformed", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "01_revenue_distribution.png")



    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(
        df["log_views"],
        df["log_revenue"],
        alpha=0.6,         
        s=60,               
        color="#4C72B0",
        edgecolors="white",
        linewidths=0.5,
    )
    slope, intercept, r, p, _ = stats.linregress(df["log_views"], df["log_revenue"])
    x_line = np.linspace(df["log_views"].min(), df["log_views"].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#C44E52", linewidth=2, label=f"r = {r:.2f}, p = {p:.3f}")
    ax.set_title("Trailer Views vs Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("log10(YouTube Trailer Views)")
    ax.set_ylabel("log10(Revenue)")
    ax.legend()
    top5 = df.nlargest(5, "revenue")
    for _, row in top5.iterrows():
        ax.annotate(
            row["title"][:20],  
            xy=(row["log_views"], row["log_revenue"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )
    save(fig, "02_views_vs_revenue.png")



    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter(
        df["avg_sentiment"],
        df["log_revenue"],
        alpha=0.6,
        s=60,
        color="#55A868",
        edgecolors="white",
        linewidths=0.5,
    )
    slope, intercept, r, p, _ = stats.linregress(df["avg_sentiment"], df["log_revenue"])
    x_line = np.linspace(df["avg_sentiment"].min(), df["avg_sentiment"].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#C44E52", linewidth=2, label=f"r = {r:.2f}, p = {p:.3f}")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.01, ax.get_ylim()[0] + 0.1, "neutral", fontsize=8, color="gray")
    ax.set_title("Average Comment Sentiment vs Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Sentiment Polarity (-1 = very negative, +1 = very positive)")
    ax.set_ylabel("log10(Revenue)")
    ax.legend()
    save(fig, "03_sentiment_vs_revenue.png")


    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 1.2, FIG_H))
    for ax, col, color, label in [
        (axes[0], "positive_comments_pct", "#55A868", "% Positive Comments"),
        (axes[1], "negative_comments_pct", "#C44E52", "% Negative Comments"),
    ]:
        ax.scatter(df[col], df["log_revenue"], alpha=0.6, s=60,
                   color=color, edgecolors="white", linewidths=0.5)
 
        slope, intercept, r, p, _ = stats.linregress(df[col], df["log_revenue"])
        x_line = np.linspace(df[col].min(), df[col].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black",
                linewidth=2, label=f"r = {r:.2f}, p = {p:.3f}")
 
        ax.set_title(f"{label} vs Revenue", fontsize=12, fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylabel("log10(evenue)")
        ax.legend(fontsize=9)
    plt.tight_layout()
    save(fig, "04_sentiment_breakdown_vs_revenue.png")


    corr_cols = [
        "log_revenue",
        "log_views",
        "log_likes",
        "comment_count",
        "engagement_rate",
        "avg_sentiment",
        "positive_comments_pct",
        "negative_comments_pct",
        "avg_subjectivity",
        "vote_average",
        "popularity",
        "budget",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr_matrix = df[corr_cols].corr()
    label_map = {
        "log_revenue":      "Revenue (log)",
        "log_views":        "Views (log)",
        "log_likes":        "Likes (log)",
        "comment_count":    "Comment Count",
        "engagement_rate":  "Engagement Rate",
        "avg_sentiment":    "Avg Sentiment",
        "positive_comments_pct":"% Positive",
        "negative_comments_pct":"% Negative",
        "avg_subjectivity": "Subjectivity",
        "vote_average": "TMDB Rating",
        "popularity": "Popularity",
        "budget":"Budget",
    }
    corr_matrix = corr_matrix.rename(index=label_map, columns=label_map)
 
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,          
        fmt=".2f",           
        cmap="RdYlGn",      
        center=0,           
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Matrix — All Variables", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_correlation_heatmap.png")



    top15 = df.nlargest(15, "revenue").sort_values("revenue")
 
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H * 1.2))
 
    bars = ax.barh(
        top15["title"].str[:35],  
        top15["revenue"],
        color=sns.color_palette("Blues_d", len(top15)),
    )
    for bar, val in zip(bars, top15["revenue"]):
        ax.text(
            bar.get_width() + 5e6,
            bar.get_y() + bar.get_height() / 2,
            millions_formatter(val, None),
            va="center",
            fontsize=8,
        )
 
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(millions_formatter))
    ax.set_title("Top 15 Movies by Revenue", fontsize=14, fontweight="bold")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("")
    plt.tight_layout()
    save(fig, "06_top15_movies.png")
 



