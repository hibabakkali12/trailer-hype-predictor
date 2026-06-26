"""
Builds three models to predict box office revenue from trailer reaction features:
a mean baseline, a linear regression, and a random forest. Evaluates each using
5-fold cross validation and reports RMSE and R2. Produces 3 charts: model comparison,
feature importance, and actual vs predicted revenue.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict , KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from visualization import save
INPUT_FILE="data/movies_with_sentiment.csv"
CHARTS_DIR  = "charts/"
SEED=42
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
if __name__=="__main__":
    df=pd.read_csv(INPUT_FILE)
    FEATURES=[
        "log_views",
        "log_likes",
        "comment_count",
        "engagement_rate",
        "avg_sentiment",
        "positive_comments_pct",
        "negative_comments_pct",
        "avg_subjectivity",
        "popularity",
        "vote_average",
        "budget",
    ]
    TARGET="log_revenue"
    df_model=df[FEATURES + [TARGET, "title"]].dropna().reset_index(drop=True)
    X=df_model[FEATURES].values
    y=df_model[TARGET].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    #MODEL1: Mean Predictor
    print("MODEL 1: Baseline(Mean Predictor)")
    y_mean=y.mean()
    y_baseline=np.full(len(y), y_mean)
    baseline_rmse=np.sqrt(mean_squared_error(y, y_baseline))
    baseline_r2   = r2_score(y, y_baseline)
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"R² : {baseline_r2:.4f}\n")
    #MODEL 2: Linear Regression
    print("MODEL 2: Linear Regression")
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)
    lr_model=LinearRegression()
    lr_rmse_scores=cross_val_score(lr_model,X_scaled,y,
                                   cv=kf,
                                   scoring="neg_root_mean_squared_error")*-1
    lr_r2_scores=cross_val_score(
        lr_model, X_scaled, y,
        cv=kf,
        scoring="r2")
    lr_rmse=lr_rmse_scores.mean()
    lr_r2=lr_r2_scores.mean()
    print(f"RMSE: {lr_rmse:.4f}")
    print(f"R² : {lr_r2:.4f}")
    lr_model.fit(X_scaled,y)
    coeffs=pd.Series(lr_model.coef_, index=FEATURES).sort_values(ascending=False)
    print("Coefficients:")
    for feat, coef in coeffs.items():
        print(f"{feat:22s} {coef:+.4f}")
    print()
    #MODEL3:Random Forest
    print("MODEL 3: Random Forest")
    rf_model=RandomForestRegressor( 
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=3,
    random_state=SEED
)
    rf_rmse_scores=cross_val_score(
        rf_model,X,y, cv=kf, scoring="neg_root_mean_squared_error"
    )*-1
    rf_r2_scores=cross_val_score(  rf_model, X, y,
        cv=kf,
        scoring="r2"
    )
    rf_rmse = rf_rmse_scores.mean()
    rf_r2   = rf_r2_scores.mean()
    print(f"  RMSE : {rf_rmse:.4f}")
    print(f"  R²   : {rf_r2:.4f}")
    rf_model.fit(X,y)
    importances=pd.Series(
        rf_model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)
    print("Feature importances:")
    for feat, imp in importances.items():
        print(f"{feat:22s}  {imp:.4f} ")
    print()
    #Summary
    print("SUMMARY")
    print(f"  {'Model':<22}  {'RMSE':>8}  {'R²':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    print(f"  {'Baseline':<22}  {baseline_rmse:>8.4f}  {baseline_r2:>8.4f}")
    print(f"  {'Linear Regression':<22}  {lr_rmse:>8.4f}  {lr_r2:>8.4f}")
    print(f"  {'Random Forest':<22}  {rf_rmse:>8.4f}  {rf_r2:>8.4f}")
    improvement = ((baseline_rmse - rf_rmse) / baseline_rmse) * 100
    print(f"\n  Random Forest reduces error by {improvement:.1f}% vs baseline.\n")
    #Visualizations-Model Comparison:
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    models=["Baseline", "Linear\nRegression", "Random\nForest"]
    rmse_values = [baseline_rmse, lr_rmse, rf_rmse]
    r2_values   = [baseline_r2,   lr_r2,   rf_r2]
    colors      = ["#8e12ad", "#574fb1", "#1a0044"]
    bars=axes[0].bar(models,rmse_values,color=colors,edgecolor="white",width=0.5)
    axes[0].set_title("RMSE by Model", fontweight="bold")
    axes[0].set_ylabel("RMSE (log10 units)")
    for bar, val in zip(bars,rmse_values):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.005,
                     f"{val:.3f}", ha="center", fontsize=10)
    bars=axes[1].bar(models, r2_values, color=colors, edgecolor="white",width=0.5)
    axes[1].set_title("R² by Model", fontweight="bold")
    axes[1].set_ylabel("R² Score")
    axes[1].axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    for bar, val in zip(bars, r2_values):
        offset = 0.01 if val >= 0 else -0.03
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     val + offset,
                     f"{val:.3f}", ha="center", fontsize=10)
    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_model_comparison.png")
    #Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    trailer_features={"log_views", "log_likes", "comment_count",
        "engagement_rate", "avg_sentiment", "positive_comments_pct",
        "negative_comments_pct", "avg_subjectivity"}
    bar_colors = [
        "#4C72B0" if f in trailer_features else "#95a5a6"
        for f in importances.index
    ]
    bars = ax.barh(
        importances.index[::-1],
        importances.values[::-1],
        color=bar_colors[::-1],
        edgecolor="white",
    )
    for bar, val in zip(bars, importances.values[::-1]):
        ax.text(bar.get_width()+0.002,
              bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9  )
    legend_elements = [
        Patch(facecolor="#4C72B0", label="Trailer reaction feature"),
        Patch(facecolor="#95a5a6", label="Movie-level feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_xlim(0, importances.max() * 1.15)
    plt.tight_layout()
    save(fig,"08_feature_importance.png")
    #Actual vs Predicted
    rf_model_fresh = RandomForestRegressor(
        n_estimators=200, max_depth=6,
        min_samples_leaf=3, random_state=SEED
    )
    y_pred_cv = cross_val_predict(rf_model_fresh, X, y, cv=kf)
 
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y, y_pred_cv, alpha=0.6, s=60,
               color="#4C72B0", edgecolors="white", linewidths=0.5)
    min_val = min(y.min(), y_pred_cv.min())
    max_val = max(y.max(), y_pred_cv.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color="#C44E52", linewidth=2, linestyle="--", label="Perfect prediction")
 
    errors = np.abs(y - y_pred_cv)
    top_errors_idx = errors.argsort()[-5:]
    for idx in top_errors_idx:
        ax.annotate(
            df_model.iloc[idx]["title"][:20],
            xy=(y[idx], y_pred_cv[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            color="#333333",
        )
 
    cv_r2 = r2_score(y, y_pred_cv)
    ax.text(0.05, 0.92, f"R² = {cv_r2:.3f}", transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", edgecolor="gray"))
 
    ax.set_title("Actual vs Predicted Revenue\n(Random Forest, cross-validation)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Actual log10(Revenue)")
    ax.set_ylabel("Predicted log10(Revenue)")
    ax.legend()
    plt.tight_layout()
    save(fig, "09_actual_vs_predicted.png")
    print(f"\n  Baseline R²       : {baseline_r2:.3f}")
    print(f"Linear Regression : {lr_r2:.3f}")
    print(f"Random Forest R²  : {rf_r2:.3f}")
    if rf_r2 > 0.15:
        print("\n Trailer reactions DO have some predictive power,")
        print("    but budget and popularity explain more variance than sentiment.")
    else:
        print("\n Trailer reactions have very limited predictive power alone.")
        print("Budget is the strongest signal in your data.")