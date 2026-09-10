import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
FILE_PATH = "tsla_2025.csv"          
df = pd.read_csv(FILE_PATH)

print("Initial shape:", df.shape)
print(df.head())
print(df.info())

# ---------------------------------------------------------------------------
# 2. CLEAN COLUMN NAMES & DATA TYPES
# ---------------------------------------------------------------------------
df.columns = [c.strip().replace("/", "_").replace(" ", "_") for c in df.columns]

money_cols = [c for c in df.columns if c.lower() in
              ("open", "high", "low", "close", "close_last", "adj_close")]
for col in money_cols:
    if df[col].dtype == object:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )

# Parse the Date column (handles both YYYY-MM-DD and MM/DD/YYYY formats)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. HANDLE MISSING VALUES
# ---------------------------------------------------------------------------
print("\nMissing values per column:\n", df.isna().sum())

# Drop rows where Date failed to parse (unusable)
df = df.dropna(subset=["Date"])

price_cols = [c for c in df.columns if c not in ("Date", "Volume")]
df[price_cols] = df[price_cols].ffill()
df = df.dropna(subset=price_cols)

# ---------------------------------------------------------------------------
# 4. HANDLE DUPLICATES
# ---------------------------------------------------------------------------
dup_count = df.duplicated(subset="Date").sum()
print(f"\nDuplicate rows (by Date): {dup_count}")
df = df.drop_duplicates(subset="Date", keep="first")

# ---------------------------------------------------------------------------
# 5. HANDLE OUTLIERS (IQR method on daily closing price)
# ---------------------------------------------------------------------------
close_col = "Close" if "Close" in df.columns else "Close_Last"

Q1 = df[close_col].quantile(0.25)
Q3 = df[close_col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df[close_col] < lower_bound) | (df[close_col] > upper_bound)]
print(f"\nPotential outliers detected: {len(outliers)}")

# For a stock price series, extreme values are often real market moves,
# not errors — so we FLAG them rather than delete them.
df["is_outlier"] = (df[close_col] < lower_bound) | (df[close_col] > upper_bound)

# ---------------------------------------------------------------------------
# 6. FEATURE ENGINEERING (useful for visualization/storytelling)
# ---------------------------------------------------------------------------
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Daily_Return_%"] = df[close_col].pct_change() * 100
df["MA_30"] = df[close_col].rolling(window=30).mean()
df["MA_90"] = df[close_col].rolling(window=90).mean()

print("\nCleaned shape:", df.shape)
print(df.describe())

# ---------------------------------------------------------------------------
# 7. VISUALIZATIONS / DASHBOARD
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("TSLA Stock — Data Cleaning & Visualization Dashboard", fontsize=16, fontweight="bold")

# (a) Closing price over time with moving averages
axes[0, 0].plot(df["Date"], df[close_col], label="Close", linewidth=1, color="steelblue")
axes[0, 0].plot(df["Date"], df["MA_30"], label="30-day MA", linewidth=1, color="orange")
axes[0, 0].plot(df["Date"], df["MA_90"], label="90-day MA", linewidth=1, color="green")
axes[0, 0].set_title("Closing Price Trend with Moving Averages")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Price ($)")
axes[0, 0].legend()

# (b) Trading volume over time
axes[0, 1].bar(df["Date"], df["Volume"], color="slategray", width=1)
axes[0, 1].set_title("Trading Volume Over Time")
axes[0, 1].set_xlabel("Date")
axes[0, 1].set_ylabel("Volume")

# (c) Distribution of daily returns
sns.histplot(df["Daily_Return_%"].dropna(), bins=100, kde=True, ax=axes[1, 0], color="purple")
axes[1, 0].set_title("Distribution of Daily Returns (%)")
axes[1, 0].set_xlabel("Daily Return (%)")

# (d) Yearly average closing price (bar chart) with outliers highlighted
yearly_avg = df.groupby("Year")[close_col].mean()
axes[1, 1].bar(yearly_avg.index.astype(str), yearly_avg.values, color="teal")
axes[1, 1].set_title("Average Closing Price by Year")
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Avg Close Price ($)")
axes[1, 1].tick_params(axis="x", rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("tsla_dashboard.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 8. CORRELATION HEATMAP (extra insight)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Daily_Return_%"] if c in df.columns]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Key Metrics")
plt.tight_layout()
plt.savefig("tsla_correlation_heatmap.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 9. EXPORT CLEANED DATA
# ---------------------------------------------------------------------------
df.to_csv("tsla_cleaned.csv", index=False)
print("\nCleaned dataset saved to tsla_cleaned.csv")
print("Dashboard image saved to tsla_dashboard.png")
print("Heatmap image saved to tsla_correlation_heatmap.png")
