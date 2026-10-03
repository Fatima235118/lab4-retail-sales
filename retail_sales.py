import pandas as pd
import numpy as np
from scipy.stats import zscore

# ========================================
# Task 2: Retail Sales
# ========================================

# 10-day retail sales dataset
data = {
    "Day": range(1, 11),
    "Revenue": [
        45, 52, np.nan, 48, 390,
        55, np.nan, 50, 47, 53
    ]
}

df = pd.DataFrame(data)

# ========================================
# 1. Detect Missing Values
# ========================================

print("Missing values in Revenue:", df["Revenue"].isnull().sum())

# ========================================
# 2. Fill Missing Values with Median
# ========================================

median_revenue = df["Revenue"].median()
df["Revenue"] = df["Revenue"].fillna(median_revenue)

print("\nAfter median fill:")
print(df)

# ========================================
# 3. Min-Max Normalization
# ========================================

df["Revenue_MinMax"] = (
    (df["Revenue"] - df["Revenue"].min())
    / (df["Revenue"].max() - df["Revenue"].min())
)

# ========================================
# 4. Z-score Standardization
# ========================================

df["Revenue_Zscore"] = zscore(df["Revenue"])

# ========================================
# 5. IQR Outlier Detection
# ========================================

Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["Outlier"] = (
    (df["Revenue"] < lower_bound)
    | (df["Revenue"] > upper_bound)
)

# ========================================
# 6. Print Outlier Days
# ========================================

outlier_days = df.loc[df["Outlier"], "Day"].tolist()

print("\nIQR outliers detected on day(s):", outlier_days)

# ========================================
# 7. Final Table
# ========================================

print("\nFinal Table:")
print(df)