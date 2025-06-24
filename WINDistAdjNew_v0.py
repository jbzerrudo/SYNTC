import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(r"G:\2025\GEVNEW\CSV\ALL_TCWS.csv", low_memory=False)

# Work with TOK_GRADE and TOK_WIND while keeping all columns
df.loc[:, ["TOK_GRADE", "TOK_WIND"]]  # This just ensures you are accessing them correctly

# Remove rows where TOK_GRADE = 0 unless TOK_WIND >= 22
df = df[(df["TOK_GRADE"] >= 2) | ((df["TOK_GRADE"] == 0) & (df["TOK_WIND"] >= 22))]

# Check if any records were removed
print(f"Rows remaining: {df.shape[0]}")

# Extract year from SEASON column
df["YEAR"] = df["SEASON"]

# Function to adjust wind speeds only for TOK_GRADE = 2 and TOK_WIND = 0
def adjust_wind_speeds(year_df):
    """Adjust wind speeds for TOK_GRADE = 2 where TOK_WIND is 0, distributing between 22 and 33 kts."""
    # Select only rows where TOK_GRADE = 2 and TOK_WIND = 0
    zero_wind_rows = year_df[(year_df["TOK_GRADE"] == 2) & (year_df["TOK_WIND"] == 0)]

    if zero_wind_rows.empty:
        return year_df  # No changes if no matching rows

    # Assign wind speeds between 22 and 33 kts, skewed more toward the middle but slightly toward 33 kts
    #new_wind_speeds = np.round(22 + np.random.beta(a=0.995, b=1.285, size=zero_wind_rows.shape[0]) * (33 - 22)).astype(int)
    new_wind_speeds = np.round(22 + np.random.beta(a=0.95, b=1.2, size=zero_wind_rows.shape[0]) * (33 - 22)).astype(int)

    # Ensure all values stay within the 22–33 kts range
    new_wind_speeds = np.clip(new_wind_speeds, 22, 33)

    # Update values in the DataFrame safely
    year_df = year_df.copy()  # Avoid SettingWithCopyWarning
    year_df.loc[zero_wind_rows.index, "TOK_WIND"] = new_wind_speeds

    return year_df

# Apply wind speed adjustments by year
df = df.groupby("YEAR", group_keys=False).apply(adjust_wind_speeds)      

# Remove any remaining rows with TOK_WIND == 0 (just in case)
df = df[df["TOK_WIND"] > 0]

# Save the modified dataset
df.to_csv(r"G:\2025\GEVNEW\SOURCE\AdjustedWinds\ALL_TCWS_mod-7.csv", index=False)

# Plot histogram of adjusted TOK_WIND values
plt.figure(figsize=(10, 6))
bins = np.arange(22, df["TOK_WIND"].max() + 5, 5)
plt.hist(df["TOK_WIND"], bins=bins, edgecolor='black', alpha=0.7)  # Adjust bins for integer values
plt.xlabel("Wind Speed (kts)")
plt.ylabel("Frequency")
plt.title("Histogram of Adjusted TOK_WIND Values")
plt.xticks(bins)  # Ensure proper tick labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()