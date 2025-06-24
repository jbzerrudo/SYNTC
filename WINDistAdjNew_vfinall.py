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
    """
    Adjust wind speeds for TOK_GRADE = 2 where TOK_WIND is 0,
    creating a U-shaped distribution in the TD category (22-33 kt)
    without affecting the TS category.
    """
    # Select only rows where TOK_GRADE = 2 and TOK_WIND = 0
    zero_wind_rows = year_df[(year_df["TOK_GRADE"] == 2) & (year_df["TOK_WIND"] == 0)]
    
    if zero_wind_rows.empty:
        return year_df  # No changes if no matching rows
    
    # Create a U-shaped distribution within ONLY the TD range (22-33 kt)
    n_rows = zero_wind_rows.shape[0]
    
    # Create a TD-only distribution with 6 points that should create 5-7 bars
    # Important: Make sure values stay strictly within TD category
    values = [22, 24, 26, 28, 30, 32]  # Note: Using 32 as max instead of 33 to avoid boundary effects
    
    # U-shaped weights - higher at the edges, lower in the middle
    weights = [0.3, 0.1, 0.08, 0.08, 0.1, 0.34]
    
    # Randomly select based on weights with NO noise added
    indices = np.random.choice(len(values), size=n_rows, p=weights)
    new_wind_speeds = np.array(values)[indices]
    
    # Update values in the DataFrame safely
    year_df = year_df.copy()  # Avoid SettingWithCopyWarning
    year_df.loc[zero_wind_rows.index, "TOK_WIND"] = new_wind_speeds
    
    return year_df

# Apply wind speed adjustments by year
df = df.groupby("YEAR", group_keys=False).apply(adjust_wind_speeds)      

# Remove any remaining rows with TOK_WIND == 0 (just in case)
df = df[df["TOK_WIND"] > 0]

# Save the modified dataset
df.to_csv(r"G:\2025\GEVNEW\SOURCE\AdjustedWinds\ALL_TCWS_mod-4.csv", index=False)

# Plot histogram of adjusted TOK_WIND values
plt.figure(figsize=(10, 6))
bins = np.arange(22, df["TOK_WIND"].max() + 5, 5)
plt.hist(df["TOK_WIND"], bins=bins, edgecolor='black', alpha=0.7)
plt.xlabel("Wind Speed (kts)")
plt.ylabel("Frequency")
plt.title("Histogram of Adjusted TOK_WIND Values")
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()