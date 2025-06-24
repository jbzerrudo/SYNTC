import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Load the dataset (keeping all columns but focusing on TOK_WIND)
file_path = r"G:\2025\GEVNEW\CSV\ALL_TCWS.csv"
df = pd.read_csv(file_path, low_memory=False)
df.loc[:, ["TOK_GRADE", "TOK_WIND"]]  # This just ensures you are accessing them correctly

# Ensure TOK_WIND is numeric
df["TOK_WIND"] = pd.to_numeric(df["TOK_WIND"], errors="coerce")

# Define wind speed categories
categories = {
    "TS": (34, 47),
    "STS": (48, 63),
    "TY": (64, 99),
    "STY": (100, df["TOK_WIND"].max())  # Max ensures full range
}

# Initialize Beta parameter dictionary
beta_params = {}

# Set up plotting
plt.figure(figsize=(12, 8))

for i, (category, (low, high)) in enumerate(categories.items(), 1):
    subset = df[(df["TOK_WIND"] >= low) & (df["TOK_WIND"] <= high)]["TOK_WIND"].dropna()
    
    if subset.empty:
        continue  # Skip empty categories

    # Normalize wind speeds to [0,1] for Beta fitting
    norm_speeds = (subset - low) / (high - low)

    # Fit Beta distribution
    try:
        a, b, _, _ = beta.fit(norm_speeds, floc=0, fscale=1)
        beta_params[category] = (a, b)
    except:
        print(f"Warning: Could not fit Beta distribution for {category}")
        continue

    # Plot histogram and fitted Beta PDF
    plt.subplot(2, 2, i)
    plt.hist(norm_speeds, bins=15, density=True, alpha=0.6, color="b", edgecolor="black", label="Data")
    
    # Generate Beta curve
    x = np.linspace(0, 1, 100)
    y = beta.pdf(x, a, b)
    plt.plot(x, y, "r-", label=f"Beta Fit (a={a:.2f}, b={b:.2f})")

    plt.title(f"{category} ({low}-{high} kts)")
    plt.xlabel("Normalized Wind Speed")
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout()
plt.show()

# Display fitted parameters
beta_params