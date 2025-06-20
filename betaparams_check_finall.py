import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import warnings

# Suppress scipy warning messages but keep the important ones
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the dataset
file_path = r"G:\2025\GEVNEW\CSV\ALL_TCWS.csv"
df = pd.read_csv(file_path, low_memory=False)

# Ensure TOK_WIND is numeric
df["TOK_WIND"] = pd.to_numeric(df["TOK_WIND"], errors="coerce")

# Define wind speed categories
categories = {
    "TD": (22,33),
    "TS": (34, 47),
    "STS": (48, 63),
    "TY": (64, 99),
    "STY": (100, df["TOK_WIND"].max())
}

# Initialize Beta parameter dictionary
beta_params = {}

# Set up plotting
plt.figure(figsize=(15, 10))

# Add a diagnostic subplot for raw data
plt.subplot(3, 2, 1)
plt.hist(df["TOK_WIND"].dropna(), bins=30, color="green", alpha=0.7)
plt.title("Overall Wind Speed Distribution")
plt.xlabel("Wind Speed (kts)")
plt.ylabel("Count")

# Print some diagnostic information
print("Dataset Statistics:")
print(f"Total records: {len(df)}")
print(f"Records with valid TOK_WIND: {df['TOK_WIND'].notna().sum()}")
print(f"TOK_WIND range: {df['TOK_WIND'].min()} to {df['TOK_WIND'].max()} kts")
print("\nCategory counts:")

for i, (category, (low, high)) in enumerate(categories.items(), 2):
    subset = df[(df["TOK_WIND"] >= low) & (df["TOK_WIND"] <= high)]["TOK_WIND"].dropna()
    
    print(f"{category} ({low}-{high} kts): {len(subset)} records")
    
    if len(subset) < 3:
        print(f"Warning: Not enough data for {category} (only {len(subset)} records)")
        plt.subplot(3, 2, i)
        plt.text(0.5, 0.5, f"Insufficient data\n({len(subset)} records)", 
                 horizontalalignment='center', verticalalignment='center')
        plt.title(f"{category} ({low}-{high} kts)")
        continue

    # Normalize wind speeds to [0,1] for Beta fitting
    norm_speeds = (subset - low) / (high - low)
    
    # Ensure values are strictly within (0,1)
    epsilon = 1e-6
    norm_speeds = norm_speeds.clip(epsilon, 1-epsilon)
    
    # Plot normalized data histogram
    plt.subplot(3, 2, i)
    plt.hist(norm_speeds, bins=20, density=True, alpha=0.6, color="b", 
             edgecolor="black", label="Data")
    
    # Try Beta fitting with different methods
    try:
        # First try MLE fitting
        a, b, loc, scale = beta.fit(norm_speeds, floc=0, fscale=1)
        fit_method = "MLE"
        
        # If parameters look suspicious (very high values), try method of moments
        if a > 100 or b > 100:
            # Method of moments for Beta
            mean = norm_speeds.mean()
            var = norm_speeds.var()
            
            # Beta parameters using method of moments
            common_factor = mean * (1 - mean) / var - 1
            a = mean * common_factor
            b = (1 - mean) * common_factor
            fit_method = "Moments"
        
        beta_params[category] = (a, b)
        
        # Generate Beta curve
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, a, b)
        plt.plot(x, y, "r-", label=f"Beta Fit (a={a:.2f}, b={b:.2f})")
        print(f"{category}: Fitted with {fit_method} method: a={a:.4f}, b={b:.4f}")
        
    except Exception as e:
        print(f"Error fitting {category}: {str(e)}")
        # If standard fit fails, try method of moments directly
        try:
            mean = norm_speeds.mean()
            var = norm_speeds.var()
            
            # Ensure variance is valid for moment estimation
            if var >= mean * (1 - mean):
                print(f"  Variance too high for moment estimation in {category}")
                raise ValueError("Invalid variance for Beta parameters")
                
            common_factor = mean * (1 - mean) / var - 1
            a = mean * common_factor
            b = (1 - mean) * common_factor
            
            beta_params[category] = (a, b)
            
            # Generate Beta curve
            x = np.linspace(0, 1, 100)
            y = beta.pdf(x, a, b)
            plt.plot(x, y, "g-", label=f"Moments (a={a:.2f}, b={b:.2f})")
            print(f"{category}: Fitted with moments: a={a:.4f}, b={b:.4f}")
            
        except Exception as e2:
            print(f"  Failed moment estimation for {category}: {str(e2)}")
    
    plt.title(f"{category} ({low}-{high} kts)")
    plt.xlabel("Normalized Wind Speed")
    plt.ylabel("Density")
    plt.legend()

plt.tight_layout()
plt.savefig("wind_speed_beta_fits.png")  # Save the figure
plt.show()

# Display fitted parameters
print("\nFinal Beta Parameters:")
for category, params in beta_params.items():
    print(f"{category}: a={params[0]:.4f}, b={params[1]:.4f}")