import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define Base Directory
base_dir = r"G:\2025\GEVNEW"

# Load Data
input_file_path = os.path.join(base_dir, "MCSV", "ForGEVComputation.csv")
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"File not found: {input_file_path}")

df = pd.read_csv(input_file_path, sep=",", dtype={"TOK_WIND": float})

# Handle NaNs in TOK_WIND
if df["TOK_WIND"].isnull().any():
    print("Warning: NaN values found in TOK_WIND. Replacing with median value...")
    df["TOK_WIND"].fillna(df["TOK_WIND"].median(), inplace=True)

wind_speeds = df["TOK_WIND"].values
mean_wind_speed = np.mean(wind_speeds)
print("Mean wind speed:", mean_wind_speed)

# Fit Distributions
fit_distributions = {
    "GEV": stats.genextreme.fit(wind_speeds),
    "Gumbel": stats.gumbel_r.fit(wind_speeds),
    "Weibull": stats.weibull_min.fit(wind_speeds),
    "Exponential": stats.expon.fit(wind_speeds)
}

# Fit Pareto with floc=0
shape, loc, scale = stats.pareto.fit(wind_speeds, floc=0)
if shape < 1.1:
    shape = max(shape, 1.1)  # Ensure shape isn't too close to Exponential
fit_distributions["Pareto"] = (shape, 0, scale)

# Compute Return Levels
return_periods = np.linspace(1.1, 200, 400)
p = 1 - 1 / return_periods

return_levels = {
    "GEV": stats.genextreme.ppf(p, *fit_distributions["GEV"]),
    "Gumbel": stats.gumbel_r.ppf(p, *fit_distributions["Gumbel"]),
    "Weibull": stats.weibull_min.ppf(p, *fit_distributions["Weibull"]),
    "Pareto": stats.pareto.ppf(p, shape, 0, scale),
    "Exponential": stats.expon.ppf(p, *fit_distributions["Exponential"])
}

# Compute Empirical Return Periods
n = len(wind_speeds)
sorted_ws = np.sort(wind_speeds)[::-1]
empirical_return_period = (n + 1) / np.arange(1, n + 1)

df["Rank"] = df["TOK_WIND"].rank(method="max", ascending=False)
df["Empirical Return Period"] = (n + 1) / df["Rank"]

# Save Numerical Data to Excel and CSV
output_dir = os.path.join(base_dir, "GEVWEIBOUT_withError3")
os.makedirs(output_dir, exist_ok=True)

output_excel_file_path = os.path.join(output_dir, "fromActualData.xlsx")
output_csv_file_path = os.path.join(output_dir, "fromActualData.csv")

return_levels_df = pd.DataFrame({
    "Return_Period": return_periods,
    "GEV": return_levels["GEV"],
    "Gumbel": return_levels["Gumbel"],
    "Weibull": return_levels["Weibull"],
    "Pareto": return_levels["Pareto"],
    "Exponential": return_levels["Exponential"]
})

empirical_df = pd.DataFrame({
    "Empirical_Return_Period": empirical_return_period,
    "Sorted_Wind_Speeds": sorted_ws
})

# ----------------- Bootstrap Uncertainty Block -----------------
print("Starting bootstrap for uncertainty estimation...")
n_bootstrap = 1000  # Number of bootstrap resamples
bootstrap_results = {name: [] for name in fit_distributions.keys()}

# Create a mapping from distribution names to scipy stats objects
dist_mapping = {
    "GEV": stats.genextreme,
    "Gumbel": stats.gumbel_r,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Pareto": stats.pareto
}

np.random.seed(42)  # For reproducibility

for i in range(n_bootstrap):
    # Generate bootstrap sample
    resample = np.random.choice(wind_speeds, size=len(wind_speeds), replace=True)
    try:
        # Fit distributions to bootstrap sample
        resample_fits = {}
        for name, dist in dist_mapping.items():
            if name == "Pareto":
                shape_b, loc_b, scale_b = dist.fit(resample, floc=0)
                if shape_b < 1.1:
                    shape_b = max(shape_b, 1.1)  # Ensure shape isn't too close to Exponential
                resample_fits[name] = (shape_b, 0, scale_b)
            else:
                resample_fits[name] = dist.fit(resample)
        
        # Calculate return levels for this bootstrap sample
        for name, params in resample_fits.items():
            levels = dist_mapping[name].ppf(p, *params)
            bootstrap_results[name].append(levels)
    except Exception as e:
        print(f"Bootstrap iteration {i} failed: {e}")
        continue

# Convert results to arrays
for name in bootstrap_results:
    bootstrap_results[name] = np.array(bootstrap_results[name])

# Calculate 2.5% and 97.5% percentiles for confidence intervals
ci_bounds = {}
for name, samples in bootstrap_results.items():
    ci_lower = np.percentile(samples, 2.5, axis=0)
    ci_upper = np.percentile(samples, 97.5, axis=0)
    ci_bounds[name] = (ci_lower, ci_upper)
# ---------------------------------------------------------------

# ----------------- Add Uncertainty Columns -----------------
for name in return_levels.keys():
    return_levels_df[f"{name}_Lower"] = ci_bounds[name][0]
    return_levels_df[f"{name}_Upper"] = ci_bounds[name][1]
# -----------------------------------------------------------

# ----------------- Add Standard Deviation Columns (Optional) -----------------
for name, samples in bootstrap_results.items():
    std_dev = np.std(samples, axis=0)
    return_levels_df[f"{name}_StdDev"] = std_dev
# ----------------------------------------------------------------------------

# ----------------- Specific Return Periods Block -----------------
specific_periods_custom = np.array([1.5, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200])
specific_p_custom = 1 - 1 / specific_periods_custom

# Calculate best estimates
specific_return_levels_custom = {}
for name, dist in dist_mapping.items():
    specific_return_levels_custom[name] = dist.ppf(specific_p_custom, *fit_distributions[name])

# Calculate confidence intervals for specific periods
specific_ci_bounds = {}
specific_stddev = {}

for name in fit_distributions.keys():
    # Extract the samples for this distribution at the specific return periods
    specific_samples = []
    
    # For each bootstrap iteration
    for i in range(len(bootstrap_results[name])):
        # Find the indices in the return_periods array that are closest to our specific_periods_custom
        indices = []
        for sp in specific_periods_custom:
            idx = np.abs(return_periods - sp).argmin()
            indices.append(idx)
        
        # Extract the return levels at those indices
        specific_sample = bootstrap_results[name][i][indices]
        specific_samples.append(specific_sample)
    
    specific_samples = np.array(specific_samples)
    
    # Calculate confidence bounds
    specific_ci_bounds[name] = (
        np.percentile(specific_samples, 2.5, axis=0),
        np.percentile(specific_samples, 97.5, axis=0)
    )
    specific_stddev[name] = np.std(specific_samples, axis=0)

# Build DataFrame
specific_return_df = pd.DataFrame({"Return_Period": specific_periods_custom})
for name in fit_distributions.keys():
    specific_return_df[name] = specific_return_levels_custom[name]
    specific_return_df[f"{name}_Lower"] = specific_ci_bounds[name][0]
    specific_return_df[f"{name}_Upper"] = specific_ci_bounds[name][1]
    specific_return_df[f"{name}_StdDev"] = specific_stddev[name]

# Save to CSV
specific_csv_path = os.path.join(output_dir, "specificReturnLevelsWithUncertainty.csv")
specific_return_df.to_csv(specific_csv_path, index=False)
print(f"Saved specific return levels with uncertainty to: {specific_csv_path}")
# ---------------------------------------------------------------


with pd.ExcelWriter(output_excel_file_path) as writer:
    return_levels_df.to_excel(writer, sheet_name='Return Levels', index=False)
    empirical_df.to_excel(writer, sheet_name='Empirical Data', index=False)

return_levels_df.to_csv(output_csv_file_path, index=False)
empirical_df.to_csv(os.path.join(output_dir, "empiricalData.csv"), index=False)

# Save Specific Return Period Levels
specific_periods = [1.5, 5, 10, 20, 50, 100]
specific_p = 1 - 1 / np.array(specific_periods)
specific_return_levels = {name: stats.genextreme.ppf(specific_p, *fit_distributions[name]) if name != "Pareto" else stats.pareto.ppf(specific_p, shape, 0, scale) for name in fit_distributions.keys()}

specific_return_levels_df = pd.DataFrame({
    "Return Period Years": specific_periods,
    "Wind Speed": specific_return_levels["GEV"]
})
specific_return_levels_df.to_excel(os.path.join(output_dir, "specificReturnLevels.xlsx"), index=False)
specific_return_levels_df.to_csv(os.path.join(output_dir, "specificReturnLevels.csv"), index=False)

# Plot
plt.figure(figsize=(10, 6))
for name, levels in return_levels.items():
    plt.plot(return_periods, levels, label=name, linestyle="--", linewidth=2)
plt.plot(empirical_return_period, sorted_ws, color="black", label="Empirical", linewidth=2)
plt.scatter(df["Empirical Return Period"], df["TOK_WIND"],
            facecolors="white", edgecolors="black", label="Observed Max Wind Speeds", s=50, zorder=5)
plt.scatter(specific_periods, specific_return_levels["GEV"], color='red', label='Return Period', s=50, zorder=6)
plt.xscale("log")
plt.xlabel("Return Period (years)")
plt.ylabel("Maximum Wind Speed (kts)")
plt.title("Return Level Plot: Maximum Wind Speed vs. Return Period")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(title="Distribution", loc='upper left')

# Add citation with small font
plt.figtext(0.99, 0.01, "Zerrudo et al. (2025)", ha='right', va='bottom', fontsize=8)

plt.tight_layout()
# Save Plot
output_image_file_path = os.path.join(output_dir, "GRAPHS", "returnplotActual.png")
os.makedirs(os.path.dirname(output_image_file_path), exist_ok=True)
plt.savefig(output_image_file_path, dpi=300, bbox_inches='tight')
plt.show()
print("Mean wind speed:", mean_wind_speed)
print("Specific Return Period Levels:", specific_return_levels)

# Create a DataFrame with return periods as the first column
specific_return_levels_detailed_df = pd.DataFrame({
    "Return_Period": specific_periods
})

# Add each distribution's values as columns
for dist_name, values in specific_return_levels.items():
    specific_return_levels_detailed_df[dist_name] = values

# Save the detailed return levels to CSV
try:
    specific_return_levels_csv_path = os.path.join(output_dir, "specificReturnLevelsAllDistributions.csv")
    specific_return_levels_detailed_df.to_csv(specific_return_levels_csv_path, index=False)
    print(f"Saved detailed return levels to: {specific_return_levels_csv_path}")
except Exception as e:
    print(f"Error saving CSV: {e}")

# ----------------- Plot Specific Return Periods with Uncertainty -----------------
plt.figure(figsize=(10, 6))
for name in fit_distributions.keys():
    plt.plot(specific_periods_custom, specific_return_df[name], label=f"{name}", linestyle="--", linewidth=2)
    plt.fill_between(specific_periods_custom,
                     specific_return_df[f"{name}_Lower"],
                     specific_return_df[f"{name}_Upper"],
                     alpha=0.2)

plt.xscale("log")
plt.xlabel("Return Period (years)")
plt.ylabel("Maximum Wind Speed (kts)")
plt.title("Specific Return Levels with 95% Uncertainty")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(title="Distribution", loc='upper left')

plt.tight_layout()
# Save separate plot
specific_plot_path = os.path.join(output_dir, "GRAPHS", "specificReturnPlot.png")
plt.savefig(specific_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved specific return level plot to: {specific_plot_path}")
# ---------------------------------------------------------------
