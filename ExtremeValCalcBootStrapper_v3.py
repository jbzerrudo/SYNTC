import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Define Base Directory
base_dir = r"D:\2026\SYNTC\SYNTC-main\data"

# Load Data
input_file_path = os.path.join(base_dir, "ForGEVComputation2026.csv")
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"File not found: {input_file_path}")

df = pd.read_csv(input_file_path, sep=",", dtype={"TOK_WIND": float})

# Handle NaNs in TOK_WIND
if df["TOK_WIND"].isnull().any():
    print("Warning: NaN values found in TOK_WIND. Replacing with median value...")
    df["TOK_WIND"] = df["TOK_WIND"].fillna(df["TOK_WIND"].median())  # Fixed: avoid deprecated inplace

wind_speeds = df["TOK_WIND"].values
mean_wind_speed = np.mean(wind_speeds)
print("Mean wind speed:", mean_wind_speed)

# Create a mapping from distribution names to scipy stats objects
dist_mapping = {
    "GEV": stats.genextreme,
    "Gumbel": stats.gumbel_r,
    "Weibull": stats.weibull_min,
    "Exponential": stats.expon,
    "Pareto": stats.pareto
}

# Fit Distributions
fit_distributions = {
    "GEV": stats.genextreme.fit(wind_speeds),
    "Gumbel": stats.gumbel_r.fit(wind_speeds),
    "Weibull": stats.weibull_min.fit(wind_speeds),
    "Exponential": stats.expon.fit(wind_speeds)
}

# Fit Pareto with floc=0
# Shape floor at 1.1 prevents near-exponential tail behavior,
# ensuring the Pareto remains distinguishable from the Exponential fit.
shape, loc, scale = stats.pareto.fit(wind_speeds, floc=0)
if shape < 1.1:
    shape = max(shape, 1.1)
fit_distributions["Pareto"] = (shape, 0, scale)

# ----------------- Goodness-of-Fit Metrics -----------------
print("\n--- Goodness-of-Fit (KS test & AIC) ---")
n = len(wind_speeds)
gof_records = []

for name, params in fit_distributions.items():
    dist = dist_mapping[name]
    # KS test
    ks_stat, ks_pval = stats.kstest(wind_speeds, dist.cdf, args=params)
    # Log-likelihood and AIC
    log_lik = np.sum(dist.logpdf(wind_speeds, *params))
    k = len(params)  # number of fitted parameters
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n) - 2 * log_lik
    gof_records.append({
        "Distribution": name,
        "KS_Statistic": ks_stat,
        "KS_pvalue": ks_pval,
        "LogLikelihood": log_lik,
        "AIC": aic,
        "BIC": bic,
        "n_params": k
    })
    print(f"  {name:12s}  KS={ks_stat:.4f}  p={ks_pval:.4f}  AIC={aic:.1f}  BIC={bic:.1f}")

gof_df = pd.DataFrame(gof_records)
# -----------------------------------------------------------

# Compute Return Levels
return_periods = np.linspace(1.1, 200, 400)
p = 1 - 1 / return_periods

# Use dist_mapping consistently for all distributions
return_levels = {}
for name in fit_distributions:
    return_levels[name] = dist_mapping[name].ppf(p, *fit_distributions[name])

# Compute Empirical Return Periods
sorted_ws = np.sort(wind_speeds)[::-1]
empirical_return_period = (n + 1) / np.arange(1, n + 1)

df["Rank"] = df["TOK_WIND"].rank(method="max", ascending=False)
df["Empirical Return Period"] = (n + 1) / df["Rank"]

# Save Numerical Data to Excel and CSV
output_dir = os.path.join(base_dir, "ExtremeValue_v2")
os.makedirs(output_dir, exist_ok=True)

output_excel_file_path = os.path.join(output_dir, "fromActualData.xlsx")
output_csv_file_path = os.path.join(output_dir, "fromActualData.csv")

return_levels_df = pd.DataFrame({"Return_Period": return_periods})
for name in fit_distributions:
    return_levels_df[name] = return_levels[name]

empirical_df = pd.DataFrame({
    "Empirical_Return_Period": empirical_return_period,
    "Sorted_Wind_Speeds": sorted_ws
})

# ----------------- Bootstrap Uncertainty Block -----------------
print("\nStarting bootstrap for uncertainty estimation...")
n_bootstrap = 1000  # Number of bootstrap resamples
bootstrap_results = {name: [] for name in fit_distributions.keys()}

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
                # Shape floor: prevents near-exponential tail behavior
                if shape_b < 1.1:
                    shape_b = max(shape_b, 1.1)
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

# ----------------- Add Uncertainty & StdDev Columns -----------------
for name in return_levels.keys():
    return_levels_df[f"{name}_Lower"] = ci_bounds[name][0]
    return_levels_df[f"{name}_Upper"] = ci_bounds[name][1]

for name, samples in bootstrap_results.items():
    return_levels_df[f"{name}_StdDev"] = np.std(samples, axis=0)
# --------------------------------------------------------------------

# ----------------- Specific Return Periods Block -----------------
# Unified specific periods (consolidated from two separate blocks in v2)
specific_periods_custom = np.array([1.5, 2, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200])
specific_p_custom = 1 - 1 / specific_periods_custom

# Calculate best estimates using correct distribution for each
specific_return_levels_custom = {}
for name in fit_distributions:
    specific_return_levels_custom[name] = dist_mapping[name].ppf(
        specific_p_custom, *fit_distributions[name]
    )

# Calculate confidence intervals for specific periods
# Optimized: compute index lookup once, then vectorized slice
indices = [np.abs(return_periods - sp).argmin() for sp in specific_periods_custom]
specific_ci_bounds = {}
specific_stddev = {}

for name in fit_distributions.keys():
    specific_samples = bootstrap_results[name][:, indices]
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

# Save main outputs
with pd.ExcelWriter(output_excel_file_path) as writer:
    return_levels_df.to_excel(writer, sheet_name='Return Levels', index=False)
    empirical_df.to_excel(writer, sheet_name='Empirical Data', index=False)

return_levels_df.to_csv(output_csv_file_path, index=False)
empirical_df.to_csv(os.path.join(output_dir, "empiricalData.csv"), index=False)

# Save GOF metrics
gof_csv_path = os.path.join(output_dir, "goodnessOfFit.csv")
gof_df.to_csv(gof_csv_path, index=False)
print(f"Saved goodness-of-fit metrics to: {gof_csv_path}")

# Save Specific Return Period Levels (all distributions, corrected)
# Fixed: uses dist_mapping[name] instead of genextreme for all distributions
specific_periods_short = [1.5, 5, 10, 20, 50, 100]
specific_p_short = 1 - 1 / np.array(specific_periods_short)
specific_return_levels = {
    name: dist_mapping[name].ppf(specific_p_short, *fit_distributions[name])
    for name in fit_distributions.keys()
}

specific_return_levels_detailed_df = pd.DataFrame({
    "Return_Period": specific_periods_short
})
for dist_name, values in specific_return_levels.items():
    specific_return_levels_detailed_df[dist_name] = values

specific_return_levels_detailed_df.to_excel(
    os.path.join(output_dir, "specificReturnLevels.xlsx"), index=False
)
specific_return_levels_detailed_df.to_csv(
    os.path.join(output_dir, "specificReturnLevelsAllDistributions.csv"), index=False
)
print("Saved specific return levels (all distributions).")

# ----------------- Plot 1: Full Return Level Plot -----------------
plt.figure(figsize=(10, 6))
for name, levels in return_levels.items():
    plt.plot(return_periods, levels, label=name, linestyle="--", linewidth=2)
plt.plot(empirical_return_period, sorted_ws, color="black", label="Empirical", linewidth=2)
plt.scatter(df["Empirical Return Period"], df["TOK_WIND"],
            facecolors="white", edgecolors="black", label="Observed Max Wind Speeds", s=50, zorder=5)
plt.scatter(specific_periods_short, specific_return_levels["GEV"],
            color='red', label='GEV Return Levels', s=50, zorder=6)
plt.xscale("log")
plt.xlabel("Return Period (years)")
plt.ylabel("Maximum Wind Speed (kts)")
plt.title("Return Level Plot: Maximum Wind Speed vs. Return Period")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(title="Distribution", loc='upper left')
plt.figtext(0.99, 0.01, "Zerrudo et al. (2025)", ha='right', va='bottom', fontsize=8)
plt.tight_layout()

output_image_file_path = os.path.join(output_dir, "GRAPHS", "returnplotActual.png")
os.makedirs(os.path.dirname(output_image_file_path), exist_ok=True)
plt.savefig(output_image_file_path, dpi=300, bbox_inches='tight')
plt.show()
# ------------------------------------------------------------------

# ----------------- Plot 2: Specific Return Periods with Uncertainty -----------------
plt.figure(figsize=(10, 6))
for name in fit_distributions.keys():
    plt.plot(specific_periods_custom, specific_return_df[name],
             label=f"{name}", linestyle="--", linewidth=2)
    plt.fill_between(specific_periods_custom,
                     specific_return_df[f"{name}_Lower"],
                     specific_return_df[f"{name}_Upper"],
                     alpha=0.2)

plt.xscale("log")
plt.xlabel("Return Period (years)")
plt.ylabel("Maximum Wind Speed (kts)")
plt.title("Specific Return Levels with 95% Bootstrap Confidence Intervals")
plt.grid(True, which="both", linestyle="--", alpha=0.7)
plt.legend(title="Distribution", loc='upper left')
plt.figtext(0.99, 0.01, "Zerrudo et al. (2025)", ha='right', va='bottom', fontsize=8)
plt.tight_layout()

specific_plot_path = os.path.join(output_dir, "GRAPHS", "specificReturnPlot.png")
plt.savefig(specific_plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved specific return level plot to: {specific_plot_path}")
# ---------------------------------------------------------------

# Summary
print("\n--- Summary ---")
print(f"Mean wind speed: {mean_wind_speed:.2f} kts")
print(f"Number of observations: {n}")
print(f"Bootstrap resamples: {n_bootstrap}")
print(f"Output directory: {output_dir}")
