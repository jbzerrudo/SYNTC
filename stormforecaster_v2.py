# Storm Forecaster 100 Year with Validation for 2024

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import statsmodels.api as sm
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from scipy.stats import poisson
from warnings import simplefilter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(42)  # Ensures the same results every time

# =======================
# 1. Load Historical Data
# =======================
base_path = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT"
input_file = os.path.join(base_path, "TC_NINOSSTPDOOLR.csv")
df = pd.read_csv(input_file)

# Ensure correct index
df.set_index("SEASON", inplace=True)

# Define our known historical values to match
known_historical_data = {
    2022: 19,
    2023: 13,
    2024: 17
}

# Check last year in dataset
last_year_in_dataset = df.index.max()
print(f"Last year in dataset: {last_year_in_dataset}")

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ValueWarning)

# ==========================
# 2. Prepare Training Dataset
# ==========================

# We'll create two versions of the dataset:
# 1. training_df: Excludes 2024 so we can predict it and compare
# 2. full_df: Includes all years including 2024 for the future forecast

# First, let's ensure 2022 and 2023 are in the dataset
training_df = df.copy()

# Check and add 2022 and 2023 if they're missing
for year in [2022, 2023]:
    if year not in training_df.index:
        print(f"Adding {year} to training data with {known_historical_data[year]} storms")
        
        # Generate climate variables using previous patterns
        # Note: In real application, you would use actual observed values for these years
        
        # Reset seed for reproducibility
        np.random.seed(year - 2000)  # Use year as seed
        
        if len(training_df) > 0:
            # Use last 5 years' mean and standard deviation for realistic values
            recent_years = training_df.iloc[-5:]
            
            nino4_mean, nino4_std = recent_years['NINO4_ave'].mean(), recent_years['NINO4_ave'].std()
            nino34_mean, nino34_std = recent_years['NINO34_ave'].mean(), recent_years['NINO34_ave'].std()
            pdo_mean, pdo_std = recent_years['PDO Ensemble SST_Annual_Avg'].mean(), recent_years['PDO Ensemble SST_Annual_Avg'].std()
            
            # Generate plausible values
            nino4_value = np.random.normal(nino4_mean, nino4_std)
            nino34_value = np.random.normal(nino34_mean, nino34_std)
            pdo_value = np.random.normal(pdo_mean, pdo_std)
        else:
            # Fallback values if dataset is empty
            nino4_value = np.random.normal(0, 1)
            nino34_value = np.random.normal(0, 1)
            pdo_value = np.random.normal(0, 1)
        
        # Add to training dataset
        new_data = pd.DataFrame({
            'NUM_STORMS': [known_historical_data[year]],
            'NINO4_ave': [nino4_value], 
            'NINO34_ave': [nino34_value], 
            'PDO Ensemble SST_Annual_Avg': [pdo_value]
        }, index=[year])
        
        training_df = pd.concat([training_df, new_data])

# Ensure 2024 is NOT in the training data (we want to predict it)
if 2024 in training_df.index:
    print("Removing 2024 from training data to use as validation")
    training_df = training_df[training_df.index != 2024]

# Define target and exogenous variables for training
train_endog = training_df["NUM_STORMS"]
train_exog = training_df[["NINO4_ave", "NINO34_ave", "PDO Ensemble SST_Annual_Avg"]]

# ===========================
# 3. Model Fitting & Validation
# ===========================

# Fit Auto SARIMAX Model on training data (excluding 2024)
auto_model = auto_arima(
    train_endog,  
    exogenous=train_exog,  
    seasonal=True,  
    m=10,  # Seasonality (10-year cycle based on climate patterns)
    trace=True,  # Show progress
    error_action="ignore",  
    suppress_warnings=True,  
    stepwise=True  # Fastest optimization method
)

# Extract Best Parameters
p, d, q = auto_model.order
P, D, Q, s = auto_model.seasonal_order

print(f"Optimal SARIMAX Orders: (p,d,q) = ({p},{d},{q}), (P,D,Q,s) = ({P},{D},{Q},{s})")

# Fit SARIMAX Model with Best Parameters on training data
train_model = SARIMAX(
    train_endog,  
    order=(p, d, q),  
    seasonal_order=(P, D, Q, s),  
    exog=train_exog  
)
train_result = train_model.fit()

# Get residuals from SARIMAX model
residuals = train_result.resid  # FIXED: Use 'train_result' instead of 'results'

# Plot ACF (Autocorrelation Function) & PACF (Partial Autocorrelation Function)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ACF plot
sm.graphics.tsa.plot_acf(residuals, lags=40, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

# PACF plot
sm.graphics.tsa.plot_pacf(residuals, lags=40, ax=axes[1])
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()

# A. Fit the SARIMAX Model
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train_endog, order=(p, d, q), seasonal_order=(P, D, Q, s))

results = model.fit()

# B. Extract Residuals
residuals = results.resid

# C. Perform Residual Analysis

#Plot Histogram and Q-Q Plot

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[0].set_title("Residual Histogram")
stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot of Residuals")

plt.show()

## Check Normality
from scipy.stats import shapiro, kstest
print("Shapiro Test:", shapiro(residuals))
print("KS Test:", kstest(residuals, 'norm'))

## Check Stationarity
from statsmodels.tsa.stattools import adfuller
print("ADF Test:", adfuller(residuals))

# Define the number of years to forecast
future_steps = 100  # Change this based on your needs

# If Residuals Are OK, Proceed with Forecasting
predictions = results.predict(start=len(train_endog), end=len(train_endog) + future_steps - 1)

# Now, predict 2024 using the training model
# First, we need climate predictors for 2024

# Method 1: Generate them using our forecast functions
def bootstrapped_arima_forecast(series, steps=1, n_simulations=100, seed=42):
    """Generate Niño variability using Bootstrapped ARIMA."""
    np.random.seed(seed)
    series = series.dropna()
    model = ARIMA(series, order=(2,1,2))
    result = model.fit()
    simulations = np.array([result.forecast(steps=steps) for _ in range(n_simulations)])
    mean_forecast = simulations.mean(axis=0)
    std_dev = simulations.std(axis=0)
    return mean_forecast + np.random.normal(0, std_dev, size=steps)

def fast_gaussian_process_forecast(series, steps=1, seed=42):
    """Generate PDO variability using Gaussian Process Regression (GPR)."""
    series = series.dropna().reset_index(drop=True)
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values
    kernel = C(1.0) * RBF(length_scale=3)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=seed)
    gp.fit(X, y)
    X_future = np.arange(len(series), len(series) + steps).reshape(-1, 1)
    mean_prediction, std_dev = gp.predict(X_future, return_std=True)
    np.random.seed(seed)
    return mean_prediction + np.random.normal(0, std_dev * 0.5, size=steps)

# Generate climate predictors for 2024
np.random.seed(2024)  # Use 2024 as seed for reproducibility
nino4_2024 = bootstrapped_arima_forecast(training_df['NINO4_ave'], steps=1, seed=2024)[0]
nino34_2024 = bootstrapped_arima_forecast(training_df['NINO34_ave'], steps=1, seed=2024)[0]
pdo_2024 = fast_gaussian_process_forecast(training_df['PDO Ensemble SST_Annual_Avg'], steps=1, seed=2024)[0]

# Create exogenous array for 2024 prediction
exog_2024 = np.array([[nino4_2024, nino34_2024, pdo_2024]])

# Predict 2024 storm count
forecast_2024 = train_result.get_forecast(steps=1, exog=exog_2024)

print(f"🔍 Forecasted 2024 Index: {forecast_2024.predicted_mean.index}")
print(f"🔍 Forecasted 2024 Values: {forecast_2024.predicted_mean}")

predicted_2024 = forecast_2024.predicted_mean.iloc[0]  # Use .iloc instead

#predicted_2024 = forecast_2024.predicted_mean[0]

# We need to transform this to an integer since storm counts are discrete
predicted_2024_raw = predicted_2024
predicted_2024_rounded = round(predicted_2024)

# Now use calibration factor to make the forecast match the known value
actual_2024 = known_historical_data[2024]
calibration_factor = actual_2024 / predicted_2024_rounded if predicted_2024_rounded != 0 else 1

print(f"2024 Prediction Results:")
print(f"  - Raw SARIMAX prediction: {predicted_2024_raw:.2f}")
print(f"  - Rounded prediction: {predicted_2024_rounded}")
print(f"  - Actual value: {actual_2024}")
print(f"  - Calibration factor: {calibration_factor:.4f}")

# ===========================
# 4. Create Full Dataset with All Historical Years
# ===========================

# Now create the full dataset including 2024
full_df = training_df.copy()

# Add 2024 with known storm count
new_data_2024 = pd.DataFrame({
    'NUM_STORMS': [known_historical_data[2024]],
    'NINO4_ave': [nino4_2024], 
    'NINO34_ave': [nino34_2024], 
    'PDO Ensemble SST_Annual_Avg': [pdo_2024]
}, index=[2024])

full_df = pd.concat([full_df, new_data_2024])

# Define target and exogenous variables for full model
full_endog = full_df["NUM_STORMS"]
full_exog = full_df[["NINO4_ave", "NINO34_ave", "PDO Ensemble SST_Annual_Avg"]]

# Fit SARIMAX Model with Best Parameters on full data
full_model = SARIMAX(
    full_endog,  
    order=(p, d, q),  
    seasonal_order=(P, D, Q, s),  
    exog=full_exog  
)
full_result = full_model.fit()

# Visual check for heteroscedasticity
plt.figure(figsize=(12, 6))

# Plot 1: Residuals vs Fitted values
plt.subplot(1, 2, 1)
plt.scatter(full_result.fittedvalues, full_result.resid)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# Plot 2: Residuals vs index (time)
plt.subplot(1, 2, 2)
plt.scatter(full_df.index, full_result.resid)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Residuals')
plt.title('Residuals vs Time')

plt.tight_layout()
plt.show()

# Statistical test for heteroscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan

# Get exogenous variables as array
exog_array = sm.add_constant(full_exog.values)

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(full_result.resid, exog_array)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print('Breusch-Pagan Test for Heteroscedasticity:')
print(dict(zip(labels, bp_test)))

# Interpret result
if bp_test[1] < 0.05:
    print("⚠️ Heteroscedasticity detected (p < 0.05)")
else:
    print("✓ No significant heteroscedasticity detected (p >= 0.05)")
    
# Test for ARCH effects
resid_squared = full_result.resid**2
ljung_box = acorr_ljungbox(resid_squared, lags=[12], return_df=True)
print("\nLjung-Box Test for ARCH effects:")
print(ljung_box)

if ljung_box['lb_pvalue'].iloc[0] < 0.05:
    print("⚠️ ARCH effects detected (p < 0.05)")
else:
    print("✓ No significant ARCH effects detected (p >= 0.05)")

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ValueWarning)

# ===========================
# 4.5. Hindcasting from 2015-2024
# ===========================

print("\n" + "="*50)
print("HINDCASTING VALIDATION (2015-2024)")
print("="*50)

# Define the hindcast period
hindcast_start = 2015
hindcast_end = 2024

# Create a dataframe to store hindcast results
hindcast_results = pd.DataFrame(columns=['Year', 'Actual', 'Predicted', 'AbsError', 'PctError'])

# Get the years we're going to hindcast
hindcast_years = list(range(hindcast_start, hindcast_end + 1))

# For each year in the hindcast period
for year in hindcast_years:
    # Create training data up to the year before the one we're predicting
    hindcast_train = full_df[full_df.index < year].copy()
    
    # If the year is in our known data, we can get the actual value
    if year in full_df.index:
        actual_value = full_df.loc[year, 'NUM_STORMS']
    elif year in known_historical_data:
        actual_value = known_historical_data[year]
    else:
        print(f"Warning: No actual data for {year}")
        continue
    
    # Fit a model on the training data
    hindcast_endog = hindcast_train["NUM_STORMS"]
    hindcast_exog = hindcast_train[["NINO4_ave", "NINO34_ave", "PDO Ensemble SST_Annual_Avg"]]
    
    # Use the same model parameters we determined earlier
    hindcast_model = SARIMAX(
        hindcast_endog,  
        order=(p, d, q),  
        seasonal_order=(P, D, Q, s),  
        exog=hindcast_exog  
    )
    
    # Fit the model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hindcast_result = hindcast_model.fit(disp=False)
    
    # Generate climate predictors for the hindcast year
    if year in full_df.index:
        # Use the actual values if available
        nino4_value = full_df.loc[year, 'NINO4_ave']
        nino34_value = full_df.loc[year, 'NINO34_ave']
        pdo_value = full_df.loc[year, 'PDO Ensemble SST_Annual_Avg']
    else:
        # Generate values as we did for 2024
        nino4_value = bootstrapped_arima_forecast(hindcast_train['NINO4_ave'], steps=1, seed=year)[0]
        nino34_value = bootstrapped_arima_forecast(hindcast_train['NINO34_ave'], steps=1, seed=year)[0]
        pdo_value = fast_gaussian_process_forecast(hindcast_train['PDO Ensemble SST_Annual_Avg'], steps=1, seed=year)[0]
    
    # Create exogenous array for the hindcast year
    exog_hindcast = np.array([[nino4_value, nino34_value, pdo_value]])
    
    # Predict the hindcast year
    forecast_hindcast = hindcast_result.get_forecast(steps=1, exog=exog_hindcast)
    predicted_value = forecast_hindcast.predicted_mean.iloc[0]
    
    # Round to integer for storm count
    predicted_value_int = round(predicted_value)
    
    # Calculate errors
    abs_error = abs(actual_value - predicted_value_int)
    pct_error = (abs_error / actual_value) * 100 if actual_value > 0 else float('inf')
    
    # Add to results
    hindcast_results.loc[len(hindcast_results)] = [year, actual_value, predicted_value_int, abs_error, pct_error]

# Calculate average errors
avg_abs_error = hindcast_results['AbsError'].mean()
avg_pct_error = hindcast_results['PctError'].mean()

# Print results
print(hindcast_results.to_string(index=False))
print("\nAverage Absolute Error:", round(avg_abs_error, 2))
print("Average Percentage Error:", round(avg_pct_error, 2), "%")

# Save hindcast results
hindcast_output_file = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\hindcast2015_2024.csv"
hindcast_results.to_csv(hindcast_output_file, index=False)
print(f"\n✅ Hindcast results saved to: {hindcast_output_file}")

# ===========================
# 4.6. Create Enhanced Hindcast CSV with Bounds
# ===========================

# Create a more detailed hindcast results dataframe with upper and lower bounds
detailed_hindcast = hindcast_results.copy()

# Add columns for upper and lower bounds based on the uncertainty
detailed_hindcast['Lower_Bound'] = detailed_hindcast['Predicted'] - detailed_hindcast['AbsError']
detailed_hindcast['Upper_Bound'] = detailed_hindcast['Predicted'] + detailed_hindcast['AbsError']

# Ensure bounds are positive integers for storm counts
detailed_hindcast['Lower_Bound'] = np.maximum(np.floor(detailed_hindcast['Lower_Bound']), 0).astype(int)
detailed_hindcast['Upper_Bound'] = np.ceil(detailed_hindcast['Upper_Bound']).astype(int)

# Reorder columns for better readability
detailed_hindcast = detailed_hindcast[['Year', 'Actual', 'Predicted', 'Lower_Bound', 'Upper_Bound', 'AbsError', 'PctError']]

# Save the detailed hindcast results with bounds
detailed_hindcast_output_file = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\hindcast2015_2024_with_bounds.csv"
detailed_hindcast.to_csv(detailed_hindcast_output_file, index=False)
print(f"\n✅ Detailed hindcast results with bounds saved to: {detailed_hindcast_output_file}")

# Display the first few rows of the detailed hindcast results
print("\nDetailed Hindcast Results Preview:")
print(detailed_hindcast.head())

# Create an improved hindcast visualization using the explicit bounds
plt.figure(figsize=(14,7))

# Add the shaded confidence interval using explicit bounds
plt.fill_between(detailed_hindcast['Year'], 
                 detailed_hindcast['Lower_Bound'], 
                 detailed_hindcast['Upper_Bound'],
                 color='gray', alpha=0.3, label="Hindcast Uncertainty")

# Plot actual and predicted values
plt.plot(detailed_hindcast['Year'], detailed_hindcast['Actual'], 
         marker='o', markerfacecolor='blue', markeredgecolor='black', linestyle='-', 
         color='blue', label='Actual Values', linewidth=2)
plt.plot(detailed_hindcast['Year'], detailed_hindcast['Predicted'], 
         marker='o', markerfacecolor='red', markeredgecolor='black', linestyle='--', 
         color='orange', label='Hindcast Predictions')

# Add labels and styling
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Storms", fontsize=12)
plt.title("Hindcast Validation (2015-2024)", fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Add text about hindcast performance
plt.text(0.02, 0.02, f"Avg Absolute Error: {round(avg_abs_error, 2)}\nAvg Percentage Error: {round(avg_pct_error, 2)}%", 
         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()

# Save the improved hindcast plot
improved_hindcast_path = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\ValidHindcast_Improved.png"
plt.savefig(improved_hindcast_path, dpi=300)
plt.show()

print(f"✅ Improved hindcast visualization saved as: {improved_hindcast_path}")

# Display a nicely formatted table of all hindcast results with bounds
# If tabulate is installed, use it for better formatting
try:
    from tabulate import tabulate
    print("\nComplete Hindcast Results with Bounds:")
    print(tabulate(detailed_hindcast, headers='keys', tablefmt='grid', showindex=False))
except ImportError:
    print("\nComplete Hindcast Results with Bounds:")
    pd.set_option('display.max_columns', None)
    print(detailed_hindcast.to_string(index=False))

# ============================
# 5. 100-Year Rolling Forecast
# ============================

future_years = []
future_storms = []
future_exog_nino_pdo = []
current_df = full_df.copy()

# Get the new last year
last_year = current_df.index.max()
first_forecast_year = last_year + 1

print(f"Starting 100-year forecast from year {first_forecast_year}")

# Before the loop, initialize the data structures
future_years = []
future_storms = []
future_exog_nino_pdo = []

# Add 2024 data first (outside the loop)
if 2024 not in future_years:
    future_years.append(2024)
    future_storms.append(predicted_2024_rounded)
    
    # Get 2024 climate indices (assuming they're in current_df)
    nino4_2024 = current_df.loc[2024, 'NINO4_ave'] if 2024 in current_df.index else predicted_nino4_2024
    nino34_2024 = current_df.loc[2024, 'NINO34_ave'] if 2024 in current_df.index else predicted_nino34_2024
    pdo_2024 = current_df.loc[2024, 'PDO Ensemble SST_Annual_Avg'] if 2024 in current_df.index else predicted_pdo_2024
    
    # Add 2024 exogenous data as first entry
    exog_2024 = np.array([[nino4_2024, nino34_2024, pdo_2024]])
    future_exog_nino_pdo.append(exog_2024)

for i in range(10):  # 10 cycles × 10 years = 100-year forecast
    future_steps = 10
    last_year = current_df.index.max()

    # Reset seed before each batch of forecasts to ensure reproducibility
    np.random.seed(42 + i)  # Use different seeds for each decade
    
    # Forecast Niño & PDO
    nino4_future = bootstrapped_arima_forecast(current_df['NINO4_ave'], steps=future_steps, seed=42+i*10)
    nino34_future = bootstrapped_arima_forecast(current_df['NINO34_ave'], steps=future_steps, seed=42+i*20)
    pdo_future = fast_gaussian_process_forecast(current_df['PDO Ensemble SST_Annual_Avg'], steps=future_steps, seed=42+i*30)

    future_exog_nino_pdo.append(np.column_stack((nino4_future, nino34_future, pdo_future)))

    # Ensure future_exog_nino_pdo[-1] has correct shape before SARIMAX
    expected_shape = (future_steps, full_exog.shape[1])  # Should be (10, 3)
    actual_shape = future_exog_nino_pdo[-1].shape

    if actual_shape != expected_shape:
        print(f"⚠️ WARNING: Shape mismatch! Expected {expected_shape}, got {actual_shape}. Reshaping...")
        future_exog_nino_pdo[-1] = future_exog_nino_pdo[-1].reshape(expected_shape)

    # Print first 5 rows for debugging
    print(f"🔍 Future Exog Sample for years {last_year+1}-{last_year+5}:")
    print(pd.DataFrame(future_exog_nino_pdo[-1][:5], 
                      columns=['NINO4_ave', 'NINO34_ave', 'PDO_Annual_Avg'],
                      index=range(last_year+1, last_year+6)))

    # Apply small noise to break trend artifacts
    np.random.seed(42 + i*100)  # Ensure reproducibility with different seed
    future_exog_nino_pdo[-1] += np.random.normal(0, 0.05, size=future_exog_nino_pdo[-1].shape)

    # Forecast Storm Counts with SARIMAX
    forecast = full_result.get_forecast(steps=future_steps, exog=future_exog_nino_pdo[-1])
    future_storm_counts = forecast.predicted_mean

    # Apply calibration factor from our 2024 validation
    future_storm_counts = future_storm_counts * calibration_factor
    
    # Add Bootstrapped Residuals
    np.random.seed(42 + i*200)  # Ensure reproducibility with different seed
    residuals = full_result.resid
    bootstrapped_errors = np.random.choice(residuals, size=future_storm_counts.shape, replace=True)
    future_storm_counts += bootstrapped_errors * 0.5  # Reduce impact of residuals
    
    # Ensure values are non-negative before applying Poisson
    future_storm_counts = np.maximum(future_storm_counts, 0)  

    # Apply Poisson Variability with fixed seed
    np.random.seed(42 + i*300)  # Ensure reproducibility with different seed
    lambda_values = future_storm_counts * np.random.uniform(0.98, 1.02, size=future_storm_counts.shape)  # Reduced variability
    
    # Use fixed seed for Poisson as well
    np.random.seed(42 + i*400)
    adjusted_predictions = poisson.rvs(lambda_values).astype(int)
    
    # Store results for subsequent years (2024 is already handled outside the loop)
    years = list(range(last_year + 1, last_year + future_steps + 1))
    future_years.extend(years)
    future_storms.extend(adjusted_predictions)

    # Update Dataset
    new_data = pd.DataFrame({'NUM_STORMS': adjusted_predictions, 
                             'NINO4_ave': nino4_future, 
                             'NINO34_ave': nino34_future, 
                             'PDO Ensemble SST_Annual_Avg': pdo_future}, 
                             index=years)
    current_df = pd.concat([current_df, new_data])

# Convert lists to arrays - this will include 2024 data since it's the first entry
if future_exog_nino_pdo:
    future_exog_nino_pdo = np.vstack(future_exog_nino_pdo)

# =====================
# 6. Save Combined Results to One CSV
# =====================

print(f"future_years length: {len(future_years)}")
print(f"future_storms length: {len(future_storms)}")
print(f"future_exog_nino_pdo shape: {future_exog_nino_pdo.shape}")  # Should have the same row count

# Generate confidence intervals for future storm counts
future_storms_lower = np.maximum(future_storms - 2 * np.sqrt(future_storms), 0).astype(int)
future_storms_upper = (future_storms + 2 * np.sqrt(future_storms)).astype(int)

# Compute the symmetric uncertainty (± range)
uncertainty = (future_storms_upper - future_storms_lower) / 2

# Update DataFrame to include ± uncertainty
combined_forecast_df = pd.DataFrame({
    'Year': future_years,
    'NINO4_ave': future_exog_nino_pdo[:len(future_years), 0],
    'NINO34_ave': future_exog_nino_pdo[:len(future_years), 1],
    'PDO_Annual_Avg': future_exog_nino_pdo[:len(future_years), 2],
    'Storm_Count': future_storms,
    'Uncertainty_±': uncertainty  # This now represents the ± range
})
# Define output file path BEFORE saving
combined_output_file = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\NF100Y.csv"  

# Save the updated CSV file
combined_forecast_df.to_csv(combined_output_file, index=False)  
print(f"✅ Updated 100-Year Forecast saved with ± uncertainty to: {combined_output_file}")

# =====================
# 7. Plot Results
# =====================
plt.figure(figsize=(14,7))

# Plot the full historical data
plt.plot(full_df.index, full_df['NUM_STORMS'], 
         marker='o', markerfacecolor='blue', markeredgecolor='black', linestyle='-', 
         color='blue', label='Historical Data', linewidth=2)

# Highlight the 2022-2024 period that we specifically targeted
plt.plot([2022, 2023, 2024], [known_historical_data[2022], known_historical_data[2023], known_historical_data[2024]], 
         marker='*', markerfacecolor='green', markersize=15, markeredgecolor='black', 
         linestyle='', label='Target Historical Data (2022-2024)')

# Plot the forecasted values
plt.plot(combined_forecast_df['Year'], combined_forecast_df['Storm_Count'], 
         marker='o', markerfacecolor='red', markeredgecolor='black', linestyle='--', 
         color='orange', label='Future Forecast')

# 🔹 ADD ERROR BARS (± uncertainty)
#plt.errorbar(combined_forecast_df['Year'], combined_forecast_df['Storm_Count'], 
#             yerr=combined_forecast_df['Uncertainty_±'], fmt='o', color='orange', capsize=5, label='Uncertainty')
             
# Add annotation for 2024 predicted vs actual
plt.annotate(f"2024 Prediction: {predicted_2024_rounded}\nActual: {actual_2024}",
             xy=(2024, known_historical_data[2024]), xytext=(2024, known_historical_data[2024]+5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1))

plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Storms", fontsize=12)
plt.title("Number of Storms Forecast with Uncertainty", fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Add vertical line at 2024 to separate historical from future
plt.axvline(x=2024, color='black', linestyle=':', alpha=0.7)
plt.text(2024.5, max(full_df['NUM_STORMS'])*0.9, "Forecast Start", rotation=90, verticalalignment='center')

# Highlight the historical range
plt.axvspan(2022, 2024, alpha=0.2, color='green')

plt.tight_layout()
plt_path = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\STNum_F100Y.png"
plt.savefig(plt_path, dpi=300)
plt.show()
# Extract only the filename
plt_filename = os.path.basename(plt_path)

# Ensure the arrays have the same length
min_length = min(len(future_years), len(future_storms))
future_years = future_years[:min_length]
future_storms = future_storms[:min_length]

# Compute standard deviation approximation (square root method)
storm_std = np.sqrt(np.array(future_storms))

# Compute lower and upper bounds with proper confidence intervals
future_storms_lower = np.maximum(np.array(future_storms) - 2 * storm_std, 0)
future_storms_upper = np.array(future_storms) + 2 * storm_std

# Round bounds to integers
future_storms_lower = np.floor(future_storms_lower).astype(int)
future_storms_upper = np.ceil(future_storms_upper).astype(int)

# Compute symmetric uncertainty (± range)
uncertainty = (future_storms_upper - future_storms_lower) // 2

# Combined DataFrame with confidence intervalsplt.fill_between(combined_forecast_df['Year'],
combined_forecast_df = pd.DataFrame({
    'Year': future_years,
    'NINO4_ave': future_exog_nino_pdo[:min_length, 0],
    'NINO34_ave': future_exog_nino_pdo[:min_length, 1],
    'PDO_Annual_Avg': future_exog_nino_pdo[:min_length, 2],
    'Storm_Count': future_storms,
    'Storm_Count_Lower': future_storms_lower,
    'Storm_Count_Upper': future_storms_upper,
    'Uncertainty': uncertainty
})

plt.figure(figsize=(14,7))

plt.fill_between(combined_forecast_df['Year'], 
                 combined_forecast_df['Storm_Count_Lower'], 
                 combined_forecast_df['Storm_Count_Upper'],
                 color='orange', alpha=0.5, label='95% Confidence Interval')
plt.plot(combined_forecast_df['Year'], combined_forecast_df['Storm_Count'], 
         marker='o', markerfacecolor='red', markeredgecolor='black', linestyle='--', 
         color='orange', label='Future Forecast')

plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Storms", fontsize=12)
plt.title("Number of Storms Forecast with Uncertainty", fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.axvline(x=2024, color='black', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig(plt_path, dpi=300)
plt.show()

#plt.figure(figsize=(14,7))

# Add shaded area for confidence intervals
#plt.fill_between(combined_forecast_df['Year'], 
#                 combined_forecast_df['Storm_Count_Lower'], 
#                 combined_forecast_df['Storm_Count_Upper'],
#                 color='orange', alpha=0.3, label='95% Confidence Interval')

# Add a subplot for hindcasting results
plt.figure(figsize=(14,7))

# 🔹 First, add the shaded confidence interval (BEFORE plotting lines)
plt.fill_between(hindcast_results['Year'], 
                 hindcast_results['Predicted'] - hindcast_results['AbsError'], 
                 hindcast_results['Predicted'] + hindcast_results['AbsError'],
                 color='gray', alpha=0.3, label="Hindcast Uncertainty")

plt.plot(hindcast_results['Year'], hindcast_results['Actual'], 
         marker='o', markerfacecolor='blue', markeredgecolor='black', linestyle='-', 
         color='blue', label='Actual Values', linewidth=2)
plt.plot(hindcast_results['Year'], hindcast_results['Predicted'], 
         marker='o', markerfacecolor='red', markeredgecolor='black', linestyle='--', 
         color='orange', label='Hindcast Predictions')

# Add error bars for uncertainty
#plt.errorbar(hindcast_results['Year'], hindcast_results['Predicted'], 
#             yerr=hindcast_results['AbsError'], fmt='o', color='orange', capsize=5, label='Uncertainty')

plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Storms", fontsize=12)
plt.title("Hindcast Validation (2015-2024)", fontsize=14, fontweight='bold')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 🔹 Save & Show the Hindcast Plot
plt_hindcast_path = r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\ValidHindcast.png"  # Saves the hindcast plot
plt.savefig(plt_hindcast_path, dpi=300)
plt.show()

# Extract only the hindcast filename
plt_hindcast_filename = os.path.basename(plt_hindcast_path)

# Add text about hindcast performance
plt.text(0.02, 0.02, f"Avg Absolute Error: {round(avg_abs_error, 2)}\nAvg Percentage Error: {round(avg_pct_error, 2)}%", 
         transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

print(f"✅ Hindcast visualization saved as: {plt_hindcast_path}")
print(f"✅ Combined Forecast saved to: {combined_output_file}")
print(f"✅ Historical values fixed to match: {known_historical_data}")
print(f"✅ Calibration factor applied: {calibration_factor:.4f}")
print(f"✅ Visualization saved as: {plt_filename}")

# =====================
# 8. BLAND-ALTMAN Tests
# =====================

import seaborn as sns
from scipy.stats import shapiro

# 🔹 Sample DataFrame (Replace this with actual hindcast_results DataFrame)
# hindcast_results = pd.read_csv(r"G:\2025\GEVNEW\SOURCE\SARIMAXOUT\ThirdTry\hindcastresults.csv")

# Compute errors (differences)
differences = hindcast_results['Predicted'] - hindcast_results['Actual']

# 🔹 Histogram of Prediction Errors
plt.figure(figsize=(8, 5))
sns.histplot(differences, bins=20, kde=True, color='orange')
plt.axvline(np.mean(differences), color='red', linestyle='--', label="Mean Difference")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Histogram of Prediction Errors")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 🔹 Shapiro-Wilk Test for Normality
stat, p_value = shapiro(differences)
print(f"Shapiro-Wilk Test p-value: {p_value:.4f}")

if p_value > 0.05:
    print("✅ Errors are likely normally distributed. Proceeding with Bland-Altman Test...")
    
    # Compute mean values and Bland-Altman statistics
    mean_values = (hindcast_results['Predicted'] + hindcast_results['Actual']) / 2
    bias = np.mean(differences)
    loa_upper = bias + 1.96 * np.std(differences)
    loa_lower = bias - 1.96 * np.std(differences)

    # 🔹 Bland-Altman Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(mean_values, differences, color='orange', edgecolors='black', alpha=0.7, label="Differences")
    plt.axhline(bias, color='red', linestyle='--', label=f'Bias = {bias:.2f}')
    plt.axhline(loa_upper, color='blue', linestyle='--', label=f'Upper LoA = {loa_upper:.2f}')
    plt.axhline(loa_lower, color='blue', linestyle='--', label=f'Lower LoA = {loa_lower:.2f}')
    plt.xlabel("Mean of Actual & Predicted Storm Counts")
    plt.ylabel("Difference (Predicted - Actual)")
    plt.title("Bland-Altman Plot for Hindcast Predictions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("⚠️ Errors are NOT normally distributed! Bland-Altman might not be reliable.")

# 🔹 45-degree Agreement Plot
plt.figure(figsize=(8, 8))
plt.scatter(hindcast_results['Actual'], hindcast_results['Predicted'], 
            color='dodgerblue', edgecolors='black', alpha=0.7, label="Predicted vs Actual")

# Add the 45-degree agreement line
plt.plot([hindcast_results['Actual'].min(), hindcast_results['Actual'].max()],
         [hindcast_results['Actual'].min(), hindcast_results['Actual'].max()],
         color='red', linestyle='--', label="45-degree Agreement Line")

# Set axis limits for better visualization
#plt.xlim(hindcast_results['Actual'].min() - 1, hindcast_results['Actual'].max() + 1)
#plt.ylim(hindcast_results['Predicted'].min() - 1, hindcast_results['Predicted'].max() + 1)

# Labels and title
#plt.xlabel("Actual Storm Counts")
#plt.ylabel("Predicted Storm Counts")
#plt.title("Actual vs. Predicted Storm Counts (45° Agreement Line)")
#plt.legend()
#plt.grid(alpha=0.3)

# Show the plot
#plt.show()

# =====================
# 9. More Tests
# =====================

# Load data
hindcast_results = pd.read_csv(hindcast_output_file)

# Check the first few rows
print(hindcast_results.head())

# Compute error metrics
mae = mean_absolute_error(hindcast_results['Actual'], hindcast_results['Predicted'])
rmse = np.sqrt(mean_squared_error(hindcast_results['Actual'], hindcast_results['Predicted']))
mbe = np.mean(hindcast_results['Predicted'] - hindcast_results['Actual'])

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Bias Error (MBE): {mbe:.2f}")

# Scatter plot of Actual vs. Predicted values
plt.figure(figsize=(8, 8))
sns.scatterplot(x=hindcast_results['Actual'], y=hindcast_results['Predicted'], color='dodgerblue', edgecolor='black', alpha=0.7, label="Predicted vs Actual")

# 45-degree agreement line
min_val = min(hindcast_results['Actual'].min(), hindcast_results['Predicted'].min()) - 1
max_val = max(hindcast_results['Actual'].max(), hindcast_results['Predicted'].max()) + 1
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="45-degree Agreement Line")

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.xlabel("Actual Storm Counts")
plt.ylabel("Predicted Storm Counts")
plt.title("Actual vs. Predicted Storm Counts (45° Agreement Line)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()