import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reference point (12.375N, 121.5E)
reference_lat = 12.375
reference_lon = 121.5

# Function to calculate distance using Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius = 6371  # Earth radius in kilometers
    return radius * c

# Function to get the highest wind per season, and if multiple, choose the closest
def get_highest_closest(group):
    max_wind = group['TOK_WIND'].max()
    max_wind_group = group[group['TOK_WIND'] == max_wind].copy()
    max_wind_group['distance'] = max_wind_group.apply(
        lambda row: haversine(row['LAT'], row['LON'], reference_lat, reference_lon), axis=1)
    return max_wind_group.loc[max_wind_group['distance'].idxmin()]

# Load and filter data
file_path = r"D:\2026\SYNTC\SYNTC-main\data\ibtracs_1977_2025.csv"
columns_needed = ['SEASON', 'NAME', 'ISO_TIME', 'LAT', 'LON', 'TOK_WIND', 'TOK_PRES']

#data = pd.read_excel(file_path, usecols=columns_needed).dropna(subset=['LAT', 'LON', 'TOK_WIND', 'TOK_PRES'])
data = pd.read_csv(file_path, usecols=columns_needed).dropna(subset=['LAT', 'LON', 'TOK_WIND', 'TOK_PRES'])
data[['LAT', 'LON']] = data[['LAT', 'LON']].apply(pd.to_numeric, errors='coerce')

# Get the highest wind per season, with the closest to the reference point if multiple
highest_wind_per_season = data.groupby('SEASON').apply(get_highest_closest).reset_index(drop=True).sort_values('SEASON')

# Save to Excel
output_path = r"D:\2026\SYNTC\SYNTC-main\data\ForGEVComputation.csv"
try:
    #highest_wind_per_season.to_excel(output_path, index=False)
    highest_wind_per_season.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
except Exception as e:
    print(f"Error saving Excel file: {e}")

# Create plot with proper borders
plt.figure(figsize=(14, 7))
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor('lightgray')
    spine.set_linewidth(0.8)

# Plot data with connected markers
x = highest_wind_per_season['SEASON'].astype(float)
y = highest_wind_per_season['TOK_WIND'].astype(float)
ax.plot(x, y, color='#1f77b4', marker='o', linestyle='-', linewidth=2, markersize=8, label='Maximum Wind Speed')

# Trend line calculation
coefficients = np.polyfit(x, y, 1)
trend = np.poly1d(coefficients)
r_squared = np.corrcoef(x, y)[0, 1] ** 2
ax.plot(x, trend(x), '--', color='#ff7f0e', label=f'Trend Line (y = {coefficients[0]:.4f}x + {coefficients[1]:.2f})')

# Labels and titles
ax.set_title('Maximum Wind Speed by Season', fontsize=14, pad=18, weight='semibold')
ax.set_xlabel('Season', fontsize=12, labelpad=10)
ax.set_ylabel('Max Wind (10-min knots)', fontsize=12, labelpad=10)
ax.set_xticks(range(1975, 2026, 10))
ax.set_xticklabels([str(year) for year in range(1975, 2026, 10)], rotation=0, ha='center')
ax.set_xlim(1975, 2025)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', frameon=True, framealpha=0.9)

# Save plot output
plot_path = r"D:\2026\SYNTC\SYNTC-main\data\plots\max_wind_trends.png"
try:
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
except Exception as e:
    print(f"Error saving plot: {e}")
plt.close()