# Has ERC function. 
# Most stable_version_20250521.
# Same as Y2_SYNTC.py from group's Google Drive's file.
# Has a land decay function that uses a GEBCO DTM (DEM).
# Storm Generator for Philippine Area of Responsibility (PAR) only.
# Final Version: SYNTC_EX.py --> SYNTC_wDTM_ENSEMBLE_final.py --> SYNTCZerr.
# Combines the synthetic storm generator with STORM methodology by Bloemendaal et al. (2020).
# Arranged, compiled, designed, directed, and edited by Jeferson B. Zerrudo (WS-1, DOST-PAGASA) with the help of Claude AI.

import os
import sys
import math
import random
import logging
import rasterio
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from typing import Union, List, Optional
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter1d
from shapely.geometry import Point, Polygon
from tc_track_statistics import DataDrivenTrackModel
from scipy.stats import weibull_min, genextreme, norm, gamma

# Force UTF-8 encoding for console output
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ——— Configure UTF-8 console handler ———
handler = logging.StreamHandler(sys.stdout)
# On Python >=3.9 you can also do:
# handler = logging.StreamHandler(sys.stdout, encoding="utf-8")
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

root = logging.getLogger()
root.handlers.clear()
root.addHandler(handler)
root.setLevel(logging.INFO)
# ————————————————————————————————
def ensure_scalar(value, default=22.0):
    """
    ENHANCED: Ensure a value is a scalar, not an array.
    This is the silver bullet for array ambiguity errors.
    """
    if value is None:
        return default
    
    # Handle pandas Series
    if hasattr(value, 'iloc'):
        if len(value) > 0:
            return float(value.iloc[0])
        else:
            return default
    
    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        elif value.size == 1:
            return float(value.item())
        else:
            # For multi-element arrays, take the first element
            return float(value.flat[0])
    
    # Handle lists and tuples
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        else:
            return float(value[0])
    
    # Handle single values
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_comparison(value1, operator, value2):
    """
    Safely compare values that might be arrays.
    Returns a boolean result.
    """
    # Ensure both values are scalars
    v1 = ensure_scalar(value1)
    v2 = ensure_scalar(value2)
    
    # Perform comparison
    if operator == '<':
        return v1 < v2
    elif operator == '<=':
        return v1 <= v2
    elif operator == '>':
        return v1 > v2
    elif operator == '>=':
        return v1 >= v2
    elif operator == '==':
        return v1 == v2
    elif operator == '!=':
        return v1 != v2
    else:
        raise ValueError(f"Unknown operator: {operator}")

        
class ErrorMonitor:
    """Monitor for critical errors and stop simulation"""
    
    def __init__(self, stop_on_error=True):
        self.stop_on_error = stop_on_error
        self.error_count = 0
        self.critical_keywords = [
            "CRITICAL ERROR",
            "Extreme wind values detected",
            "Length mismatch",
            "Array ERROR detected",
            "Failed to load",
            "ambiguous",
            "NoneType",
            "Emergency generation error",
            "has no attribute",
            "truth value of an array"
        ]
    
    def check_error(self, message):
        """Check if message contains critical error"""
        if any(keyword in message for keyword in self.critical_keywords):
            self.error_count += 1
            if self.stop_on_error:
                print(f"\n{'='*80}")
                print("CRITICAL ERROR DETECTED - STOPPING SIMULATION")
                print(f"Error message: {message}")
                print(f"Total errors: {self.error_count}")
                print(f"{'='*80}")
                import sys
                sys.exit(1)

# Global error monitor
error_monitor = ErrorMonitor(stop_on_error=True)

class ErrorStoppingHandler(logging.Handler):
    """Custom logging handler that stops on errors"""
    
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            error_monitor.check_error(record.getMessage())
            
# Define BASE_DIR first, with executable detection
def get_application_path():
    """Get the base path where the application is running from"""
    if getattr(sys, 'frozen', False):
        # Running as executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))
        
BASE_DIR = get_application_path()

# Define log_dir and other paths after BASE_DIR is defined
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(log_dir, f"par_storm_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ),
        logging.StreamHandler(),
        ErrorStoppingHandler() 
    ]
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
DATA_DIR = os.path.join(BASE_DIR, "data")
FORECAST_PATH = os.path.join(DATA_DIR, "F100Y.csv")
WIND_PATH = os.path.join(DATA_DIR, "ALL_TCWS_mod-7.csv")
POSITIONS_PATH = os.path.join(DATA_DIR, "big_perimeter_1977_2023.csv")         
DEM_PATH = os.path.join(BASE_DIR, "data", "DEM", "dtm_phil_1km.tif")
DEM_FALLBACK_PATH = os.path.join(BASE_DIR, "data", "DEM", "phl_dem.tif")  # fallback DEM
SHAPEFILE_PATH = os.path.join(BASE_DIR, "data", "SHP", "PHL_ADM0.shp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
CSV_DIR = os.path.join(BASE_DIR, "csv") 

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# Define a larger boundary for genesis points
GENESIS_BOUNDS = [
    (25.00, 141.461026),  # 25.0°N, 141.461026°E
    (5.00, 141.461026),   # 5.0°N, 141.461026°E
    (5.00, 114.115689),   # 5.0°N, 114.115689°E
    (25.00, 114.115689),  # 25.0°N, 114.115689°E
]

# Opt for PAR boundaries as GENESIS_BOUNDS
#GENESIS_BOUNDS = [
#    (25, 135),      # 25°N, 135°E
#    (5, 135),       # 5°N, 135°E
#    (5, 115),       # 5°N, 115°E
#    (15, 115),      # 15°N, 115°E
#    (21, 120),      # 21°N, 120°E
#    (25, 120),      # 25°N, 120°E
#]

# This is the actual PAR coordinates for final filtering
PAR_BOUNDS = [
    (25, 135),      # 25°N, 135°E
    (5, 135),       # 5°N, 135°E
    (5, 115),       # 5°N, 115°E
    (15, 115),      # 15°N, 115°E
    (21, 120),      # 21°N, 120°E
    (25, 120),      # 25°N, 120°E
]

# Create PAR polygon
PAR_POLYGON = Polygon([(lon, lat) for lat, lon in PAR_BOUNDS])
# Create genesis area polygon
GENESIS_POLYGON = Polygon([(lon, lat) for lat, lon in GENESIS_BOUNDS])
# Create a buffered version of the genesis polygon for track generation
BUFFERED_PAR_POLYGON = GENESIS_POLYGON.buffer(1.0)

# Load Philippine land boundaries
try:
    PHIL_LAND = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
    logging.info("Successfully loaded Philippine land boundaries")
except Exception as e:
    logging.error(f"Failed to load Philippine land boundaries: {e}")
    # Fallback to the simplified land check if shapefile loading fails
    PHIL_LAND = None

# PAR Calculated Future Trends for Monthly TC Categories Distribution
_monthly_category_dist = {
    1: [45.8, 34.4, 13.6, 6.2, 0.0],    # January: Minor reduction in TY
    2: [57.4, 33.9, 5.8, 2.9, 0.0],     # February: Minor adjustments
    3: [38.1, 25.4, 17.0, 16.1, 3.4],   # March: Significant increase in TS, decreased TD
    4: [25.5, 25.9, 18.5, 26.6, 3.5],   # April: Minor reduction in TY
    5: [31.6, 26.5, 14.9, 23.5, 3.5],   # May: Minor reduction in TY
    6: [38.6, 24.2, 17.4, 18.7, 1.1],   # June: Decreased TY based on negative trend
    7: [29.4, 23.3, 17.6, 25.9, 3.8],   # July: Increased TS based on strong positive trend
    8: [31.4, 22.2, 16.5, 23.8, 6.1],   # August: Decreased TY, increased STY
    9: [28.2, 19.0, 16.8, 27.4, 8.6],   # September: Minor redistribution
    10: [22.3, 19.3, 21.6, 28.7, 8.1],  # October: Significant decrease in TY, redistribution
    11: [33.4, 23.3, 16.0, 23.6, 3.7],  # November: Significant decrease in STY
    12: [22.7, 26.3, 22.6, 25.7, 2.7]   # December: Minor adjustment to TY
}

# Future Trend Calculated from WP (all data set) for Monthly TC Categories Distribution
#_monthly_category_dist = {
#    1: [45.8, 35.4, 12.6, 6.2, 0.0]   # Minor adjustments
#    2: [58.4, 33.9, 4.8, 2.9, 0.0]     # Minor adjustments
#    3: [39.1, 24.4, 17.0, 16.1, 3.4]   # Increased TS, decreased TY
#    4: [25.5, 25.9, 19.5, 25.6, 3.5]   # Minor adjustments
#    5: [31.6, 26.5, 15.9, 22.5, 3.5]   # Minor decrease in TY
#    6: [38.6, 25.2, 17.4, 17.7, 1.1]   # Decreased TY
#    7: [32.4, 21.3, 19.6, 23.9, 2.8]   # Increased TS, decreased TY
#    8: [31.4, 23.2, 17.5, 23.8, 4.1]   # Decreased TY 
#    9: [28.2, 20.0, 18.8, 25.4, 7.6]   # Decreased TY
#    10: [22.3, 21.3, 22.6, 28.7, 5.1]   # Significantly decreased TY and STY
#    11: [32.4, 24.3, 17.0, 21.6, 4.7]   # Decreased STY and TY
#    12: [22.7, 27.3, 23.6, 24.7, 1.7]   # Modest decrease in TY

# Historical (1977-2023) Monthly TC Categories Distribution
#_monthly_category_dist = {
#    1: [44.8, 34.4, 13.6, 7.1, 0.0],    # January
#    2: [57.4, 32.9, 5.8, 3.9, 0.0],    # February
#    3: [41.1, 20.4, 17.0, 18.1, 3.4],    # March
#    4: [25.5, 23.9, 18.5, 28.6, 3.4],    # April
#    5: [30.6, 25.5, 14.9, 25.5, 3.5],    # May
#    6: [37.6, 23.2, 17.4, 20.7, 1.2],    # June
#    7: [32.4, 18.3, 17.6, 27.9, 3.7],    # July
#    8: [30.4, 21.2, 15.5, 27.8, 5.1],    # August
#    9: [28.2, 18.0, 15.8, 29.4, 8.6],    # September
#    10: [19.3, 17.3, 19.6, 35.7, 8.2],    # October
#    11: [31.4, 22.3, 15.0, 23.6, 7.8],    # November
#    12: [21.7, 25.3, 22.6, 27.7, 2.7],    # December
#}

# Cache for optimization
_historical_wind_speeds = None
_df_positions = None
_df_wind = None
_df_forecast = None
_weibull_params = None
_gev_params = None
_month_params_cache = None
_dem_data = None
_dem_transform = None
track_model = None  # Will be initialized in main()

#_rmax_pressure = {
#    0: np.array([25, 30, 35, 40, 45, 50]),     # P <= 920
#    1: np.array([30, 35, 40, 45, 50, 55, 60]), # 920 < P <= 940
#    2: np.array([35, 40, 45, 50, 55, 60, 65, 70, 75, 80])  # P > 940
#}

# Using empirical equation based on wind speed and latitude from Knaff et al. (2015) 
# The empirical equation used:
# rmw = 218.3784 - 1.2014 * wind_speed + ((wind_speed) / 10.9844) ** 2 - 
#       ((wind_speed) / 35.3052) ** 3 - 145.5090 * np.cos(np.radians(lat))

_rmax_pressure = {
    0: np.array([18, 27, 30, 32, 34, 35, 37, 39, 43, 97]),     # P <= 920
    1: np.array([25, 34, 36, 38, 41, 43, 45, 47, 50, 59]), # 920 < P <= 940
    2: np.array([27, 50, 57, 65, 72, 79, 87, 92, 98, 117])  # P > 940
}

ENABLE_STY_THRESHOLD_BOOST = False  # Toggle (FALSE/TRUE) this to control the feature found in the "def generate_synthetic_storm()" function 
    
###############################################################################
# STORM MODULE INTEGRATIONS (FROM BLOEMENDAAL ET AL.)
###############################################################################
class StormIntensityManager:
    """
    Centralized management of storm wind speeds to prevent competing processes.
    """
    def __init__(self, initial_winds, storm_id):
        self.winds = np.array(initial_winds, dtype=np.float32)
        self.storm_id = storm_id
        self.modification_log = []
        self.category_locked = False
        self.max_intensity_reached = 0
        
    def apply_modification(self, modification_func, *args, **kwargs):
        """Apply a wind modification and log it."""
        old_winds = self.winds.copy()
        
        # Apply the modification
        new_winds = modification_func(self.winds, *args, **kwargs)
        
        # Validate
        new_winds = validate_wind_speeds(new_winds, caller=f"IntensityManager-{modification_func.__name__}")
        
        # Check for category persistence rules
        new_winds = self._apply_category_persistence(old_winds, new_winds)
        
        # Update tracking
        self.winds = new_winds
        self.max_intensity_reached = max(self.max_intensity_reached, np.max(new_winds))
        
        # Log the change
        self.modification_log.append({
            'function': modification_func.__name__,
            'old_max': np.max(old_winds),
            'new_max': np.max(new_winds),
            'change': np.max(new_winds) - np.max(old_winds)
        })
        
        return self.winds
    
    def _apply_category_persistence(self, old_winds, new_winds):
    #    """Ensure realistic category persistence."""
    #    # If storm reached STY status, prevent rapid weakening below TY
    #    if self.max_intensity_reached >= 100:
    #        # Don't allow sudden drops below 85kt without land interaction
    #        min_allowed = 85.0
    #        rapid_drops = new_winds < min_allowed
    #        if np.any(rapid_drops) and not self.category_locked:
    #            # Gradual transition instead of sudden drop
    #            new_winds[rapid_drops] = np.maximum(new_winds[rapid_drops], min_allowed)
        
        # Prevent rapid oscillation around category boundaries
    #    for i in range(1, len(new_winds)-1):
            # STY boundary smoothing
    #        if old_winds[i] >= 100 and new_winds[i] < 100 and new_winds[i] > 95:
    #            if old_winds[i-1] >= 100 or (i+1 < len(old_winds) and old_winds[i+1] >= 100):
    #                new_winds[i] = 100.5  # Keep as STY
            
            # TY boundary smoothing  
    #        elif old_winds[i] >= 64 and new_winds[i] < 64 and new_winds[i] > 60:
    #            if old_winds[i-1] >= 64 or (i+1 < len(old_winds) and old_winds[i+1] >= 64):
    #                new_winds[i] = 64.5  # Keep as TY
        
        return new_winds
    
    def lock_category_for_land(self):
        """Lock category changes for land interaction."""
        self.category_locked = True
    
    def get_final_winds(self):
        """Get the final processed wind speeds."""
        return self.winds
    
class StormDataManager:
    """Context manager for storm data with automatic cleanup"""
    
    def __init__(self):
        self.dataframes = []
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def add_dataframe(self, df):
        self.dataframes.append(df)
        return df
        
    def cleanup(self):
        for df in self.dataframes:
            if hasattr(df, 'memory_usage'):
                del df
        self.dataframes.clear()
        gc.collect()
    
class StormConfig:
    """Configuration class to replace global variables"""
    
    def __init__(self):
        self.dem_data = None
        self.dem_transform = None
        self.track_model = None
        self.historical_wind_speeds = None
        self.df_positions = None
        self.df_wind = None
        self.df_forecast = None
        self.month_params_cache = None
        self._initialized = False
    
    def initialize(self, data_dir):
        """Initialize all required data"""
        if self._initialized:
            return
            
        # Load DEM
        dem_path = os.path.join(data_dir, "DEM", "dtm_phil_1km.tif")
        self.load_dem(dem_path)
        
        # Initialize track model
        positions_path = os.path.join(data_dir, "big_perimeter_1977_2023.csv")
        if os.path.exists(positions_path):
            from tc_track_statistics import DataDrivenTrackModel
            self.track_model = DataDrivenTrackModel(positions_path)
        
        self._initialized = True
    
    def load_dem(self, dem_path):
        """Load DEM data safely"""
        try:
            import rasterio
            with rasterio.open(dem_path) as src:
                self.dem_data = src.read(1)
                self.dem_transform = src.transform
        except Exception as e:
            logging.error(f"Failed to load DEM: {e}")
            self.create_fallback_dem()
    
    def create_fallback_dem(self):
        """Create minimal fallback DEM"""
        # Simplified fallback implementation
        self.dem_data = np.zeros((100, 100))
        import rasterio.transform
        self.dem_transform = rasterio.transform.from_bounds(115, 4, 135, 22, 100, 100)
        
def gradual_land_cap(lats, lons, winds):
    """Apply gradual wind capping - allows extreme winds to briefly touch land"""
    modified_winds = winds.copy()
    
    for i in range(len(winds)):
        if is_over_land(lons[i], lats[i]):
            current_wind = winds[i]
            
            # Count recent land points (last 3 timesteps)
            recent_land_count = 0
            for j in range(max(0, i-2), i+1):
                if j < len(winds) and is_over_land(lons[j], lats[j]):
                    recent_land_count += 1
            
            # Progressive capping based on land exposure
            current_wind_scalar = ensure_scalar(current_wind)
            if current_wind_scalar >= 106:  # Extreme winds
                if recent_land_count == 1:
                    new_wind = current_wind  # First contact - no reduction
                elif recent_land_count == 2:
                    new_wind = max(105, current_wind * 0.95)  # 5% reduction, min 100kt
                else:
                    new_wind = max(95, current_wind * 0.85)  # Stronger reduction after 6+ hours
                
                # ADD DEBUG LOGGING HERE:
                current_wind_scalar = ensure_scalar(current_wind)
                if current_wind_scalar >= 106 and new_wind < 106:
                    logging.warning(f"EXTREME WIND CONTACT: {current_wind:.1f}kt -> {new_wind:.1f}kt (land_count={recent_land_count})")
                    
                modified_winds[i] = new_wind
                
            elif current_wind > 100.0:  # Strong winds
                if recent_land_count == 1:
                    new_wind = max(100, current_wind * 0.97)  # 3% reduction
                else:
                    new_wind = 90  # Cap at 90kt
                modified_winds[i] = new_wind
                
            elif current_wind > 70.0:  # Moderate winds - gentler treatment
                reduction = 0.95 if recent_land_count == 1 else 0.90
                modified_winds[i] = min(current_wind, 70.0 + (current_wind - 70.0) * reduction)
    
    return modified_winds
    
def apply_gentle_erc(winds):
    """Apply a gentler ERC that doesn't cause segmentation."""
    if len(winds) < 8:
        return winds
    
    # Find peak intensity period
    peak_idx = np.argmax(winds)
    
    # Apply gentle temporary reduction (5-10%) for 2-3 points
    erc_winds = winds.copy()
    erc_start = max(0, peak_idx - 1)
    erc_end = min(len(winds), peak_idx + 2)
    
    # Gentle reduction
    reduction_factor = 0.95  # Only 5% reduction
    erc_winds[erc_start:erc_end] *= reduction_factor
    
    return erc_winds

def smooth_storm_intensity(storm_df, window_size=3):
    """
    Apply stronger smoothing to eliminate artificial segmentation,
    with special handling for extreme storms.
    """
    if len(storm_df) < window_size:
        return storm_df
    
    # Create a copy
    smoothed_df = storm_df.copy()
    
    # Check if this is an extreme storm
    max_intensity = smoothed_df['WIND'].max()
    is_extreme = max_intensity >= 106
    
    if is_extreme:
        # Lighter smoothing for extreme storms to preserve intensity peaks
        actual_window = max(2, window_size - 2)  # Smaller window
        
        # Only smooth non-peak regions
        peak_indices = smoothed_df['WIND'] >= 100
        
        # Apply rolling mean only to non-peak areas
        non_peak_winds = smoothed_df.loc[~peak_indices, 'WIND']
        if len(non_peak_winds) > 1:
            smoothed_non_peak = non_peak_winds.rolling(
                window=actual_window, center=True, min_periods=1
            ).mean()
            smoothed_df.loc[~peak_indices, 'WIND'] = smoothed_non_peak
            
        logging.info(f"EXTREME STORM: Applied reduced smoothing to preserve {peak_indices.sum()} peak points")
    else:
        # Normal smoothing for regular storms
        actual_window = window_size
        smoothed_df['WIND'] = smoothed_df['WIND'].rolling(
            window=actual_window, center=True, min_periods=1
        ).mean()
    
    # Prevent category flickering around boundaries
    # If a storm reaches STY status, keep it there for at least 3 points
    winds = smoothed_df['WIND'].values
    for i in range(1, len(winds)-1):
        if winds[i-1] >= 100 and winds[i+1] >= 100 and winds[i] < 100:
            if winds[i] > 95:  # Close to threshold
                winds[i] = 100.5  # Keep it as STY
    
    smoothed_df['WIND'] = winds
    
    # Recalculate categories
    categories = []
    for wind in smoothed_df['WIND']:
        if wind >= 100:
            categories.append("Super Typhoon")
        elif wind >= 64:
            categories.append("Typhoon")
        elif wind >= 48:
            categories.append("Severe Tropical Storm")
        elif wind >= 34:
            categories.append("Tropical Storm")
        elif wind >= 22:
            categories.append("Tropical Depression")
        else:
            categories.append("Remnant Low")
    
    smoothed_df['CATEGORY'] = categories
    
    return smoothed_df

def _apply_profile_adjustment(winds, target_max_wind, storm_category):
    """Modified version of adjust_wind_profile for use with IntensityManager."""
    return adjust_wind_profile(winds, target_max_wind, storm_category)

def _apply_land_decay(winds, storm_df, dem_path):
    """Modified version for use with IntensityManager."""
    # Create temporary df with new winds
    temp_df = storm_df.copy()
    temp_df['WIND'] = winds
    
    # Apply decay
    decayed_df = apply_dem_based_decay(temp_df, dem_path)
    
    return decayed_df['WIND'].values

def validate_wind_speeds(
    wind_values: Union[np.ndarray, float, List[float]], 
    min_val: float = 0.0, 
    max_val: float = 126.0, 
    caller: str = "unknown",
    convert_to_int: bool = False
) -> np.ndarray:
    """
    Validate and correct wind speeds to ensure they remain within physical limits.
    
    Args:
        wind_values: Wind speed values to validate
        min_val: Minimum allowed wind speed
        max_val: Maximum allowed wind speed
        caller: Name of calling function for logging
        convert_to_int: Whether to convert to integers for final output
        
    Returns:
        Validated wind speeds as numpy array
        
    Raises:
        ValueError: If input data is invalid
    """
    try:
        # Convert to numpy array if not already
        if not isinstance(wind_values, np.ndarray):
            wind_array = np.array(wind_values, dtype=np.float32)
        else:
            wind_array = wind_values.astype(np.float32)
            
        # Check for extreme values (1000+ knots)
        if np.any(wind_array > 1000.0):
            extreme_count = np.sum(wind_array > 1000.0)
            extreme_max = np.max(wind_array)
            logging.error(f"CRITICAL ERROR: Extreme wind values detected in {caller}: "
                         f"{extreme_count} values > 1000 knots, max={extreme_max}")
        
        # Check for NaN values
        if np.any(np.isnan(wind_array)):
            nan_count = np.sum(np.isnan(wind_array))
            logging.error(f"ERROR: {nan_count} NaN wind values detected in {caller}")
            # Replace NaNs with minimum value
            wind_array = np.nan_to_num(wind_array, nan=min_val)
        
        # Check for normal violations of limits
        if np.any(wind_array < min_val) or np.any(wind_array > max_val):
            low_count = np.sum(wind_array < min_val)
            high_count = np.sum(wind_array > max_val)
            if low_count > 0 or high_count > 0:
                logging.warning(f"Wind validation in {caller}: {low_count} values < {min_val}, "
                              f"{high_count} values > {max_val}. Clipping to [{min_val}, {max_val}]")
        
        # Clip values to valid range
        wind_array = np.clip(wind_array, min_val, max_val)
        
        # Only convert to integers if specifically requested
        if convert_to_int:
            return np.round(wind_array).astype(np.int32)
        else:
            return wind_array
        
    except Exception as e:
        logging.error(f"Error in wind validation: {e}, caller: {caller}")
        # Return safe default if processing fails
        if isinstance(wind_values, (list, np.ndarray)):
            return np.full_like(np.array(wind_values, dtype=np.float32), min_val)
        else:
            return np.array([min_val], dtype=np.float32) 
            
def is_inside_par(lat, lon, use_buffer=False):
    """
    Check if a given latitude and longitude point is inside the PAR boundary.
    
    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        use_buffer (bool): Whether to use the buffered (extended) PAR boundary.
    
    Returns:
        bool: True if the point is inside the PAR boundary, False otherwise.
    """
    # Validate inputs
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or math.isnan(lat) or math.isnan(lon):
        return False

    # Create a Point object
    point = Point(float(lon), float(lat))

    # Check if the point is inside the appropriate polygon
    if use_buffer:
        return BUFFERED_PAR_POLYGON.contains(point)
    else:
        return PAR_POLYGON.contains(point)

def get_elevation(lon, lat):
    """
    Get elevation at a specific point from the DEM.
    Returns elevation in meters, or 0.0 if outside DEM bounds or in case of error.
    """
    global _dem_data, _dem_transform
    
    if _dem_data is None or _dem_transform is None:
        logging.warning(f"ELEV DEBUG: DEM not available for ({lon}, {lat})")
        return 0.0  # No DEM data available
    
    try:
        # Convert geographic coordinates to pixel indices
        row, col = rasterio.transform.rowcol(_dem_transform, lon, lat)
        
        # DEBUGGING:
        logging.info(f"ELEV DEBUG: ({lat:.2f}N, {lon:.2f}E) -> row={row}, col={col}, bounds=({_dem_data.shape})")
        
        # Check if indices are within bounds
        if 0 <= row < _dem_data.shape[0] and 0 <= col < _dem_data.shape[1]:
            elevation = _dem_data[row, col]
            
            # DEBUGGING:
            logging.info(f"ELEV DEBUG: Raw elevation = {elevation}")
            
            # Handle nodata values (often represented as very negative values like -9999)
            if elevation < -1000 or np.isnan(elevation):
                logging.info(f"ELEV DEBUG: NoData value {elevation}, returning 0.0")
                return 0.0
            
            return float(elevation)
        else:
            logging.info(f"ELEV DEBUG: Outside bounds, returning 0.0")
            return 0.0  # Outside DEM bounds
    except Exception as e:
        logging.warning(f"ELEV DEBUG: Error sampling DEM at ({lon}, {lat}): {e}")
        return 0.0  # Return 0 in case of error

def is_over_land(lon, lat, buffer_km=30):
    """
    Enhanced check if a point is over land using DEM with buffer zone.
    """
    # APPROACH 1: Check using DEM elevation (primary method)
    global _dem_data, _dem_transform
    
    # First check exact point
    is_land = False
    
    if _dem_data is not None and _dem_transform is not None:
        try:
            row, col = rasterio.transform.rowcol(_dem_transform, lon, lat)
            if 0 <= row < _dem_data.shape[0] and 0 <= col < _dem_data.shape[1]:
                elevation = _dem_data[row, col]
                if not np.isnan(elevation) and elevation > 0.5:
                    is_land = True
        except Exception:
            pass
    
    # If not land, check points in a buffer zone around the location
    if not is_land and buffer_km > 0:
        # Calculate buffer in degrees (approximate)
        buffer_deg = buffer_km / 111.0  # ~111km per degree
        
        # Check points in a grid around the target
        for dlat in [-buffer_deg, 0, buffer_deg]:
            for dlon in [-buffer_deg, 0, buffer_deg]:
                if dlat == 0 and dlon == 0:
                    continue  # Skip center point (already checked)
                
                check_lat = lat + dlat
                check_lon = lon + dlon
                
                # Check if this point is over land using DEM
                if _dem_data is not None and _dem_transform is not None:
                    try:
                        row, col = rasterio.transform.rowcol(_dem_transform, check_lon, check_lat)
                        if 0 <= row < _dem_data.shape[0] and 0 <= col < _dem_data.shape[1]:
                            elevation = _dem_data[row, col]
                            if not np.isnan(elevation) and elevation > 0.5:
                                return True  # Found land in buffer
                    except Exception:
                        pass
    
    # If still not found, use fallback methods (shapefile or simplified mask)
    if not is_land:
        # Use existing fallback methods
        if PHIL_LAND is not None:
            try:
                pt = Point(lon, lat)
                if PHIL_LAND.contains(pt).any():
                    return True
                
                # Check with buffer (larger than before)
                buffer_size = 0.1  # approximately 10km near equator
                buffer_pt = pt.buffer(buffer_size)
                if PHIL_LAND.intersects(buffer_pt).any():
                    return True
            except Exception:
                pass
        
        # APPROACH 3: Fallback to enhanced simplified land mask (final fallback)
        # Luzon (northern Philippines)
        if 13.5 <= lat <= 19.5 and 119.5 <= lon <= 122.5:
            return True
        
        # Visayas (central Philippines)
        if 8.5 <= lat <= 12.5 and 122.5 <= lon <= 126.5:
            return True
        
        # Mindanao (southern Philippines)
        if 4.5 <= lat <= 9.5 and 121.5 <= lon <= 126.5:
            return True
        
        # Additional island regions
        # Palawan
        if 8.0 <= lat <= 12.0 and 117.0 <= lon <= 120.0:
            return True
        
        # Batanes group (northernmost)
        if 20.0 <= lat <= 21.0 and 121.5 <= lon <= 122.5:
            return True
            
        # Babuyan Islands
        if 19.0 <= lat <= 20.0 and 121.0 <= lon <= 122.0:
            return True
    
    return is_land

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth using Haversine formula.
    """
    # Earth radius in kilometers
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    
    return distance
        
def load_track_coefficients(coef_file=None):
    """
    Load track coefficients from pickle file, or use defaults if file not found.
    
    Args:
        coef_file: Path to coefficients pickle file
        
    Returns:
        Dictionary with track coefficients
    """
    import pickle
    
    # Default coefficient file path
    if coef_file is None:
        # Set a default path that's unlikely to exist so it falls back to default coefficients
        coef_file = os.path.join(DATA_DIR, "nonexistent_path", "track_coefficients.pkl")
        # You can uncomment this if you want to use the saved coefficients
        # coef_file = os.path.join(DATA_DIR, "TCCOEFS", "tc_tracks_20250426_143810", "track_coefficients.pkl") 
    
    try:
        # Try to load the coefficients
        with open(coef_file, 'rb') as f:
            coefficients = pickle.load(f)
        logging.info(f"Loaded track coefficients from {coef_file}")
        return coefficients
        
    except Exception as e:
        logging.warning("Skipping the *.pkl file, using default coefficients.")
        #logging.info("Skipping the *.pkl file, using default coefficients")
        
        # Default coefficients for PAR region
        return {
            5: [-0.40, 0.60, 0.11, 0.65, 0.10, 0.05, 0.20, -0.15, 0.25],  # 5-10°N
            10: [-0.35, 0.65, 0.15, 0.70, 0.08, 0.05, 0.15, -0.10, 0.20],  # 10-15°N
            15: [-0.30, 0.70, 0.18, 0.75, 0.05, 0.10, 0.15, -0.05, 0.20],  # 15-20°N
            20: [-0.15, 0.75, 0.25, 0.80, 0.01, 0.10, 0.15, 0.05, 0.25]    # 20-25°N
        }

# Load track coefficients at program start
track_coefficients = load_track_coefficients()

def Basins_WMO(basin):
    """
    Define basin boundaries following Bloemendaal's STORM approach.
    Modified to focus on Western Pacific (WP) basin.
    """
    # We'll focus on Western Pacific (WP) basin which includes the Philippines
    # Return basin parameters: storms count, month, lat0, lat1, lon0, lon1
    if basin == 'WP':  # Western Pacific
        return None, None, 5, 60, 100, 180
    else:
        # Default to Western Pacific if any other basin is specified
        logging.warning(f"Basin '{basin}' not recognized, defaulting to Western Pacific")
        return None, None, 5, 60, 100, 180

def get_data_path(filename):
    """Get data file path with fallback options"""
    possible_paths = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(os.getcwd(), "data", filename),
        os.path.join(os.path.expanduser("~"), "storm_data", filename)
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Could not find {filename} in any expected location: {possible_paths}")
    
def Genesis_month(idx, storms):
    """
    Sample genesis months for each TC based on historical distribution.
    Fully based on Bloemendaal's SELECT_BASIN.py approach using external data.
    
    Parameters:
        idx (int): Basin index (e.g., 5 for Western Pacific)
        storms (int): Number of tropical cyclones to simulate
    
    Returns:
        List of genesis months
    """
    import os
    import numpy as np
    
    # Get the data file path using the helper function
    storm_data_path = get_data_path("GENESIS_MONTHS.npy")
    
    try:
        monthlist = np.load(storm_data_path, allow_pickle=True).item()
    except Exception as e:
        raise IOError(f"Failed to load GENESIS_MONTHS.npy from {storm_data_path}: {e}")
    
    if idx not in monthlist:
        available_indices = list(monthlist.keys())
        raise ValueError(f"Basin index {idx} not found in GENESIS_MONTHS.npy. Available indices: {available_indices}")
    
    return [np.random.choice(monthlist[idx]) for _ in range(storms)]

def load_historical_data():
    """
    Load and preprocess historical TC position, wind, and forecast data.
    """
    global _df_positions, _df_wind, _df_forecast, _historical_wind_speeds, _month_params_cache
    
    # Load position data
    if _df_positions is None:
        logging.info("Loading historical position data...")
        try:
            _df_positions = pd.read_csv(
                POSITIONS_PATH,
                low_memory=False,
                usecols=['SID', 'SEASON', 'NUMBER', 'BASIN', 'ISO_TIME', 'LAT', 'LON']
            )
            # Convert ISO_TIME to datetime
            _df_positions['ISO_TIME'] = pd.to_datetime(_df_positions['ISO_TIME'], errors='coerce')
            # Drop rows with invalid coordinates
            _df_positions = _df_positions.dropna(subset=['LAT', 'LON'])
            logging.info(f"Loaded {len(_df_positions)} historical position records")
        except Exception as e:
            logging.error(f"Error loading position data: {e}")
            _df_positions = pd.DataFrame()
    
    # Load wind data
    if _df_wind is None:
        logging.info("Loading historical wind data...")
        try:
            _df_wind = pd.read_csv(WIND_PATH, low_memory=False)
            # Extract wind speeds
            if 'TOK_WIND' in _df_wind.columns:
                _historical_wind_speeds = _df_wind['TOK_WIND'].dropna().values
            logging.info(f"Loaded {len(_df_wind)} historical wind records")
        except Exception as e:
            logging.error(f"Error loading wind data: {e}")
            _df_wind = pd.DataFrame()
            _historical_wind_speeds = np.array([30, 50, 70, 90, 110])  # Default values
    
    # Load forecast data
    if _df_forecast is None:
        logging.info("Loading forecast data...")
        try:
            _df_forecast = pd.read_csv(FORECAST_PATH)
            logging.info(f"Loaded {len(_df_forecast)} forecast records")
        except Exception as e:
            logging.error(f"Error loading forecast data: {e}")
            _df_forecast = pd.DataFrame()
    
    # Calculate monthly parameters if not already done
    if _month_params_cache is None and _df_positions is not None and _df_wind is not None:
        try:
            # Prepare monthly data
            monthly_data = prepare_monthly_data(_df_positions, _df_wind)
            
            # Calculate parameters
            _month_params_cache = calculate_monthly_parameters(monthly_data)
            logging.info("Calculated monthly parameters from historical data")
        except Exception as e:
            logging.error(f"Error calculating monthly parameters: {e}")
            # Set default monthly parameters
            _month_params_cache = {
                1: {'blend': 0.25, 'cap': 100, 'extreme_prob': 0.005},  # January - low activity
                2: {'blend': 0.25, 'cap': 80, 'extreme_prob': 0.005},  # February - low activity
                3: {'blend': 0.25, 'cap': 115, 'extreme_prob': 0.005},  # March - rare typhoons
                4: {'blend': 0.25, 'cap': 140, 'extreme_prob': 0.005},  # April - transition
                5: {'blend': 0.25, 'cap': 130, 'extreme_prob': 0.005},  # May - transition
                6: {'blend': 0.25, 'cap': 120, 'extreme_prob': 0.005},  # June - transition
                7: {'blend': 0.25, 'cap': 125, 'extreme_prob': 0.005},  # July - increasing activity
                8: {'blend': 0.25, 'cap': 140, 'extreme_prob': 0.005},  # August - high activity
                9: {'blend': 0.25, 'cap': 140, 'extreme_prob': 0.005},  # September - peak
                10: {'blend': 0.25, 'cap': 145, 'extreme_prob': 0.005},  # October - peak
                11: {'blend': 0.25, 'cap': 145, 'extreme_prob': 0.005},  # November - declining
                12: {'blend': 0.25, 'cap': 130, 'extreme_prob': 0.005},  # December - transition
            }
    try:
        # Clear any temporary variables that might be using memory
        del monthly_data
        # Force garbage collection
        import gc
        gc.collect()
    except:
        pass
    return True
    
    return True

def prepare_monthly_data(df_positions, df_wind):
    """
    Organize historical wind data by month.
    
    Args:
        df_positions: DataFrame with storm positions and timestamps
        df_wind: DataFrame with wind data
    
    Returns:
        Dictionary with month as key and array of wind speeds as value
    """
    monthly_data = {month: [] for month in range(1, 13)}
    
    if df_wind is not None and 'TOK_WIND' in df_wind.columns:
        if 'ISO_TIME' in df_wind.columns:
            # If wind data has timestamps
            for _, row in df_wind.iterrows():
                if pd.notna(row['ISO_TIME']) and pd.notna(row['TOK_WIND']):
                    try:
                        date = pd.to_datetime(row['ISO_TIME'])
                        month = date.month
                        monthly_data[month].append(row['TOK_WIND'])
                    except:
                        pass
        elif 'SID' in df_wind.columns and df_positions is not None:
            # If wind data can be linked to positions via storm ID
            for _, wind_row in df_wind.iterrows():
                if pd.notna(wind_row['SID']) and pd.notna(wind_row['TOK_WIND']):
                    storm_id = wind_row['SID']
                    # Find matching position data
                    storm_pos = df_positions[df_positions['SID'] == storm_id]
                    if not storm_pos.empty:
                        try:
                            # Get month from first position point
                            date = pd.to_datetime(storm_pos.iloc[0]['ISO_TIME'])
                            month = date.month
                            monthly_data[month].append(wind_row['TOK_WIND'])
                        except:
                            pass
    
    return {month: np.array(winds) for month, winds in monthly_data.items() if winds}

def calculate_monthly_parameters(historical_data_by_month):
    """
    Calculate month-specific distribution parameters based on historical data.
    
    Args:
        historical_data_by_month: Dictionary with month numbers (1-12) as keys
                                 and arrays of historical wind speeds as values
    
    Returns:
        Dictionary with monthly parameters
    """
    month_params = {}
    
    # For each month, calculate parameters based on that month's historical data
    for month, data in historical_data_by_month.items():
        if len(data) < 10:  # Not enough data for this month
            continue
            
        # Filter to storm-force winds
        storm_data = data[data >= 22]
        if len(storm_data) < 10:
            continue
            
        # Calculate blend weight based on goodness-of-fit tests
        # Higher weight to GEV when it fits better than Weibull for extremes
        weibull_params = weibull_min.fit(storm_data, floc=22)
        gev_params = genextreme.fit(storm_data)
        
        # Calculate AIC for each distribution
        weibull_aic = -2 * np.sum(weibull_min.logpdf(storm_data, *weibull_params)) + 6
        gev_aic = -2 * np.sum(genextreme.logpdf(storm_data, *gev_params)) + 6
        
        # Calculate blend weight based on relative performance for extremes
        # Focus on the upper quartile of the data
        upper_quartile = np.quantile(storm_data, 0.75)
        extreme_data = storm_data[storm_data >= upper_quartile]
        
        if len(extreme_data) > 0:
            try:
                weibull_extreme_aic = -2 * np.sum(weibull_min.logpdf(extreme_data, *weibull_params)) + 6
                gev_extreme_aic = -2 * np.sum(genextreme.logpdf(extreme_data, *gev_params)) + 6
            
                # Calculate blend weight (more weight to GEV if it models extremes better)
                denominator = weibull_extreme_aic + gev_extreme_aic
                if abs(denominator) < 1e-10: # Near-zero denominator check
                    blend = 0.25 # Default
                elif denominator > 0:
                    blend = gev_extreme_aic / denominator
                    # Invert since lower AIC is better
                    blend = 1 - blend
                    # Constrain to reasonable range
                    blend = max(0.05, min(0.28, blend))
                else:
                    # Both AICs are negative, use reciprocal approach
                    blend = abs(gev_extreme_aic) / (abs(weibull_extreme_aic) + abs(gev_extreme_aic))
                    blend = max(0.05, min(0.28, blend))
            except Exception as e:
                    logging.warning(f"Error calculating blend weight: {e}. Using default.")
                    blend = 0.30  # Default
        else:
            blend = 0.25 # Default
        
        # Calculate cap based on empirical maximum with safety margin
        empirical_max = np.max(storm_data)
        cap = min(empirical_max * 1.05, 126)  # 5% above maximum observed
        
        # Calculate extreme probability based on empirical frequency
        # Define extreme as top 1% of observations
        extreme_threshold = np.percentile(storm_data, 99)
        extreme_count = np.sum(storm_data >= extreme_threshold)
        extreme_prob = extreme_count / len(storm_data) * 0.10  # Scale factor increased
        extreme_prob = max(0.005, min(0.010, extreme_prob))  # Reasonable constraints
        
        month_params[month] = {
            'blend': blend,
            'cap': cap,
            'extreme_prob': extreme_prob
        }
    
    # Fill in missing months with interpolated or seasonal values
    all_months = set(range(1, 13))
    missing_months = all_months - set(month_params.keys())
    
    for month in missing_months:
        # Find nearest months with data
        available_months = sorted(month_params.keys())
        if not available_months:
            # No data for any month, use defaults
            month_params[month] = {'blend': 0.10, 'cap': 100, 'extreme_prob': 0.002}
            continue
            
        # Find closest available months
        distances = [min((month - m) % 12, (m - month) % 12) for m in available_months]
        nearest_month = available_months[np.argmin(distances)]
        
        # Use nearest month's parameters
        month_params[month] = month_params[nearest_month].copy()
    
    return month_params

def sample_from_historical_distribution(size, historical_winds, category=None):
    """
    Sample wind speeds with continuous distribution to avoid artificial gaps.
    """
    # Define category boundaries (keep existing)
    cat_bounds = {
        'TD': (22, 33),
        'TS': (34, 47),
        'STS': (48, 63),
        'TY': (64, 99),
        'STY': (100, 126)
    }
    
    # Filter by category if specified
    if category and category in cat_bounds:
        min_wind, max_wind = cat_bounds[category]
        filtered_winds = historical_winds[(historical_winds >= min_wind) & 
                                         (historical_winds <= max_wind)]
        
        if len(filtered_winds) < 10:
            filtered_winds = historical_winds
    else:
        filtered_winds = historical_winds
    
    # Create bins with finer resolution
    hist, bin_edges = np.histogram(filtered_winds, bins=60, range=(22, 126))
    
    # Calculate bin probabilities
    bin_sum = np.sum(hist)
    if bin_sum == 0:
        bin_probs = np.ones_like(hist) / len(hist)
    else:
        bin_probs = hist / bin_sum
    
    # Normalize probabilities
    bin_probs = np.nan_to_num(bin_probs)
    bin_probs = bin_probs / np.sum(bin_probs)
    
    # Sample bin indices according to their probabilities
    bin_indices = np.random.choice(len(hist), size=size, p=bin_probs)
    
    # Generate samples within each bin without rounding
    samples = []
    for idx in bin_indices:
        # Get bin boundaries
        lower = bin_edges[idx]
        upper = bin_edges[idx + 1]
        
        # Use uniform sampling within the bin WITHOUT ROUNDING
        sample = np.random.uniform(lower, upper)
        samples.append(sample)
    
    # Return continuous values
    return np.array(samples, dtype=np.float32)

def scale_winds_to_target_mean(storms_df, target_mean=58.0):
    """
    Scale all wind speeds in the dataset to achieve exact target mean.
    This should be called as the FINAL step after all other processing.
    
    Args:
        storms_df: DataFrame with storm data
        target_mean: Target mean wind speed in knots
        
    Returns:
        DataFrame with scaled wind speeds
    """
    # Make a copy to avoid modifying original
    scaled_df = storms_df.copy()
    
    # Get current mean (excluding sub-tropical depression winds)
    valid_winds = scaled_df[scaled_df['WIND'] >= 22.0]['WIND']
    current_mean = valid_winds.mean()
    
    if current_mean <= 0:
        logging.error("No valid wind speeds found for scaling")
        return scaled_df
    
    # Calculate scaling factor
    scale_factor = target_mean / current_mean
    
    logging.info(f"Scaling winds from current mean {current_mean:.2f} to target {target_mean:.2f} (factor: {scale_factor:.3f})")
    
    # Apply scaling BUT preserve extreme winds (>=120 kt)
    extreme_mask = scaled_df['WIND'] >= 120.0
    extreme_winds = scaled_df.loc[extreme_mask, 'WIND'].copy()  # Save extreme winds

    # Scale all winds
    scaled_df['WIND'] = scaled_df['WIND'] * scale_factor

    # Restore extreme winds to preserve 124+ kt storms
    if extreme_mask.any():
        scaled_df.loc[extreme_mask, 'WIND'] = extreme_winds
        logging.info(f"Preserved {extreme_mask.sum()} extreme winds >=120kt from scaling")

    # Ensure we don't exceed physical limits
    scaled_df['WIND'] = scaled_df['WIND'].clip(upper=126.0)
    
    # Ensure TD threshold is maintained
    scaled_df.loc[scaled_df['WIND'] < 22.0, 'WIND'] = 22.0
    
    # Recalculate pressure and other derived fields
    scaled_df['PRES'] = calculate_pressure_from_wind(
        scaled_df['WIND'].values, 
        env_pressure=1010.0, 
        lat=scaled_df['LAT'].values
    )
    
    # Update categories
    categories = []
    for wind in scaled_df['WIND']:
        if wind >= 100:
            categories.append("Super Typhoon")
        elif wind >= 64:
            categories.append("Typhoon")
        elif wind >= 48:
            categories.append("Severe Tropical Storm")
        elif wind >= 34:
            categories.append("Tropical Storm")
        elif wind >= 22:
            categories.append("Tropical Depression")
        else:
            categories.append("Remnant Low")
    scaled_df['CATEGORY'] = categories
    
    # Verify the new mean
    new_mean = scaled_df[scaled_df['WIND'] >= 22.0]['WIND'].mean()
    logging.info(f"Final wind speed mean: {new_mean:.2f} knots")
    
    return scaled_df

def sample_blend_truncated(size, cap, blend_weight,
                           gev_shape, gev_loc, gev_scale,
                           wbl_shape, wbl_loc, wbl_scale):
    out = []
    while len(out) < size:
        n_draw = size - len(out)
        g = genextreme.rvs(gev_shape, loc=gev_loc, scale=gev_scale, size=n_draw)
        w = weibull_min.rvs(wbl_shape, loc=wbl_loc, scale=wbl_scale, size=n_draw)
        b = blend_weight * g + (1 - blend_weight) * w
        valid = b[b < cap]
        out.extend(valid.tolist())
    return np.array(out[:size], dtype=np.float32)

def weibull_gev_blend(size, historical_data=None, month=None, region=None, year=None):
    """
    Generate wind speeds using a blend of Weibull and GEV distributions.
    Calibrated to achieve a 58-knot mean (2024–2124), with optional year trend.
    
    Args:
        size: Number of samples
        historical_data: array of historical wind speeds (or None)
        month: 1–12 for seasonal caps/blend
        region: unused placeholder for future regional tweaks
        year: if >=2024, applies a linear trend to intensity_factor
        
    Returns:
        NumPy array of wind speeds (knots), clipped to [22, 126].
    """
    global _monthly_category_dist, _historical_wind_speeds, _month_params_cache
    
    # Ensure size is scalar
    size = int(ensure_scalar(size, default=1))
    
    # Ensure historical_data is a proper numpy array if provided
    if historical_data is not None:
        if not isinstance(historical_data, np.ndarray):
            historical_data = np.array(historical_data).flatten()
        # Remove any NaN or infinite values
        historical_data = historical_data[np.isfinite(historical_data)]
    
    # 1) Compute intensity_factor
    if year is not None and year >= 2024:
        base_year      = 2024
        trend_per_year = 0.11024   # derived so 1.20833 at year=2074
        years_elapsed  = min(year - base_year, 100)
        # divide by original base 53 to keep trend consistent
        intensity_factor = 1.0 + (years_elapsed * trend_per_year / 53.0)
    else:
        # mid-period default to scale raw ~48->58 knots
        intensity_factor = 1.20833

    # 2) Parameter definitions (shared by both branches)
    weibull_default  = (2.2, 25.0, 20.0)   # shape, loc, scale [historical: 1.8, 20.0, 18.0] [fut. estimate: (2.2, 25.0, 20.0)]
    gev_default      = (-0.15, 65.0, 25.0)  # shape, loc, scale [historical: -0.18, 55, 18.0] [fut. estimate: (-0.15, 65.0, 25.0)]
    gev_blend_weight = 0.40                # weight on GEV [historical: 0.25] -> 25% GEV, 75% Weibull [fut. estimate: 0.40]
    extreme_prob     = 0.010               # chance of extra boost, from 0.020 to 0.010 [historical: 0.005] -> 0.5% chance [fut. est.: 0.010]
    absolute_cap     = 126.0               # global maximum [historical: 125] [empirical future: 126.0]

    # 3) Build bumped monthly params (+5 knots each)
    if _month_params_cache:
        month_params = _month_params_cache
    else:
        base_caps = {1:100, 2:95, 3:115, 4:120, 5:120, 6:120,
                     7:126, 8:126, 9:126,10:126,11:120,12:115}
        month_params = {
            m: {
                'blend': gev_blend_weight,
                'cap': min(absolute_cap, cap + 5),
                'extreme_prob': extreme_prob
            }
            for m, cap in base_caps.items()
        }

    params = month_params.get(month, {
        'blend': gev_blend_weight,
        'cap': absolute_cap,
        'extreme_prob': extreme_prob
    })

    # 4) Fallback branch (no historical data)
    if historical_data is None:
        if _historical_wind_speeds is not None:
            historical_data = _historical_wind_speeds
        else:
            logging.warning("No historical wind data available; using parametric fallback")
            
            # 1) draw all blends by rejection so no hard caps ever appear
            blend = sample_blend_truncated(
                size, 
                params['cap'],
                params['blend'],
                gev_default[0], gev_default[1], gev_default[2],
                weibull_default[0], weibull_default[1], weibull_default[2],
            )

            # 2) optional triangular nudge—but force it < cap
            #mask = np.random.random(size) < params['extreme_prob']
            #if mask.any():
            #    low, mode, high = (0.70*params['cap'],
            #                       0.75*params['cap'],
            #                       0.90*params['cap'])
            #    tail = np.random.triangular(low, mode, high, mask.sum())
                # ensure we never push up to the cap
            #    tail = np.minimum(tail, params['cap'] - 1e-3)
            #    blend[mask] = np.maximum(blend[mask], tail)

            # 3) apply intensity_factor, floor, and post‐calibration…
            blend *= intensity_factor
            blend = np.maximum(blend, 22.0).astype(np.float32)

            # -- post-hoc auto-calibration to exact 58 knots mean --
            mean0 = blend.mean()
            if mean0 > 0:
                scale_to_target = 58.0 / mean0
                blend *= scale_to_target
                blend = np.clip(blend, 22.0, absolute_cap)
            
            # Average WIND
            #base_factor = 58.0 / 53.0
            #blend = np.clip(blend * base_factor, 22.0, absolute_cap)

            logging.info(
                f"Fallback stats -> mean={blend.mean():.2f}, max={blend.max():.2f}"
            )
            return blend

    # 5) Historical‐data branch
    # (sample per category, scale, shuffle, pad/trim)
    if month in _monthly_category_dist:
        dist = _monthly_category_dist[month]
    
        # NEW: Shift distribution toward TS while reducing extremes
        adjusted_dist = [
            dist[0] * 0.80,  # TD - reduce [historical: 1.50] [fut: 0.80]
            dist[1] * 1.60,  # TS - increase [historical: 1.00] [fut: 1.60]
            dist[2] * 1.50,  # STS - increase [historical: 0.70] [fut: 1.50]
            dist[3] * 1.40,  # TY - increase [historical: 0.50] [fut: 1.40]
            dist[4] * 0.15   # STY - reduced from 0.30 to 0.10 [historical: 0.30] [fut: 0.10 or 0.15]
        ]
        # Normalize to 100%
        total = sum(adjusted_dist)
        adjusted_dist = [d/total * 100 for d in adjusted_dist]
    
        # Use adjusted_dist instead of dist
        counts = [
            max(1, int(size * pct / 100))
            for pct in adjusted_dist[:-1]
        ]
        counts.append(max(0, int(size * adjusted_dist[-1] / 100)))
        # adjust to exactly `size`
        total = sum(counts)
        counts[0] += (size - total)

        cats = ['TD', 'TS', 'STS', 'TY', 'STY']
        samples = []
        for cat, cnt in zip(cats, counts):
            if cnt > 0:
                samples.append(
                    sample_from_historical_distribution(cnt, historical_data, cat)
                )
        blend = np.concatenate(samples)

    else:
        blend = sample_from_historical_distribution(size, historical_data)

    # -- apply trend scaling --
    blend = blend * intensity_factor

    # -- shuffle, pad/trim --
    np.random.shuffle(blend)
    if len(blend) < size:
        miss = size - len(blend)
        extra = sample_from_historical_distribution(miss, historical_data, 'TD')
        blend = np.append(blend, extra * intensity_factor)
    # 1) Trim to size
    blend = blend[:size]

    # 2) Smooth high-end distribution instead of rejection resampling
    # Use sigmoid compression for values approaching cap
    #for i in range(len(blend)):
    #    if blend[i] > 115:  # # Start transition higher (was 110 kts)
            # Sigmoid compression toward higher ceiling
    #        x = blend[i] - 115 
    #        range_available = 20.0  # Increased from 15.5
    #        compressed = 115 + range_available * (2 / (1 + np.exp(-x/10)) - 1)
    #        blend[i] = min(compressed, 135)
    
    # 3) Triangular “nudge” on the top end—but keep it strictly below cap
    #mask = np.random.random(size) < params['extreme_prob']
    #if mask.any():
    #    low, mode, high = (
    #        0.70 * params['cap'],
    #        0.75 * params['cap'],
    #        0.80 * params['cap'],
    #    )
    #    tail = np.random.triangular(low, mode, high, mask.sum())
    #    tail = np.minimum(tail, params['cap'] - 1e-3)   # never hit the cap exactly
    #    blend[mask] = np.maximum(blend[mask], tail)

    # 4) Enforce the 22-kt lower bound
    blend = np.maximum(blend, 22.0)

    # 1) First validate (floor, cap, any other corrections)
    validated = validate_wind_speeds(blend.astype(np.float32),
                                     caller="weibull_gev_blend")
    
    # allows higher values but smooths distribution:
    if np.any(blend > 120):
        # Apply gentler smoothing to extreme values
        extreme_indices = np.where(blend > 120)[0]
        
        # Only reduce 20% of extreme values slightly, keep 80% at full intensity
        num_to_reduce = int(len(extreme_indices) * 0.2)
        if num_to_reduce > 0:
            indices_to_reduce = np.random.choice(extreme_indices, num_to_reduce, replace=False)
            for idx in indices_to_reduce:
                # Minor smoothing - reduce by only 5-10%, keeping in extreme range
                blend[idx] = blend[idx] * np.random.uniform(0.90, 0.95)
            
        logging.info(f"Preserved {len(extreme_indices) - num_to_reduce} extreme winds >=120kt out of {len(extreme_indices)}")

        for idx in extreme_indices:
            # 80% chance to keep values with minor smoothing
            if np.random.random() < 0.8:
                # Minor smoothing - reduces slightly but keeps in STY range
                blend[idx] = 100 + (blend[idx] - 100) * 0.95
            
    # DEBUG to see what is going on:
    logging.info(f"Raw distribution stats BEFORE scaling: mean={validated.mean():.2f}, "
                f"min={validated.min():.2f}, max={validated.max():.2f}, "
                f"median={np.median(validated):.2f}")
    
    # 2) Then post-hoc scale to EXACTLY 58 knots *after* validate
    mean_valid = validated.mean()
    if mean_valid > 0:
        scale_to_target = 58.0 / mean_valid
        validated = validated * scale_to_target
        
        # 3) Simple clipping without any jitter or complex decay
        validated = np.clip(validated, 22.0, 126.0)

    # 4) Final logging survives any console encoding pitfalls
    logging.info("Historical-data stats -> mean=%.2f, max=%.2f",
                 validated.mean(), validated.max())

    return validated

def adjust_wind_profile(base_winds, target_max_wind, storm_category):
    """
    Adjust a wind speed profile to match a target maximum wind and ensure
    it conforms to the expected intensification and weakening patterns for
    the specified storm category.
    
    Args:
        base_winds: Array of baseline wind speeds
        target_max_wind: Target maximum wind speed to reach
        storm_category: Category of storm ('TD', 'TS', 'STS', 'TY', 'STY')
        
    Returns:
        Array of adjusted wind speeds
    """
    n = len(base_winds)
    
    # Create a new wind profile array
    adjusted_winds = np.zeros(n)
    
    # Determine profile parameters based on storm category - OPTIMIZED VALUES
    if storm_category == 'STY':
        # Super Typhoons: rapid intensification, usually maintain peak for longer
        ramp_up_fraction = np.random.uniform(0.25, 0.30)  # First 30-35% is intensification
        peak_fraction = np.random.uniform(0.35, 0.45)     # Next 20-25% is peak intensity
        ramp_down_fraction = 1.0 - ramp_up_fraction - peak_fraction  # Remaining is weakening
    elif storm_category == 'TY':
        # Typhoons: steady intensification, moderate peak duration
        ramp_up_fraction = np.random.uniform(0.35, 0.40)  # First 35-40% is intensification
        peak_fraction = np.random.uniform(0.15, 0.20)     # Next 15-20% is peak intensity
        ramp_down_fraction = 1.0 - ramp_up_fraction - peak_fraction  # Remaining is weakening
    elif storm_category == 'STS':
        # Severe Tropical Storms: moderate intensification, shorter peak
        ramp_up_fraction = np.random.uniform(0.35, 0.40)  # First 35-40% is intensification
        peak_fraction = np.random.uniform(0.10, 0.15)     # Next 10-15% is peak intensity
        ramp_down_fraction = 1.0 - ramp_up_fraction - peak_fraction  # Remaining is weakening
    elif storm_category == 'TS':
        # Tropical Storms: gradual intensification, brief peak
        ramp_up_fraction = np.random.uniform(0.40, 0.45)  # First 40-45% is intensification
        peak_fraction = np.random.uniform(0.10, 0.15)     # Next 10-15% is peak intensity
        ramp_down_fraction = 1.0 - ramp_up_fraction - peak_fraction  # Remaining is weakening
    else:  # TD
        # Tropical Depressions: slow intensification, minimal peak
        ramp_up_fraction = np.random.uniform(0.45, 0.50)  # First 45-50% is intensification
        peak_fraction = np.random.uniform(0.05, 0.10)     # Next 5-10% is peak intensity
        ramp_down_fraction = 1.0 - ramp_up_fraction - peak_fraction  # Remaining is weakening
    
    # Calculate indices for each phase
    ramp_up_end = int(n * ramp_up_fraction)
    peak_end = ramp_up_end + int(n * peak_fraction)
    
    # Set minimum wind speed (usually TD strength)
    min_wind = max(22.0, base_winds.min())
    
    # Create the wind profile
    # Intensification phase
    for i in range(ramp_up_end):
        # Sigmoid-like intensification (steeper for stronger storms)
        progress = i / max(1, ramp_up_end - 1)
        if storm_category in ['STY', 'TY']:
            # More rapid intensification for stronger storms
            factor = 1.0 - np.cos(progress * np.pi/2)
        else:
            # More gradual intensification for weaker storms
            factor = progress ** 1.5
        adjusted_winds[i] = min_wind + (target_max_wind - min_wind) * factor
    
    # Peak intensity phase
    for i in range(ramp_up_end, peak_end):
        # Add slight fluctuations to peak intensity
        fluctuation = np.random.uniform(0.97, 1.03)
        adjusted_winds[i] = target_max_wind * fluctuation
    
    # Weakening phase
    for i in range(peak_end, n):
        # Calculate decay progress (0 at peak_end, 1 at end)
        progress = (i - peak_end) / max(1, n - peak_end - 1)
        
        # Different decay patterns by category
        if storm_category == 'STY':
            # Super Typhoons can maintain strength longer then decay more rapidly
            if progress < 0.3:
                decay_factor = progress * 0.5  # Slow initial decay
            else:
                decay_factor = 0.15 + (progress - 0.3) * 1.2  # Faster later decay
        elif storm_category == 'TY':
            # Typhoons have more steady decay
            decay_factor = progress * 0.9
        else:
            # Weaker storms decay more gradually
            decay_factor = progress * 0.8
        
        # Ensure decay factor is within [0, 1]
        decay_factor = min(1.0, max(0.0, decay_factor))
        
        # Calculate wind speed
        adjusted_winds[i] = target_max_wind - (target_max_wind - min_wind) * decay_factor
    
    # Apply optimized smoothing with Gaussian filter
    from scipy.ndimage import gaussian_filter1d
    adjusted_winds = gaussian_filter1d(adjusted_winds, sigma=0.05)  # Much lighter smoothing
    
    # Ensure the maximum wind matches the target (may have been reduced by smoothing)
    max_idx = np.argmax(adjusted_winds)
    if adjusted_winds[max_idx] < target_max_wind:
        scale_factor = target_max_wind / adjusted_winds[max_idx]
        # Only scale up the peaks, not the entire profile
        kernel_size = max(3, n // 10)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        peak_region = slice(max(0, max_idx - kernel_size//2), min(n, max_idx + kernel_size//2 + 1))
        adjusted_winds[peak_region] *= scale_factor
    
    # Ensure all winds are at least tropical depression strength
    adjusted_winds = np.maximum(adjusted_winds, min_wind)
    
    # Prevent re-intensification of weak storms that have fallen below TD strength
    for i in range(1, len(adjusted_winds)):
        if i > 1 and adjusted_winds[i-1] < 22.0 and adjusted_winds[i-2] < 22.0:
            # If two consecutive points are below TD strength, prevent significant re-intensification
            adjusted_winds[i] = min(adjusted_winds[i], adjusted_winds[i-1] + 2.0)  # Allow only small increases
        
    return validate_wind_speeds(adjusted_winds, caller="adjust_wind_profile")

def sample_rmax(p, rmax_pres):
    """
    Sample radius of maximum winds based on pressure categories.
    Directly from Bloemendaal's SAMPLE_RMAX.py
    """
    if p > 940:
        r = np.random.choice(rmax_pres[2], 1)
    elif p <= 940 and p > 920:
        r = np.random.choice(rmax_pres[1], 1)
    else:
        r = np.random.choice(rmax_pres[0], 1)
    return float(r)
    
def calculate_pressure_from_wind(wind_speed, env_pressure=1010.0, lat=None):
    """
    Calculate central pressure from wind speed using Atkinson-Holliday formula
    with latitude adjustment. Adapted from SAMPLE_TC_PRESSURE.py.
    
    Args:
        wind_speed: Wind speed in knots
        env_pressure: Environmental pressure in hPa
        lat: Latitude for adjustment
        
    Returns:
        Central pressure in hPa
    """
    # If wind_speed is an array, handle it vectorized
    if isinstance(wind_speed, (list, np.ndarray)) and hasattr(wind_speed, '__iter__'):
        # Ensure wind_speed is a numpy array
        wind_speed = np.array(wind_speed)
        
        # Calculate base pressure drop using Atkinson-Holliday relationship
        delta_p = (wind_speed / 5.896) ** (1 / 0.644)
        
        # Apply latitude adjustment if provided
        if lat is not None and hasattr(lat, '__iter__'):
            lat = np.array(lat)
            # Holland B parameter varies with latitude - higher values at higher latitudes
            B_factor = 1.5 - 0.5 * np.cos(np.radians(np.abs(lat)))
            # Normalize to get adjustment factor
            lat_adjustment = B_factor / 1.5
            # Apply adjustment to pressure drop
            delta_p = delta_p * lat_adjustment
        
        central_pressure = env_pressure - delta_p
        
        # Ensure pressure is within reasonable bounds
        return np.clip(central_pressure, 850, env_pressure)
    else:
        # Handle scalar inputs
        # Calculate base pressure drop using Atkinson-Holliday relationship
        delta_p = (wind_speed / 5.896) ** (1 / 0.644)
        
        # Apply latitude adjustment if provided
        if lat is not None:
            # Holland B parameter varies with latitude - higher values at higher latitudes
            B_factor = 1.5 - 0.5 * np.cos(np.radians(np.abs(lat)))
            # Normalize to get adjustment factor
            lat_adjustment = B_factor / 1.5
            # Apply adjustment to pressure drop
            delta_p = delta_p * lat_adjustment
        
        central_pressure = env_pressure - delta_p
        
        # Ensure pressure is within reasonable bounds
        return np.clip(central_pressure, 850, env_pressure)

def sample_windpressure_coefficients(basin, month):
    """
    Sample wind-pressure relationship coefficients for a given basin and month.
    Based on historical data from the Philippine region.
    
    Returns:
        List of coefficients [a, b] for the wind-pressure relationship
    """
    # For Western Pacific basin (includes Philippines)
    # Format: [a, b] for each month where V = a*(P_env - P_c)^b (Atkinson and Holliday, 1977)
    
    # Monthly coefficients for Western Pacific
    coefficients = {
        1: [6.0, 0.672],  # January
        2: [6.0, 0.693],  # February
        3: [6.0, 0.65],  # March
        4: [6.0, 0.639],  # April
        5: [6.0, 0.642],  # May
        6: [6.0, 0.646],  # June
        7: [6.0, 0.638],  # July
        8: [6.0, 0.639],  # August
        9: [6.0, 0.638],  # September
        10: [6.0, 0.637],  # October
        11: [6.0, 0.641],  # November
        12: [6.0, 0.653],  # December
    }
    
    # Return the coefficients for the specified month
    return coefficients.get(month, [3.0, 0.644])  # Default if month not found

def calculate_rmw(wind_speed, lat=None):
    """
    Calculate Radius of Maximum Wind (RMW) based on wind speed and latitude.
    Adapted from Bloemendaal's SAMPLE_RMAX.py.
    
    Args:
        wind_speed: Wind speed in knots
        lat: Latitude in degrees
        
    Returns:
        RMW in kilometers
    """
    if wind_speed is None or wind_speed <= 0:
        return 50.0  # Default RMW
    
    # Use default latitude if not provided
    if lat is None:
        lat = 15.0  # Default to central Philippines
    
    # Calculate RMW using the formula from Bloemendaal
    rmw = 218.3784 - 1.2014 * wind_speed + ((wind_speed) / 10.9844) ** 2 - \
          ((wind_speed) / 35.3052) ** 3 - 145.5090 * np.cos(np.radians(lat))
    
    # Clip to reasonable values
    return np.clip(rmw, 5.0, 150.0)
    
def find_lat_index_bins(basin, lat):
    """
    Find index of latitude bin in coefficients list.
    From Bloemendaal's SAMPLE_TC_MOVEMENT.py
    """
    s, monthdummy, lat0, lat1, lon0, lon1 = Basins_WMO(basin)
    base = 5
    latindex = np.floor(float(lat-lat0)/base)
    return latindex

def LAT_JAMES_MASON(dlat, lat, a, b, c):
    """
    Calculate forward change in latitude using James-Mason formula.
    From Bloemendaal's SAMPLE_TC_MOVEMENT.py
    """
    y = a + b*dlat + c/lat
    return y

def LON_JAMES_MASON(dlon, a, b):
    """
    Calculate forward change in longitude using James-Mason formula.
    From Bloemendaal's SAMPLE_TC_MOVEMENT.py
    """
    y = a + b*dlon
    return y

def calculate_decay_parameters(elevation, landfall_wind):
    """Calculate decay parameters based on actual elevation and storm intensity."""
    
    # Base decay rate calibrated for Philippines
    alpha_base = 0.031
    
    # Create continuous relationship between elevation and terrain factor
    # Exponential relationship better captures impact of mountains
    terrain_factor = 1.0 + 1.5 * (1.0 - np.exp(-elevation / 500.0))
    
    # Scale intensity factor based on storm strength with stronger effect for extreme storms
    if landfall_wind >= 106:  # Extreme super typhoons
        intensity_factor = 1.8 + 0.7 * (landfall_wind - 106) / 20.0  # Scales with intensity beyond 106kt
    elif landfall_wind >= 100:  # Super typhoons
        intensity_factor = 1.5
    elif landfall_wind >= 80:
        intensity_factor = 1.3
    elif landfall_wind >= 64:
        intensity_factor = 1.1
    else:
        intensity_factor = 0.95
    
    # Calculate local terrain roughness based on actual elevation
    roughness_factor = 1.0 + 0.5 * (1.0 - np.exp(-elevation / 1000.0))
    
    # Calculate final decay parameter with higher upper bound for extreme storms
    alpha = alpha_base * terrain_factor * intensity_factor * roughness_factor
    
    # Apply constraints based on storm category
    #max_alpha = 0.3 if landfall_wind >= 106 else 0.2 if landfall_wind >= 100 else 0.15
    #alpha = min(max_alpha, max(0.01, alpha))
    
    # Adjust max_alpha with lower values for intense storms
    max_alpha = 0.25 if landfall_wind >= 106 else 0.18 if landfall_wind >= 100 else 0.15
    alpha = min(max_alpha, max(0.01, alpha))
    
    return alpha

def apply_orographic_effects(storm_df, dem_path):
    """Apply realistic orographic effects to storm tracks."""
    modified_df = storm_df.copy()
    
    # Only process extreme storms
    max_wind = modified_df['WIND'].max()
    if max_wind < 100:  # Not an extreme storm
        return modified_df
    
    track_lats = modified_df['LAT'].values
    track_lons = modified_df['LON'].values
    
    # Check for approaching major terrain
    for i in range(1, len(track_lats) - 1):
        # Look ahead a few points to check coming terrain
        ahead_points = min(3, len(track_lats) - i - 1)
        terrain_ahead = False
        max_elev_ahead = 0
        
        for j in range(1, ahead_points + 1):
            look_lat = track_lats[i + j]
            look_lon = track_lons[i + j]
            
            if is_over_land(look_lon, look_lat):
                elev = get_elevation(look_lon, look_lat)
                max_elev_ahead = max(max_elev_ahead, elev)
                if elev > 500:  # Significant terrain ahead
                    terrain_ahead = True
        
        # If extreme storm approaching high terrain, apply realistic deflection
        if terrain_ahead and max_elev_ahead > 800 and max_wind >= 106:
            # Calculate dominant terrain orientation (simplified for Philippines)
            # Philippine mountains generally run north-south
            terrain_angle = 0  # North-south orientation (in radians)
            
            # Calculate current storm heading
            if i > 0:
                heading = np.arctan2(
                    track_lats[i] - track_lats[i-1],
                    track_lons[i] - track_lons[i-1]
                )
                
                # Calculate angle between storm track and terrain orientation
                angle_diff = np.abs((heading - terrain_angle + np.pi) % (2 * np.pi) - np.pi)
                
                # If storm is approaching terrain at significant angle
                if angle_diff > np.pi/6:  # More than 30 degrees
                    # Apply deflection proportional to elevation and approach angle
                    deflection_strength = (max_elev_ahead / 2000.0) * (angle_diff / (np.pi/2))
                    
                    # Determine deflection direction (follow terrain)
                    deflection_dir = 1 if heading > terrain_angle else -1
                    
                    # Apply deflection to remaining track
                    for k in range(i+1, len(track_lats)):
                        # Gradually increase deflection
                        deflection = deflection_strength * min(1.0, (k - i) / 3.0)
                        
                        # Calculate distance from current point
                        dist = np.sqrt(
                            (track_lats[k] - track_lats[i])**2 + 
                            (track_lons[k] - track_lons[i])**2
                        )
                        
                        # Apply terrain-following deflection
                        if dist > 0:
                            # Perpendicular to terrain orientation
                            modified_df.loc[k, 'LON'] += deflection * deflection_dir * np.sin(terrain_angle)
                            modified_df.loc[k, 'LAT'] += deflection * deflection_dir * np.cos(terrain_angle)
    
    return modified_df

def apply_enhanced_termination_rules(storm_df):
    """Apply three-strike rule for more robust detection of genuine weakening."""
    df = storm_df.copy()
    n = len(df)
    
    # Track positions and winds
    winds = df['WIND'].values
    
    # Apply three-strike rule for open water termination
    if 'ELEVATION' in df.columns:
        over_land = df['ELEVATION'].values > 0.5
    else:
        # Fallback if elevation data not available
        over_land = np.array([is_over_land(lon, lat) for lon, lat in zip(df['LON'], df['LAT'])])
    
    # Initialize variables
    consec_low = 0
    term_idx = None
    
    # Start from beginning or after landfall
    landfall_idx = np.where(over_land)[0][0] if np.any(over_land) else 0
    water_exit_idx = None
    
    # Find water exit index if there was landfall
    if landfall_idx > 0:
        for j in range(landfall_idx+1, n):
            if not over_land[j]:
                water_exit_idx = j
                break
    
    # Start checking after water exit or from beginning if no landfall
    start_idx = water_exit_idx if water_exit_idx is not None else 0
    
    # Apply three-strike rule (instead of two-strike)
    for i in range(start_idx, n):
        if winds[i] < 22.0:
            consec_low += 1
            if consec_low >= 3:  # Changed from 2 to 3
                term_idx = i - 2
                break
        else:
            consec_low = 0
    
    # Truncate track if three-strike rule was triggered
    if term_idx is not None:
        df = df.iloc[:term_idx+3]  # Include all three sub-threshold points
    
    return df

def apply_aging_decay(storm_df):
    """Apply gradual intensity reduction for long-lived storms."""
    df = storm_df.copy()
    
    # Check if the storm has time information
    if 'ISO_TIME' not in df.columns:
        # Create simple timesteps (assuming 3-hour intervals)
        timesteps = np.arange(len(df)) * 3  # Hours since genesis
    else:
        # Convert timestamps to hours since genesis
        genesis_time = pd.to_datetime(df['ISO_TIME'].iloc[0])
        timesteps = [(pd.to_datetime(t) - genesis_time).total_seconds() / 3600 
                     for t in df['ISO_TIME']]
    
    # Define aging parameters
    typical_lifespan_hours = 192  # 8 days at 3-hour intervals (64 points)
    max_lifespan_hours = 240      # 10 days
    
    # Only apply aging decay if storm exceeds typical lifespan
    if max(timesteps) > typical_lifespan_hours:
        winds = df['WIND'].values.copy()
        
        # Find maximum wind before typical lifespan
        early_idx = np.where(np.array(timesteps) <= typical_lifespan_hours)[0]
        if len(early_idx) > 0:
            peak_wind = max(winds[early_idx])
        else:
            peak_wind = max(winds)
        
        # Apply gradual decay for points beyond typical lifespan
        for i, hours in enumerate(timesteps):
            if hours > typical_lifespan_hours:
                # Calculate how far past typical lifespan
                excess_fraction = min(1.0, (hours - typical_lifespan_hours) / 
                                     (max_lifespan_hours - typical_lifespan_hours))
                
                # Apply increasing decay factor
                decay_factor = 1.0 - (0.5 * excess_fraction * excess_fraction)
                
                # Ensure wind doesn't increase
                max_wind = min(peak_wind * decay_factor, winds[i])
                winds[i] = max_wind
        
        # Update winds
        df['WIND'] = winds
        
        # Update derived fields
        df['PRES'] = calculate_pressure_from_wind(winds, env_pressure=1010.0, lat=df['LAT'])
        if 'RMW' in df.columns:
            df['RMW'] = [calculate_rmw(w, la) for w, la in zip(winds, df['LAT'])]
        
        # Update categories
        categories = []
        for w in winds:
            if w >= 100:
                categories.append("Super Typhoon")
            elif w >= 64:
                categories.append("Typhoon")
            elif w >= 48:
                categories.append("Severe Tropical Storm")
            elif w >= 34:
                categories.append("Tropical Storm")
            elif w >= 22:
                categories.append("Tropical Depression")
            else:
                categories.append("Remnant Low")
        df['CATEGORY'] = categories
    
    return df
    
def preserve_eastern_approaches(storm_df):
    """Ensure storms approaching from the east maintain extreme intensity"""
    
    # Check if storm approaches Philippines from the east
    eastern_approach = False
    for i in range(min(10, len(storm_df))):
        if storm_df.iloc[i]['LON'] > 130 and storm_df.iloc[i]['LAT'] < 20:  # More restrictive
            eastern_approach = True
            break
    
    if eastern_approach:
        # Find points near FAR eastern Philippines only
        for i, row in storm_df.iterrows():
            # CHANGED: More restrictive longitude range
            if (122 < row['LON'] < 127 and 10 < row['LAT'] < 18):  # Include eastern Luzon seaboard
                # If this was originally a very strong storm, preserve some intensity
                max_intensity = storm_df['WIND'].max()
                if max_intensity >= 106 and row['WIND'] < 106:
                    # Restore intensity near eastern coastline
                    storm_df.loc[i, 'WIND'] = min(max_intensity * 0.95, 126)
                    logging.info(f"PRESERVED EASTERN INTENSITY: {row['WIND']:.1f}kt -> {storm_df.loc[i, 'WIND']:.1f}kt")
    
    return storm_df
    
def filter_anomalous_terminations(all_storms_df, max_anomalies=50):
    """
    Filter out statistically anomalous terminations while preserving storm count.
    
    Args:
        all_storms_df: DataFrame with all synthetic storms
        max_anomalies: Maximum number of storms to filter
    
    Returns:
        DataFrame with anomalous terminations replaced
    """
    # Group by storm ID
    storm_ids = all_storms_df['SID'].unique()
    required_count = len(storm_ids)
    
    # Dictionary to store termination statistics
    terminations = {}
    
    # Calculate termination statistics for each storm
    for sid in storm_ids:
        storm_data = all_storms_df[all_storms_df['SID'] == sid]
        
        # Get final point
        final_point = storm_data.iloc[-1]
        final_wind = final_point['WIND']
        final_lat = final_point['LAT']
        final_lon = final_point['LON']
        
        # Check if final point is near domain boundary
        near_west = final_lon < PAR_BOUNDS[2][1] + 1.0
        near_east = final_lon > PAR_BOUNDS[0][1] - 1.0
        near_south = final_lat < PAR_BOUNDS[2][0] + 1.0
        near_north = final_lat > PAR_BOUNDS[0][0] - 1.0
        near_boundary = near_west or near_east or near_south or near_north
        
        # Check if over land
        over_land = False
        if 'ELEVATION' in final_point:
            over_land = final_point['ELEVATION'] > 0.5
        else:
            over_land = is_over_land(final_lon, final_lat)
        
        # Store termination data
        terminations[sid] = {
            'final_wind': final_wind,
            'near_boundary': near_boundary,
            'over_land': over_land,
            'storm_df': storm_data
        }
    
    # Identify anomalous terminations
    anomalies = []
    for sid, data in terminations.items():
        # Criteria for anomalous termination:
        # 1. Not near boundary
        # 2. Not over land
        # 3. Final wind speed above 40 knots
        if (not data['near_boundary'] and 
            not data['over_land'] and 
            data['final_wind'] > 40.0):
            anomalies.append((sid, data['final_wind']))
    
    # Sort anomalies by final wind speed (most anomalous first)
    anomalies.sort(key=lambda x: x[1], reverse=True)
    
    # Limit number of anomalies to filter
    anomalies = anomalies[:max_anomalies]
    anomaly_ids = [a[0] for a in anomalies]
    
    logging.info(f"Identified {len(anomalies)} anomalous terminations to replace")
    
    # Only process if we found anomalies
    if anomalies:
        # Create copy to modify
        filtered_df = all_storms_df.copy()
        
        # Remove anomalous storms
        filtered_df = filtered_df[~filtered_df['SID'].isin(anomaly_ids)]
        
        # Process each anomalous storm to create better version
        replacement_dfs = []
        for sid, final_wind in anomalies:
            # Get original storm data
            orig_storm = terminations[sid]['storm_df']
            
            # Extract year and month
            year = orig_storm['YEAR'].iloc[0]
            month = orig_storm['MONTH'].iloc[0]
            
            # Generate a replacement storm with better decay properties
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    # Generate new storm
                    new_storm = generate_synthetic_storm(year, month, _df_positions, 
                                                       _historical_wind_speeds, 
                                                       set(filtered_df['SID'].unique()))
                    
                    # Verify it has proper termination
                    final_point = new_storm.iloc[-1]
                    proper_termination = (
                        final_point['WIND'] < 30.0 or
                        is_over_land(final_point['LON'], final_point['LAT']) or
                        abs(final_point['LON'] - PAR_BOUNDS[0][1]) < 1.0 or
                        abs(final_point['LON'] - PAR_BOUNDS[2][1]) < 1.0 or
                        abs(final_point['LAT'] - PAR_BOUNDS[0][0]) < 1.0 or
                        abs(final_point['LAT'] - PAR_BOUNDS[2][0]) < 1.0
                    )
                    
                    if proper_termination:
                        replacement_dfs.append(new_storm)
                        break
                        
                except Exception as e:
                    logging.warning(f"Failed to generate replacement for {sid}: {e}")
            
            # If all attempts failed, keep original
            if attempt == max_attempts - 1:
                replacement_dfs.append(orig_storm)
                logging.warning(f"Keeping original storm {sid} after {max_attempts} failed replacements")
        
        # Add replacement storms
        if replacement_dfs:
            filtered_df = pd.concat([filtered_df] + replacement_dfs, ignore_index=True)
        
        # Verify exact count is maintained
        if len(filtered_df['SID'].unique()) != required_count:
            logging.warning(f"Storm count mismatch: {len(filtered_df['SID'].unique())} vs required {required_count}")
            
            # Adjust count if needed by adding or removing storms
            current_count = len(filtered_df['SID'].unique())
            if current_count < required_count:
                # Generate additional storms
                missing = required_count - current_count
                logging.info(f"Generating {missing} additional storms to maintain count")
                
                # Use original generation code to create more storms
                extra_storms = []
                for _ in range(missing):
                    year = np.random.randint(int(start_year), int(end_year) + 1)
                    month = np.random.choice([7, 8, 9, 10, 6, 11])  # Focus on peak and shoulder months
                    new_storm = generate_synthetic_storm(year, month, _df_positions, 
                                                      _historical_wind_speeds, 
                                                      set(filtered_df['SID'].unique()))
                    extra_storms.append(new_storm)
                
                # Add the extra storms
                if extra_storms:
                    filtered_df = pd.concat([filtered_df] + extra_storms, ignore_index=True)
            
            elif current_count > required_count:
                # Remove excess storms (least concerning anomalies)
                excess = current_count - required_count
                logging.info(f"Removing {excess} excess storms to maintain count")
                
                # Find IDs to remove (avoid removing original good storms)
                all_ids = filtered_df['SID'].unique()
                replacement_ids = [df['SID'].iloc[0] for df in replacement_dfs 
                                  if df['SID'].iloc[0] in all_ids]
                
                # Sort by final wind (highest first)
                to_remove = []
                for rid in replacement_ids:
                    storm_data = filtered_df[filtered_df['SID'] == rid]
                    final_wind = storm_data['WIND'].iloc[-1]
                    to_remove.append((rid, final_wind))
                
                to_remove.sort(key=lambda x: x[1], reverse=True)
                remove_ids = [r[0] for r in to_remove[:excess]]
                
                # Remove the excess storms
                filtered_df = filtered_df[~filtered_df['SID'].isin(remove_ids)]
        
        return filtered_df
    else:
        # No anomalies found
        return all_storms_df

def initialize_dem(dem_path, backup_dem_path=None):
    """Properly initialize and validate DEM data for storm modeling."""
    global _dem_data, _dem_transform
    
    try:
        with rasterio.open(dem_path) as src:
            # Read the DEM data
            _dem_data = src.read(1)
            _dem_transform = src.transform
            
            # Apply moderate smoothing to reduce noise
            _dem_data = gaussian_filter(_dem_data, sigma=1.0)
            
            # Basic validation - check for reasonable elevation range
            valid_mask = ~np.isnan(_dem_data)
            min_elev = np.min(_dem_data[valid_mask])
            max_elev = np.max(_dem_data[valid_mask])
            
            logging.info(f"DEM loaded successfully. Shape: {_dem_data.shape}")
            logging.info(f"Elevation range: {min_elev}m to {max_elev}m")
            
            # Validate Philippines region has correct elevation data
            has_ph_mountains = np.any(_dem_data > 1000)
            if not has_ph_mountains:
                logging.warning("DEM validation failed: No high terrain detected in Philippines region")
                raise ValueError("DEM does not contain expected Philippine mountains")
                
            return True
            
    except Exception as e:
        logging.error(f"Failed to load primary DEM: {e}")
        
        # Try backup DEM if provided
        if backup_dem_path:
            try:
                logging.info(f"Attempting to load backup DEM from {backup_dem_path}")
                with rasterio.open(backup_dem_path) as src:
                    _dem_data = src.read(1)
                    _dem_transform = src.transform
                    _dem_data = gaussian_filter(_dem_data, sigma=1.0)
                    logging.info("Backup DEM loaded successfully")
                    return True
            except Exception as backup_e:
                logging.error(f"Failed to load backup DEM: {backup_e}")
        
        # Create minimal fallback DEM if all else fails
        logging.critical("Creating minimal fallback DEM - results will be approximate")
        create_fallback_philippines_dem()
        return False

def create_fallback_philippines_dem():
    """Load the older phl_dem.tif as fallback, or create minimal DEM if that fails too."""
    global _dem_data, _dem_transform
    
    # First try to load the older DEM file (phl_dem.tif)
    try:
        logging.info("Attempting to load fallback DEM: phl_dem.tif")
        with rasterio.open(DEM_FALLBACK_PATH) as src:
            _dem_data = src.read(1)
            _dem_transform = src.transform
            
            # Apply moderate smoothing to reduce noise
            _dem_data = gaussian_filter(_dem_data, sigma=1.0)
            
            # Basic validation
            valid_mask = ~np.isnan(_dem_data)
            if np.any(valid_mask):
                min_elev = np.min(_dem_data[valid_mask])
                max_elev = np.max(_dem_data[valid_mask])
                
                logging.info(f"Fallback DEM loaded successfully. Shape: {_dem_data.shape}")
                logging.info(f"Fallback DEM elevation range: {min_elev}m to {max_elev}m")
                return True
            else:
                raise ValueError("Fallback DEM contains no valid data")
                
    except Exception as e:
        logging.error(f"Failed to load fallback DEM (phl_dem.tif): {e}")
        logging.info("Creating minimal artificial DEM as last resort...")
    
    # If loading phl_dem.tif failed, create the artificial DEM as last resort
    try:
        # Create a simplified DEM covering Philippines region
        lat_range = np.linspace(4, 22, 180)  # ~10km resolution
        lon_range = np.linspace(115, 135, 200)
        
        # Create empty grid
        _dem_data = np.zeros((len(lat_range), len(lon_range)))
        
        # Add major mountain ranges (very simplified)
        for i, lat in enumerate(lat_range):
            for j, lon in enumerate(lon_range):
                # Cordillera (Luzon)
                if 16.5 <= lat <= 18.5 and 120.5 <= lon <= 121.5:
                    _dem_data[i, j] = 1500
                # Sierra Madre (Luzon)
                elif 14.5 <= lat <= 18.0 and 121.5 <= lon <= 122.2:
                    _dem_data[i, j] = 1200
                # Central Visayas
                elif 9.5 <= lat <= 11.5 and 123.0 <= lon <= 125.0:
                    _dem_data[i, j] = 800
                # Mindanao highlands
                elif 7.0 <= lat <= 9.0 and 124.5 <= lon <= 126.0:
                    _dem_data[i, j] = 1800
        
        # Apply smoothing
        _dem_data = gaussian_filter(_dem_data, sigma=2.0)
        
        # Create transform
        _dem_transform = rasterio.transform.from_bounds(
            lon_range[0], lat_range[0], lon_range[-1], lat_range[-1], 
            len(lon_range), len(lat_range)
        )
        
        logging.info("Created minimal artificial DEM as final fallback")
        return False  # Indicates we had to use artificial DEM
        
    except Exception as e:
        logging.critical(f"Failed to create even artificial DEM: {e}")
        # Set minimal defaults to prevent crashes
        _dem_data = np.zeros((100, 100))
        _dem_transform = rasterio.transform.from_bounds(115, 4, 135, 22, 100, 100)
        return False

def Check_if_landfall(lat, lon, lat1, lon0, land_mask=None):
    """
    Check if a point is over land.
    Simplified adaptation from Bloemendaal's code.
    """
    return is_over_land(lat, lon)
    
def synchronize_track_data(df, new_winds, caller="unknown"):
    """Enhanced synchronization with better error handling"""
    df_len = len(df)
    wind_len = len(new_winds)
    
    if df_len == wind_len:
        return df, new_winds
    
    logging.warning(f"Length mismatch in {caller}: df={df_len}, winds={wind_len}")
    
    # Use the shorter length and pad if necessary
    min_len = min(df_len, wind_len)
    
    if df_len > wind_len:
        # Pad winds with last valid value
        last_wind = new_winds[-1] if len(new_winds) > 0 else 22.0
        new_winds = np.pad(new_winds, (0, df_len - wind_len), 
                          mode='constant', constant_values=last_wind)
    else:
        # Truncate DataFrame
        df = df.iloc[:wind_len].copy()
    
    return df, new_winds
    
def fix_category_threshold_oscillations(wind_speeds):
    """
    Fix visual segmentation by smoothing only rapid oscillations across category thresholds.
    Preserves natural intensity variation while eliminating threshold flickering.
    """
    if len(wind_speeds) < 3:
        return wind_speeds
    
    # Category thresholds where segmentation is most visible
    thresholds = [34, 48, 64, 100, 106, 110, 115, 120]  # TS, STS, TY, STY
    
    for threshold in thresholds:
        for i in range(1, len(wind_speeds)-1):
            # Check for oscillation across this threshold
            prev_above = wind_speeds[i-1] >= threshold
            curr_above = wind_speeds[i] >= threshold
            next_above = wind_speeds[i+1] >= threshold
            
            # If current point is different from both neighbors across threshold
            if (prev_above == next_above) and (curr_above != prev_above):
                # Average to eliminate oscillation
                wind_speeds[i] = (wind_speeds[i-1] + wind_speeds[i+1]) / 2
                
    return validate_wind_speeds(wind_speeds, caller="fix_category_threshold_oscillations") 
    
def enforce_western_intensity_cap(storm_df):
    """
    Enforce strict intensity caps for storms west of the Philippines.
    """
    modified_df = storm_df.copy()
    
    for i, row in modified_df.iterrows():
        # Define regions where 106kt+ should not exist
        is_western_region = (
            (row['LON'] <= 120.0) or  # All areas 120E and below
            (10.0 <= row['LAT'] <= 15.0 and row['LON'] <= 122.53)  # 10N-15N window ≤122.53E
        )
        
        if is_western_region and row['WIND'] >= 106:
            # Hard cap at 102kt in these regions
            modified_df.loc[i, 'WIND'] = 102.0
            modified_df.loc[i, 'CATEGORY'] = 'Super Typhoon'
            
            if i % 10 == 0:  # Log occasionally to avoid spam
                logging.info(f"WESTERN CAP: Storm at {row['LON']:.1f}°E capped at 102kt")
    
    return modified_df
    
def apply_dem_based_decay(storm_df, dem_path):
    """
    Apply DEM-based wind decay to a tropical cyclone track.

    - Landfall detection via DEM (elev > 0.5 m).
    - Exponential decay V(t) = Vb + (V0 - Vb) * exp(-α t), with α from elevation and V0:contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}.
    - One-strike track termination after landfall (stop at first V<22 kt); two-strike rule over water.
    - Optional mountain disruption: extra decay if V0>=106 kt and elev>800 m.
    - Possible slight re-intensification if storm moves back over water.
    - Recompute central pressure (Atkinson–Holliday) and assign category per Philippine scale:contentReference[oaicite:7]{index=7}.
    - We use the exponential decay model of Kaplan & DeMaria (1995, aoml.noaa.gov)and standard terrain-induced friction effects (hurricanescience.org). 
    - Storm categories follow PAGASA definitions (Super Typhoon >=100 kt, Typhoon 64–99 kt, etc.), with any sub-22 kt point treated as a remnant low.
    """
    import numpy as np, logging, random
    
    # ADD THIS DEBUG LINE HERE (before land detection):
    print(f"DEM DECAY DEBUG: Processing storm with {len(storm_df)} points")
    print(f"BEFORE DEM DECAY: Max wind = {storm_df['WIND'].max():.1f}kt, Points = {len(storm_df)}")

    # ADD THIS DEBUG LINE:
    print(f"BEFORE DEM DECAY: Max wind = {storm_df['WIND'].max():.1f}kt, Points = {len(storm_df)}")
    
    # Copy original
    df = storm_df.copy()
    n = len(df)
    if n == 0:
        return df

    # SAFETY: Store original length for validation
    original_length = len(df)
    logging.debug(f"Starting DEM decay with {original_length} points")
    
    # Check if this is an extreme storm (max wind >= 106)
    max_storm_intensity = df['WIND'].max()
    is_extreme_storm = max_storm_intensity >= 106

    # For extreme storms, track eastern approach phase
    eastern_approach_phase = False
    if is_extreme_storm:
        # Check if storm is approaching from the east
        first_few_points = df.head(min(10, len(df)))
        if any(row['LON'] > 125 and row['LAT'] < 20 for _, row in first_few_points.iterrows()):
            eastern_approach_phase = True
            logging.info(f"EXTREME STORM: Detected eastern approach, preserving track continuity")
    
    # Initialize potential termination index
    potential_term_idx = None
    
    # Load DEM if not already done (initialize_dem should set global _dem_data/_transform)
    global _dem_data, _dem_transform
    try:
        if _dem_data is None:
            initialize_dem(dem_path)
    except NameError:
        logging.error("DEM init function not found")
        _dem_data = None
    if _dem_data is None:
        logging.error("DEM not available; returning original track")
        df['ELEVATION'] = [0.0]*n
        return df

    # Track positions and original winds
    lats = df['LAT'].values
    lons = df['LON'].values
    wind_orig = np.array(df['WIND'], dtype=float)

    # Compute elevation profile (DEM sample)
    elevation = []
    for lon, lat in zip(lons, lats):
        try:
            elev = get_elevation(lon, lat)
            # If DEM returns 0 but is_over_land says it's land, use fallback elevation
            if elev <= 0.0 and is_over_land(lon, lat):
                elev = 50.0  # Assign reasonable land elevation for decay calculation
        except Exception:
            # If DEM fails completely, check if it's land and assign elevation
            if is_over_land(lon, lat):
                elev = 50.0
            else:
                elev = 0.0
        elevation.append(elev)
    elevation = np.array(elevation)
    df['ELEVATION'] = elevation

    # DIAGNOSTIC CODE for Elevation
    land_points_count = np.sum(elevation > 0.5)
    logging.info(f"DEM sampling found {land_points_count} points over land (elev > 0.5m) out of {len(elevation)} total points")
    
    # Add at the beginning of apply_dem_based_decay
    land_detected = False
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if is_over_land(lon, lat):
            logging.info(f"LAND DETECTION: Point {i} at ({lat}, {lon}) is over land")
            land_detected = True
            break

    # ADD THE NEW DEBUGGING CODE HERE:
    logging.info(f"DEM DEBUG: Storm track bounds: lat {lats.min():.2f}-{lats.max():.2f}, lon {lons.min():.2f}-{lons.max():.2f}")
    logging.info(f"DEM DEBUG: DEM loaded: {_dem_data is not None}, shape: {_dem_data.shape if _dem_data is not None else 'None'}")
    logging.info(f"DEM DEBUG: Elevation sampling results: min={elevation.min():.2f}m, max={elevation.max():.2f}m")

    # Test a specific point that should be land
    test_lat, test_lon = 19.45, 122.42  # From your log
    test_elev = get_elevation(test_lon, test_lat)
    test_land = is_over_land(test_lon, test_lat)
    logging.info(f"DEM DEBUG: Test point ({test_lat}N, {test_lon}E) - get_elevation(): {test_elev:.2f}m, is_over_land(): {test_land}")

    # Check if coordinates are within DEM bounds
    if _dem_data is not None and _dem_transform is not None:
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            try:
                row, col = rasterio.transform.rowcol(_dem_transform, lon, lat)
                if 0 <= row < _dem_data.shape[0] and 0 <= col < _dem_data.shape[1]:
                    raw_elev = _dem_data[row, col]
                    if raw_elev > 0.5:  # Only log land points to avoid spam
                        logging.info(f"DEM DEBUG: Point {i} ({lat:.2f}N, {lon:.2f}E) -> row={row}, col={col}, raw_elev={raw_elev}")
                else:
                    logging.info(f"DEM DEBUG: Point {i} ({lat:.2f}N, {lon:.2f}E) -> OUTSIDE DEM BOUNDS (row={row}, col={col})")
            except Exception as e:
                logging.info(f"DEM DEBUG: Point {i} ({lat:.2f}N, {lon:.2f}E) -> TRANSFORM ERROR: {e}")

    if not land_detected:
        logging.warning(f"No land points detected in track - check DEM and land detection")
    
    # If any land points were found, log their positions
    if land_points_count > 0:
        land_indices = np.where(elevation > 0.5)[0]
        for idx in land_indices[:5]:  # Log first 5 land points
            logging.info(f"Land point at ({lons[idx]:.2f}E, {lats[idx]:.2f}N) with elevation {elevation[idx]:.2f}m")
            
    # Detect land points (elev > 0.5 m)
    over_land = (elevation > 0.5) & (elevation < 5000)  # Exclude extreme values (e.g., 1-000 m)
    
    # Log overland
    print(f"Land points found: {np.sum(over_land)} out of {len(storm_df)}")

    # Find first landfall index (first True after False or at 0)
    landfall_idx = None
    for i in range(n):
        if over_land[i] and (i == 0 or not over_land[i-1]):
            landfall_idx = i
            break

    # Prepare output wind array
    new_winds = wind_orig.copy()
    df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: init_winds")

    if landfall_idx is not None:
        V0 = float(wind_orig[landfall_idx])          # Wind at landfall
        V0_preserved = V0  # Default to original value
    
        # ENHANCED: Preserve STY intensity during initial eastern contact
        if V0 >= 106:
            # Calculate land duration for this specific point
            current_land_duration = 0
            for j in range(max(0, landfall_idx-5), landfall_idx+1):
                if j < len(over_land) and over_land[j]:
                    current_land_duration += 1
                else:
                    current_land_duration = 0
        
            if eastern_approach_phase and current_land_duration <= 3:  # First 9 hours over eastern land
                # NO REDUCTION - preserve full STY intensity during eastern approach
                logging.info(f"PRESERVING STY: {V0:.1f}kt maintained during eastern landfall (duration={current_land_duration})")
                V0_preserved = V0
            elif eastern_approach_phase and current_land_duration <= 6:  # Next 9-18 hours
                # Gentle reduction for eastern approach
                V0_preserved = max(102.0, V0 * 0.96)  # 4% reduction, minimum 102kt
                logging.info(f"GENTLE STY REDUCTION: {V0:.1f}kt -> {V0_preserved:.1f}kt (eastern approach)")
            elif current_land_duration <= 1:  # First 3 hours for non-eastern
                # Keep original intensity for brief contact
                logging.info(f"PRESERVING STY: {V0:.1f}kt at first land contact")
                V0_preserved = V0
            else:
                # Standard reduction for longer contact or non-eastern approach
                V0_preserved = min(V0, max(100.0, V0 * 0.92))
                logging.info(f"STANDARD STY REDUCTION: {V0:.1f}kt -> {V0_preserved:.1f}kt")
        
        # Use the preserved value for subsequent calculations
        V0 = V0_preserved
        
        Vb = 25.0 if V0 >= 90.0 else 22.0            # Background wind floor

        # Determine intensity factor (Kaplan–DeMaria style)
        if V0 >= 106:
            intensity_factor = 2.5 + 0.8*(V0 - 106)/20.0    # Significantly increased from 1.4/0.5
        elif V0 >= 100:
            intensity_factor = 2.0  # Increased from 1.5
        elif V0 >= 80:
            intensity_factor = 1.5  # Increased from 1.3
        elif V0 >= 64:
            intensity_factor = 1.2  # Slightly increased
        else:
            intensity_factor = 0.95
            
        # Find index where storm first re-enters water (after landfall)
        water_exit_idx = None
        for j in range(landfall_idx+1, n):
            if not over_land[j]:
                water_exit_idx = j
                break
        land_end = (water_exit_idx - 1) if water_exit_idx is not None else (n - 1)

        # Times since landfall (assuming 3-hourly steps)
        hours = 3.0 * (np.arange(landfall_idx, land_end+1) - landfall_idx)

        # Compute α for each land point
        elev_land = elevation[landfall_idx:land_end+1]
        alpha_base = 0.031
        
        # Add location-based adjustment for Philippines
        # Luzon and mountainous areas decay faster
        if any(elev_land > 600):  # Mountains in track
            intensity_factor *= 1.2  # Increase decay for mountainous terrain
            
        # Enhanced terrain factors that better capture Philippines topography
        terrain_factor = 1.0 + 1.8*(1.0 - np.exp(-elev_land/400.0))  # Steeper response
        roughness_factor = 1.0 + 0.7*(1.0 - np.exp(-elev_land/600.0))  # Increased effect
        
        alpha_vals = alpha_base * terrain_factor * intensity_factor * roughness_factor
        
        # Cap α by storm strength - slightly reduced for better persistence
        max_alpha = 0.25 if V0>=106 else 0.18 if V0>=100 else 0.15
        alpha_vals = np.clip(alpha_vals, 0.01, max_alpha)
        
        # Apply special case for water points - reduce decay rate for intense storms over water
        water_points = ~over_land[landfall_idx:land_end+1]  # Create boolean mask of water points
        if np.any(water_points):
            if V0 >= 100:
                alpha_vals[water_points] *= 0.3  # 70% reduction in decay rate over water for STY
            elif V0 >= 64:
                alpha_vals[water_points] *= 0.5  # 50% reduction for typhoons
        
        # Enhanced exponential decay on land segment
        decayed = Vb + (V0 - Vb)*np.exp(-alpha_vals * hours)
        decayed = np.minimum(decayed, V0)   # cannot exceed initial

        # Log it
        extreme_preserved = np.sum(decayed >= 106)
        if extreme_preserved > 0:
            logging.warning(f"PRESERVED {extreme_preserved} points >=106kt after DEM decay")

        # Mountain disruption for extreme storms
        if V0 >= 106:
            mask_high = (elev_land > 600.0)
            if mask_high.any():
                mf = 1.2 + (elev_land[mask_high] - 600.0)/600.0
                decayed[mask_high] = Vb + (decayed[mask_high] - Vb)/mf

        # Update new winds for land points
        new_winds[landfall_idx:land_end+1] = decayed
        df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: landfall_decay")
        
        # Check for potential one-strike termination, but don't apply it yet
        potential_term_idx = None
        lows = np.where(decayed < 22.0)[0]
        if lows.size > 0:
            rel = lows[0]
            potential_term_idx = landfall_idx + rel
            logging.info(f"First sub-threshold point detected at position {potential_term_idx}, but continuing to check for re-intensification")

        # If storm moves back to water, allow LIMITED re-intensification
        if water_exit_idx is not None and water_exit_idx < n:
            # Mark that this storm has touched land - will be used for final check
            has_touched_land = True
            
            # Check if we're still over land - if so, NO re-intensification
            current_lat = df.iloc[water_exit_idx]['LAT']
            current_lon = df.iloc[water_exit_idx]['LON']
            still_over_land = is_over_land(current_lon, current_lat)
            
            if still_over_land:
                logging.info(f"Skipping re-intensification: still over land at ({current_lat:.2f}N, {current_lon:.2f}E)")
            else:
                baseline = new_winds[water_exit_idx]
                
                # Check location relative to Philippines
                is_west_of_philippines = (current_lon < 120.0 and 5.0 <= current_lat <= 20.0)
                is_south_china_sea = (current_lon < 118.0)
                
                # HISTORICAL CONSTRAINT: No STY (106+ kt) west of Philippines
                if is_west_of_philippines or is_south_china_sea:
                    # Hard cap at 95kt west of Philippines - no exceptions
                    max_allowed_west = 95.0
                    if new_winds[water_exit_idx] > max_allowed_west:
                        new_winds[water_exit_idx] = max_allowed_west
                        logging.info(f"WEST PHILIPPINES CAP: Reduced to {max_allowed_west}kt (lon={current_lon:.1f})")
                else:
                    # East of Philippines - allow normal re-intensification
                    if V0 >= 106 and new_winds[water_exit_idx] < V0 * 0.9:
                        reintensify_probability = 0.25  # 25% chance for former STY
                        if random.random() < reintensify_probability:
                            # Can recover towards original intensity
                            max_recovery = min(V0 * 0.95, 130.0)  # Up to 95% of original, max 130kt
                            boost_amount = (max_recovery - new_winds[water_exit_idx]) * 0.4
                            new_winds[water_exit_idx] += boost_amount
                            logging.info(f"EAST RE-INTENSIFICATION: +{boost_amount:.1f}kt to {new_winds[water_exit_idx]:.1f}kt")
    
            # After potential re-intensification, check for any subsequent landfall
            in_water = True
            second_landfall_idx = None
    
            # Search for second landfall after water exit
            for i in range(water_exit_idx+1, n):
                if i < len(over_land) and over_land[i] and in_water:  # Add bounds check
                # Found the start of second landfall
                    second_landfall_idx = i
                    in_water = False
                    logging.info(f"Detected second landfall at position {i}")
                    break
    
            # If a second landfall is detected, apply decay again
            if second_landfall_idx is not None:
                # Calculate the length of the second landfall segment
                second_segment_length = n - second_landfall_idx
    
                # Only proceed if there are points to process
                if second_segment_length > 0:
                    # Get wind speed at second landfall (becomes the new V0)
                    second_V0 = float(new_winds[second_landfall_idx])
                    second_Vb = 20.0  # Background wind floor for second landfall
        
                    # Calculate times since second landfall (assuming 3-hourly steps)
                    second_hours = 3.0 * np.arange(second_segment_length)
        
                    # Calculate decay parameters for each land point in second landfall
                    second_land_segment = slice(second_landfall_idx, n)
                    if second_landfall_idx < len(elevation):  # Add bounds check
                        elev_second_landfall = elevation[second_land_segment]
            
                        # Ensure we have valid elevation data
                        if len(elev_second_landfall) > 0:
                            # Decay parameter calculation for second landfall
                            alpha_base = 0.031  # Same as first landfall
                
                            # Apply full elevation-based formula as with first landfall
                            terrain_factor = 1.0 + 1.5*(1.0 - np.exp(-elev_second_landfall/500.0))
                            roughness_factor = 1.0 + 0.5*(1.0 - np.exp(-elev_second_landfall/1000.0))
                
                            # Use the same intensity factor logic as the first landfall
                            if second_V0 >= 106:
                                intensity_factor = 1.8 + 0.7*(second_V0 - 106)/20.0
                            elif second_V0 >= 100:
                                intensity_factor = 1.5
                            elif second_V0 >= 80:
                                intensity_factor = 1.3
                            elif second_V0 >= 64:
                                intensity_factor = 1.1
                            else:
                                intensity_factor = 0.95
                
                            # Calculate alpha values with complete formula
                            alpha_vals = alpha_base * terrain_factor * intensity_factor * roughness_factor
                
                            # Cap alpha by storm strength (using higher values for stronger decay)
                            max_alpha = 0.5 if second_V0>=106 else (0.35 if second_V0>=100 else 0.25)
                            alpha_vals = np.clip(alpha_vals, 0.01, max_alpha)
                
                            # Exponential decay with proper alpha values
                            second_decay = second_Vb + (second_V0 - second_Vb) * np.exp(-alpha_vals * second_hours)
                
                            # Mountain disruption for extreme storms during second landfall
                            if second_V0 >= 106:
                                # Find high elevation points in the second landfall segment
                                mask_high = (elev_second_landfall > 800.0)
                                if mask_high.any():
                                    # Calculate mountain factor based on excess height above 800m threshold
                                    mf = 1.0 + (elev_second_landfall[mask_high] - 800.0)/1000.0
                                    # Apply additional weakening to points over high terrain
                                    second_decay[mask_high] = second_Vb + (second_decay[mask_high] - second_Vb)/mf
                                    logging.info(f"Applied mountain disruption to second landfall - {np.sum(mask_high)} high elevation points")
                
                            # Make sure the arrays have compatible shapes
                            if len(second_decay) == second_segment_length:
                                # Apply the decay to the remainder of the track
                                new_winds[second_landfall_idx:n] = second_decay
                                logging.info(f"Applied enhanced decay to second landfall: {second_V0:.1f}kt -> {new_winds[-1]:.1f}kt")
                                df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: second_landfall_decay")
                            else:
                                logging.warning(f"Shape mismatch in second landfall decay: {len(second_decay)} vs {second_segment_length}")
                        else:
                            logging.warning(f"No elevation data for second landfall segment")
                    else:
                        logging.warning(f"Second landfall index {second_landfall_idx} out of bounds for elevation data")
        else:
            # No landfall: whole track is over water
            water_exit_idx = 0

    # Ensure arrays match DataFrame length before updating
    if len(new_winds) == len(df):
        df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: pre_update")
        df['WIND'] = new_winds
        df['PRES'] = calculate_pressure_from_wind(new_winds, env_pressure=1010.0, lat=df['LAT'])
        if 'RMW' in df.columns:
            df['RMW'] = [calculate_rmw(w, la) for w, la in zip(new_winds, df['LAT'])]
    else:
        logging.error(f"Length mismatch in DEM decay: new_winds={len(new_winds)}, df={len(df)}. Attempting to fix...")
        
        # FIX: Ensure we work with the correct subset
        if len(new_winds) > len(df):
            # Trim new_winds to match df length
            new_winds = new_winds[:len(df)]
            logging.info(f"Trimmed new_winds to match DataFrame length: {len(new_winds)}")
            df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: fix_mismatch")
        else:
            # This shouldn't happen, but if it does, we need to handle it
            # Pad with the last valid value or 22.0
            last_valid = new_winds[-1] if len(new_winds) > 0 else 22.0
            padding_needed = len(df) - len(new_winds)
            new_winds = np.append(new_winds, [last_valid] * padding_needed)
            logging.warning(f"Padded new_winds with {padding_needed} values of {last_valid}")
        
        # Now update with matched lengths
        df['WIND'] = new_winds
        df['PRES'] = calculate_pressure_from_wind(new_winds, env_pressure=1010.0, lat=df['LAT'])
        if 'RMW' in df.columns:
            df['RMW'] = [calculate_rmw(w, la) for w, la in zip(new_winds, df['LAT'])]
        
    df, new_winds = synchronize_track_data(df, new_winds, "apply_dem_based_decay: final_sync")
    
    # Update categories for all points
    cats = []
    for w in new_winds:
        if w >= 100:
            cats.append("Super Typhoon")
        elif w >= 64:
            cats.append("Typhoon")
        elif w >= 48:
            cats.append("Severe Tropical Storm")
        elif w >= 34:
            cats.append("Tropical Storm")
        elif w >= 22:
            cats.append("Tropical Depression")
        else:
            cats.append("Remnant Low")
    df['CATEGORY'] = cats

    # Smooth category transitions to reduce segmentation
    prev_cat = None
    smooth_window = 2  # Points on each side

    for i in range(len(new_winds)):
        # Get current category
        if new_winds[i] >= 100:
            curr_cat = "STY"
        elif new_winds[i] >= 64:
            curr_cat = "TY"
        elif new_winds[i] >= 48:
            curr_cat = "STS"
        elif new_winds[i] >= 34:
            curr_cat = "TS"
        else:
            curr_cat = "TD"
    
        # Smooth transitions when moving up a category
        if prev_cat and curr_cat != prev_cat:
            # If going up, look ahead to see if it's maintained
            if (curr_cat == "STY" and prev_cat != "STY") or \
               (curr_cat == "TY" and prev_cat not in ["TY", "STY"]):
            
                # Check if intensity is maintained for at least smooth_window points
                maintained = True
                for j in range(i+1, min(i+smooth_window+1, len(new_winds))):
                    if (curr_cat == "STY" and new_winds[j] < 100) or \
                       (curr_cat == "TY" and new_winds[j] < 64):
                        maintained = False
                        break
            
                # If not maintained, adjust to avoid segmentation
                if not maintained:
                    if curr_cat == "STY":
                        # Allow brief STY episodes - don't cap immediately
                        if i > 0 and new_winds[i-1] >= 106:  # If previous was extreme
                            new_winds[i] = max(105.0, new_winds[i])  # Keep at least 105kt
                        else:
                            new_winds[i] = 99.0  # Standard cap
                    elif curr_cat == "TY":
                        new_winds[i] = 63.0  # Just below TY threshold
    
        prev_cat = curr_cat
    
    # After smoothing, recompute categories with smoothed winds
    cats = []
    for w in new_winds:
        if w >= 100:
            cats.append("Super Typhoon")
        elif w >= 64:
            cats.append("Typhoon")
        elif w >= 48:
            cats.append("Severe Tropical Storm")
        elif w >= 34:
            cats.append("Tropical Storm")
        elif w >= 22:
            cats.append("Tropical Depression")
        else:
            cats.append("Remnant Low")
    df['CATEGORY'] = cats

    # Update winds in DataFrame after smoothing
    df['WIND'] = new_winds

    # Final safety check before returning, final validation 
    if len(df) == 0:
        logging.error("DataFrame became empty during DEM decay processing!")
        return storm_df  # Return original if processing failed
    
    # Two-strike termination (over open water)
    if landfall_idx is None:
        start = 0
    else:
        start = water_exit_idx if water_exit_idx is not None else n
    consec = 0
    term_idx = None
    for i in range(start, len(new_winds)):
        if new_winds[i] < 22.0:
            consec += 1
            if consec >= 2:
                term_idx = i - 1
                break
        else:
            consec = 0

    if term_idx is not None:
        df = df.iloc[:term_idx+2]  # include two sub-22 kt points
        # CRITICAL: Also truncate new_winds to match!
        new_winds = new_winds[:term_idx+2]
        n = len(df)  # Update n to reflect new length

    # Now apply termination rules based on the complete wind profile
    # First, check the previously identified potential termination point
    if potential_term_idx is not None:
        # Check if the storm actually re-intensified after this point
        if all(new_winds[potential_term_idx:] < 22.0):
            # No re-intensification occurred, apply one-strike termination
            logging.info(f"Applying delayed one-strike termination at position {potential_term_idx}")
            df = df.iloc[:potential_term_idx+1]
            new_winds = new_winds[:potential_term_idx+1]
            n = len(df) # Update n
        else:
            # Storm re-intensified, check if it met the more lenient two-strike rule
            consec_low = 0
            final_term_idx = None
        
            for i in range(potential_term_idx, len(new_winds)):
                if new_winds[i] < 22.0:
                    consec_low += 1
                    if consec_low >= 2:  # Two-strike rule
                        final_term_idx = i - 1
                        break
                else:
                    consec_low = 0
        
            if final_term_idx is not None:
                logging.info(f"Applying two-strike termination after re-intensification at position {final_term_idx}")
                df = df.iloc[:final_term_idx+2]  # Include two sub-22kt points
                new_winds = new_winds[:final_term_idx+2]
                n = len(df) # Update n

    # Update all columns based on the final wind values (with safety check)
    if len(new_winds) != len(df):
        logging.error(
            f"CRITICAL: Wind column length {len(new_winds)} "
            f"doesn't match DataFrame length {len(df)}"
        )
        # Synchronize lengths
        min_length = min(len(new_winds), len(df))
        new_winds = new_winds[:min_length]
        df = df.iloc[:min_length]
        logging.info(f"ERROR FIXED: Synchronized to length: {min_length}")
       
    df['WIND'] = new_winds
    df['PRES'] = calculate_pressure_from_wind(new_winds, env_pressure=1010.0, lat=df['LAT'])
    if 'RMW' in df.columns:
        df['RMW'] = [calculate_rmw(w, la) for w, la in zip(new_winds, df['LAT'])]

    # Recompute categories for all points
    cats = []
    for w in new_winds:
        if w >= 100:
            cats.append("Super Typhoon")
        elif w >= 64:
            cats.append("Typhoon")
        elif w >= 48:
            cats.append("Severe Tropical Storm")
        elif w >= 34:
            cats.append("Tropical Storm")
        elif w >= 22:
            cats.append("Tropical Depression")
        else:
            cats.append("Remnant Low")
    df['CATEGORY'] = cats
    
    # Final safety check before returning, final validation 
    if len(df) == 0:
        logging.error("DataFrame became empty during DEM decay processing!")
        return storm_df  # Return original if processing failed
    
    # Ensure all arrays are the correct length
    #expected_length = len(df)
    
    # Verify WIND column
    #if 'WIND' in df.columns and len(df['WIND']) != expected_length:
    #    logging.error(f"WIND column length mismatch: {len(df['WIND'])} vs expected {expected_length}")
        # Fix by using the validated new_winds array
    #    if len(new_winds) == expected_length:
    #        df['WIND'] = new_winds
    #    else:
    #        logging.error("Cannot fix WIND column length mismatch")
    
    # Verify everything is synchronized
    if len(df['WIND']) != len(df):
        logging.error(f"CRITICAL: Wind column length {len(df['WIND'])} doesn't match DataFrame length {len(df)}")
        return storm_df  # Return original to avoid corruption

    logging.debug(f"DEM decay completed successfully: {len(df)} points")

    # ADD THIS DEBUG LINE BEFORE RETURN:
    print(f"AFTER DEM DECAY: Max wind = {df['WIND'].max():.1f}kt, Points = {len(df)}")
    
    return df
    
def compensate_for_land_losses(storms_df, target_sty_percent=5.49):
    """Boost STY formation over water to compensate for land capping."""
    
    current_sty = (len(storms_df[storms_df['WIND'] >= 100]) / len(storms_df)) * 100
    
    if current_sty < target_sty_percent:
        # Find strong TY points over water and promote to STY
        water_mask = storms_df.apply(lambda row: not is_over_land(row['LON'], row['LAT']), axis=1)
        
        # ADD DEBUG HERE (right after water_mask):
        total_points = len(storms_df)
        water_points = water_mask.sum()
        land_points = total_points - water_points
        print(f"COMPENSATION DEBUG: {water_points} water points, {land_points} land points out of {total_points}")
        
        strong_ty_mask = (storms_df['WIND'] >= 95) & (storms_df['WIND'] < 100) & water_mask
        
        # ADD MORE DEBUG HERE (after strong_ty_mask):
        eligible_points = strong_ty_mask.sum()
        print(f"COMPENSATION DEBUG: {eligible_points} eligible TY points over water for promotion")
        
        deficit_points = int(len(storms_df) * (target_sty_percent - current_sty) / 100)
        promote_indices = storms_df[strong_ty_mask].nlargest(deficit_points, 'WIND').index
        
        storms_df.loc[promote_indices, 'WIND'] = 100.0
        print(f"Promoted {len(promote_indices)} over-water TY points to STY")
        
        # Create some extreme winds (124+ kt) 
        extreme_target_percent = 0.015  # 1.5% should be 124+ kt
        current_extreme = (len(storms_df[storms_df['WIND'] >= 124]) / len(storms_df)) * 100

        if current_extreme < extreme_target_percent:
            # Find some STY points over water east of Philippines to boost to extreme
            east_water_mask = (storms_df['LON'] >= 125.0) & water_mask
            extreme_candidates = (storms_df['WIND'] >= 100) & (storms_df['WIND'] < 124) & east_water_mask
    
            if extreme_candidates.any():
                extreme_deficit = int(len(storms_df) * (extreme_target_percent - current_extreme) / 100)
                extreme_promote_indices = storms_df[extreme_candidates].nlargest(extreme_deficit, 'WIND').index
        
                # Boost to 124-126 kt range
                for idx in extreme_promote_indices:
                    storms_df.loc[idx, 'WIND'] = np.random.uniform(124, 126)
        
                print(f"Promoted {len(extreme_promote_indices)} STY points to extreme (124+ kt)")
    
    return storms_df
    
def apply_boundary_decay(storm_df):
    """Apply realistic weakening to storms approaching domain boundaries."""
    df = storm_df.copy()
    n = len(df)
    if n < 3:
        return df
    
    # Define boundary buffer zones
    boundary_buffer = 3.0  # from 2 to 3 degrees from domain edge
    west_boundary = PAR_BOUNDS[2][1] + boundary_buffer  # 115°E + buffer
    east_boundary = PAR_BOUNDS[0][1] - boundary_buffer  # 135°E - buffer
    south_boundary = PAR_BOUNDS[2][0] + boundary_buffer  # 5°N + buffer
    north_boundary = PAR_BOUNDS[0][0] - boundary_buffer  # 25°N - buffer
    
    # Original wind values
    winds = df['WIND'].values.copy()
    lats = df['LAT'].values
    lons = df['LON'].values
    
    # Check if storm approaches boundaries
    approaching_boundary = False
    boundary_approach_idx = None
    
    for i in range(n-1):
        # Calculate distance to boundaries
        dist_to_west = lons[i] - PAR_BOUNDS[2][1]
        dist_to_east = PAR_BOUNDS[0][1] - lons[i]
        dist_to_south = lats[i] - PAR_BOUNDS[2][0]
        dist_to_north = PAR_BOUNDS[0][0] - lats[i]
        
        # Check if within buffer zone and moving toward boundary
        if ((dist_to_west < boundary_buffer and lons[i] > lons[i+1]) or
            (dist_to_east < boundary_buffer and lons[i] < lons[i+1]) or
            (dist_to_south < boundary_buffer and lats[i] > lats[i+1]) or
            (dist_to_north < boundary_buffer and lats[i] < lats[i+1])):
            approaching_boundary = True
            boundary_approach_idx = i
            break
    
    # Apply gradual decay if approaching boundary
    if approaching_boundary and boundary_approach_idx is not None:
        # Get peak intensity before approaching boundary
        peak_wind = np.max(winds[:boundary_approach_idx+1])
        min_wind = 20.0  # Terminal wind speed
        
        # Calculate remaining steps
        remaining_steps = n - boundary_approach_idx
        
        # Apply exponential decay toward boundary
        for j in range(boundary_approach_idx+1, n):
            # Normalized position in remaining track (0.0 to 1.0)
            pos = (j - boundary_approach_idx) / remaining_steps
            
            # Exponential decay formula
            decay_factor = np.exp(-4.0 * pos)
            
            # Set decayed wind (cannot increase)
            winds[j] = min(winds[j], min_wind + (peak_wind - min_wind) * decay_factor)
    
    # Update DataFrame with decayed winds
    df['WIND'] = winds
    
    # Update derived fields
    df['PRES'] = calculate_pressure_from_wind(winds, env_pressure=1010.0, lat=df['LAT'])
    if 'RMW' in df.columns:
        df['RMW'] = [calculate_rmw(w, la) for w, la in zip(winds, df['LAT'])]
    
    # Update categories
    categories = []
    for w in winds:
        if w >= 100:
            categories.append("Super Typhoon")
        elif w >= 64:
            categories.append("Typhoon")
        elif w >= 48:
            categories.append("Severe Tropical Storm")
        elif w >= 34:
            categories.append("Tropical Storm")
        elif w >= 22:
            categories.append("Tropical Depression")
        else:
            categories.append("Remnant Low")
    df['CATEGORY'] = categories
    
    return df

def adjust_with_coriolis(lat, lon, month):
    """
    Apply Coriolis-based adjustments based on beta-drift effect.
    Data-driven parameters from historical track analysis.
    """
    import numpy as np
    
    # Earth's angular velocity (radians per second)
    omega = 7.2921e-5
    
    # Earth radius in meters
    earth_radius = 6371000
    
    # Determine season from month for configuration
    if month in [12, 1, 2]:
        season = 'winter'
        params = {'factor_north': 0.09999999999999964, 'factor_west': 0.30000000000001137, 'max_factor': 0.7810249675906611}
    elif month in [3, 4, 5]:
        season = 'spring'
        params = {'factor_north': 0.1999999999999993, 'factor_west': 0.20000000000000284, 'max_factor': 0.7071067811865643}
    elif month in [6, 7, 8, 9]:
        season = 'summer'
        params = {'factor_north': 0.1999999999999993, 'factor_west': -0.1, 'max_factor': 0.7810249675906611}
    else:
        season = 'fall'
        params = {'factor_north': 0.10000000000000142, 'factor_west': -0.1, 'max_factor': 0.8000000000000114}
    
    # Coriolis parameter: f = 2Ω*sin(φ)
    f = 2 * omega * np.sin(np.radians(lat))
    
    # Beta parameter: β = df/dy = (2Ω*cos(φ))/R
    beta = 2 * omega * np.cos(np.radians(lat)) / earth_radius
    
    # Convert to degrees per 6 hours
    meters_per_degree_lat = 111000  # approximate
    seconds_in_6hrs = 6 * 3600
    
    # Calculate beta drift factor with handling for equatorial regions
    if abs(lat) < 0.1:
        beta_drift_factor = params['max_factor'] * 0.5  # Reduced at equator
    else:
        beta_drift_factor = min(params['max_factor'], beta / (abs(f) + 1e-10))
    
    # Calculate drift components with seasonal adjustments
    northward_drift = params['factor_north'] * beta_drift_factor * seconds_in_6hrs / meters_per_degree_lat
    westward_drift = params['factor_west'] * beta_drift_factor * seconds_in_6hrs / meters_per_degree_lat
    
    # Apply latitude band adjustments
    north_values = [0.10000000000000009, 0.10000000000000142, 0.1999999999999993, 0.20000000000000284]
    west_values = [0.4000000000000057, 0.4000000000000057, 0.29999999999999716, 0.10000000000000853]
    band_bounds = [5, 10, 15, 20, 25]  # Boundaries of your latitude bands

    # Find which band(s) the current latitude falls in/near
    if lat < band_bounds[0]:
        # Below the first band - use first band values
        northward_drift = max(northward_drift, north_values[0])
        westward_drift += west_values[0]
    elif lat >= band_bounds[-1]:
        # Above the last band - use last band values
        northward_drift = max(northward_drift, north_values[-1])
        westward_drift += west_values[-1]
    else:
        # Find the neighboring bands for interpolation
        for i in range(len(band_bounds)-1):
            lower_bound = band_bounds[i]
            upper_bound = band_bounds[i+1]
        
            if lower_bound <= lat < upper_bound:
                # Calculate position within this band (0.0 to 1.0)
                band_pos = (lat - lower_bound) / (upper_bound - lower_bound)
            
                # Add slight randomization to position to avoid artificial accumulation
                # Small enough to maintain statistical properties, large enough to break patterns
                band_pos += np.random.normal(0, 0.08)
                band_pos = np.clip(band_pos, 0.0, 1.0)
            
                # Apply smooth transition using cosine interpolation (smoother than linear)
                # This preserves the exact values at band centers while creating smooth transitions
                weight = 0.5 - 0.5 * np.cos(band_pos * np.pi)
            
                # Interpolate between band values
                north_val = north_values[i] * (1 - weight) + north_values[i+1] * weight
                west_val = west_values[i] * (1 - weight) + west_values[i+1] * weight
            
                # Apply the interpolated values
                northward_drift = max(northward_drift, north_val)
                westward_drift += west_val
            
                break

    # Apply longitude region adjustments
    if 115 <= lon < 122:  # western region
        northward_drift = max(northward_drift, 0.10000000000000142)
        westward_drift += 0.29999999999999716
    elif 122 <= lon < 130:  # central region
        northward_drift = max(northward_drift, 0.1999999999999993)
        westward_drift += 0.29999999999999716
    elif 130 <= lon < 135:  # eastern region
        northward_drift = max(northward_drift, 0.1999999999999993)
        westward_drift += 0.30000000000001137
    
    if lat < 10.0:
        # Get data-driven probability for this latitude
        lat_bin = track_model.bin_latitude(lat)
        data_prob = track_model.southward_prob.get(lat_bin, 0.3)
    
        # Calculate minimum northward component based on data
        # This allows some southward movement but with latitude dependence
        data_factor = 1.0 - (2.0 * data_prob)  # Convert probability to factor
    
        # Blend with your existing approach
        model_min = 0.05 * (1.0 - (lat / 10.0))
        min_northward = 0.6 * model_min + 0.4 * (0.03 * data_factor)
    
        # Apply the minimum but less strictly
        if northward_drift < min_northward:
            # Instead of hard minimum, make it probabilistic
            if np.random.random() < 0.8:  # 80% chance of correction
                northward_drift = min_northward * np.random.uniform(0.9, 1.1)
            
        if lat < 7.0:  # Even stronger northward bias for very low latitudes
            northward_drift = max(northward_drift, 0.1)
    
    # Add seasonal variations
    if month in [7, 8, 9]:  # Peak typhoon season
        northward_drift *= 1.1  # Slightly stronger northward component
    
    # Enhanced adjustments for northern latitudes
    if lat > 14:  # North of Central Philippines
        latitude_factor = min(1.0, (lat - 14) / 8)  # 0 at 14°N, 1.0 at 22°N
        eastward_adjustment = min(0.1, 0.15 * latitude_factor)  # Cap adjustment
        westward_drift += eastward_adjustment  # Less westward = more eastward
        northward_drift *= (1.0 + 0.3 * latitude_factor)  # Add extra northward component at higher latitudes
    
    # Eastern Pacific pattern adjustment
    if lon > 130:
        eastward_factor = min(1.0, (lon - 130) / 10)
        westward_drift += 0.12 * eastward_factor  # More eastward motion
    
    # Normalize drift values to prevent excessive movement
    total_drift = abs(northward_drift) + abs(westward_drift)
    if total_drift > 0.8:
        scale_factor = 0.8 / total_drift
        northward_drift *= scale_factor
        westward_drift *= scale_factor
    
    return northward_drift, westward_drift

def generate_genesis_points(df_historical, year, month, count, category=None):
    """
    Generate realistic TC genesis points within PAR using historical patterns
    that vary by month and storm category, ensuring they are over water.
    
    Args:
        df_historical: DataFrame with historical TC positions
        year: Target year
        month: Target month (1-12)
        count: Number of genesis points to generate
        category: Storm category ('TD', 'TS', 'STS', 'TY', 'STY') or None
        
    Returns:
        List of tuples (lat, lon) for genesis positions
    """
    # Ensure df_historical is properly formatted
    if isinstance(df_historical, pd.DataFrame) and not df_historical.empty:
        # Check for unexpected dimensions
        try:
            if hasattr(df_historical, 'ndim') and df_historical.ndim > 2:
                logging.warning(f"Historical data has unexpected dimensions: {df_historical.shape}. Attempting to flatten.")
                # Convert to a standard 2D DataFrame
                df_historical = pd.DataFrame(df_historical.values.reshape(-1, df_historical.shape[-1]), 
                                            columns=df_historical.columns)
        except Exception as e:
            logging.error(f"Could not fix dimensions for historical data: {e}")
            # Fall back to default genesis points
            if category == 'STY':
                return [(np.random.uniform(12, 18), np.random.uniform(132, 138))]
            elif category == 'TY':
                return [(np.random.uniform(10, 16), np.random.uniform(128, 135))]
            elif category in ['STS', 'TS']:
                return [(np.random.uniform(8, 14), np.random.uniform(125, 132))]
            else:  # TD
                return [(np.random.uniform(6, 12), np.random.uniform(122, 130))]
                
    # Enhanced month-specific and category-specific genesis regions based on historical patterns
    # Format: {month: {category: [(center_lat, center_lon, lat_std, lon_std, weight), ...], ...}, ...}
    # Higher weight value means more likely to generate genesis in that region
    genesis_regions = {
        # January: Eastern / southeastern pattern - shifted eastward
        1: {
            'TD': [(6.0, 132.0, 1.0, 1.5, 0.5), (8.0, 129.0, 1.2, 1.2, 0.5)],
            'TS': [(7.0, 134.0, 1.0, 1.5, 0.6), (10.0, 132.0, 1.0, 1.2, 0.4)],
            'STS': [(7.5, 133.0, 1.0, 1.5, 0.6), (9.0, 135.0, 1.0, 1.0, 0.4)],
            'TY': [(8.0, 136.0, 1.2, 1.5, 0.6), (10.0, 134.0, 1.0, 1.2, 0.4)],
            'STY': [(9.0, 137.0, 1.0, 1.0, 0.8), (11.0, 135.0, 1.0, 1.0, 0.2)]
        },
        # February: Eastern Genesis - shifted farther east
        2: {
            'TD': [(5.0, 133.0, 1.0, 1.5, 0.6), (8.0, 130.0, 1.0, 1.0, 0.4)],
            'TS': [(6.0, 135.0, 1.0, 1.5, 0.7), (9.0, 133.0, 1.0, 1.0, 0.3)],
            'STS': [(7.0, 136.0, 1.0, 1.2, 0.6), (10.0, 135.0, 1.0, 1.0, 0.4)],
            'TY': [(8.0, 137.0, 1.2, 1.0, 0.7), (11.0, 136.0, 1.0, 1.0, 0.3)],
            'STY': [(9.0, 138.0, 1.0, 1.0, 0.8), (12.0, 137.0, 1.0, 1.0, 0.2)]
        },
        # March: Central and Eastern Philippines - focus more eastward
        3: {
            'TD': [(6.0, 129.0, 1.5, 1.5, 0.3), (7.0, 134.0, 1.5, 1.5, 0.7)],
            'TS': [(7.0, 132.0, 1.2, 1.2, 0.4), (9.0, 137.0, 1.0, 1.5, 0.6)],
            'STS': [(7.5, 134.0, 1.0, 1.2, 0.5), (10.0, 137.0, 1.0, 1.0, 0.5)],
            'TY': [(8.0, 136.0, 1.0, 1.5, 0.6), (10.5, 138.0, 1.0, 1.0, 0.4)],
            'STY': [(9.0, 138.0, 1.0, 1.0, 0.7), (12.0, 140.0, 1.0, 1.0, 0.3)]
        },
        # April: Eastern Central Philippines - eastward focus
        4: {
            'TD': [(7.0, 130.0, 1.5, 1.5, 0.5), (6.0, 135.0, 1.5, 1.5, 0.5)],
            'TS': [(8.0, 134.0, 1.2, 1.5, 0.6), (10.0, 137.0, 1.0, 1.5, 0.4)],
            'STS': [(9.0, 135.0, 1.0, 1.2, 0.6), (11.0, 138.0, 1.0, 1.0, 0.4)],
            'TY': [(9.5, 136.0, 1.0, 1.5, 0.7), (12.0, 138.0, 1.0, 1.0, 0.3)],
            'STY': [(10.0, 137.0, 1.0, 1.0, 0.7), (13.0, 140.0, 1.0, 1.0, 0.3)]
        },
        # May: Transition month - Eastern and Northern pattern
        5: {
            'TD': [(7.0, 129.0, 1.5, 1.5, 0.4), (8.0, 134.0, 1.5, 1.5, 0.6)],
            'TS': [(9.0, 132.0, 1.2, 1.5, 0.5), (11.0, 136.0, 1.0, 1.5, 0.5)],
            'STS': [(10.0, 134.0, 1.0, 1.5, 0.5), (12.0, 138.0, 1.0, 1.5, 0.5)],
            'TY': [(10.5, 136.0, 1.0, 1.5, 0.6), (13.0, 138.0, 1.0, 1.0, 0.4)],
            'STY': [(11.0, 138.0, 1.0, 1.0, 0.7), (14.0, 140.0, 1.0, 1.0, 0.3)]
        },
        # June: Eastern and Northern pattern
        6: {
            'TD': [(9.0, 130.0, 1.5, 1.5, 0.3), (11.0, 134.0, 1.5, 1.5, 0.7)],
            'TS': [(10.0, 132.0, 1.2, 1.5, 0.4), (13.0, 135.0, 1.0, 1.5, 0.6)],
            'STS': [(11.0, 134.0, 1.0, 1.5, 0.4), (14.0, 137.0, 1.0, 1.5, 0.6)],
            'TY': [(12.0, 136.0, 1.0, 1.5, 0.4), (15.0, 138.0, 1.0, 1.0, 0.6)],
            'STY': [(13.0, 138.0, 1.0, 1.0, 0.5), (16.0, 139.0, 1.0, 1.0, 0.5)]
        },
        # July: Strong Northern pattern - extended eastward
        7: {
            'TD': [(10.0, 129.0, 1.5, 1.5, 0.3), (14.0, 133.0, 1.5, 1.5, 0.7)],
            'TS': [(12.0, 131.0, 1.2, 1.5, 0.3), (16.0, 134.0, 1.0, 1.5, 0.7)],
            'STS': [(13.0, 132.0, 1.0, 1.5, 0.3), (17.0, 135.0, 1.0, 1.5, 0.7)],
            'TY': [(14.0, 134.0, 1.0, 1.5, 0.3), (18.0, 136.0, 1.0, 1.0, 0.7)],
            'STY': [(15.0, 135.0, 1.0, 1.0, 0.3), (19.0, 137.0, 1.0, 1.0, 0.7)]
        },
        # August: Strong Northern pattern - extended eastward
        8: {
            'TD': [(11.0, 128.0, 1.5, 1.5, 0.3), (16.0, 132.0, 1.5, 1.5, 0.7)],
            'TS': [(13.0, 130.0, 1.2, 1.5, 0.3), (17.0, 134.0, 1.0, 1.5, 0.7)],
            'STS': [(14.0, 132.0, 1.0, 1.5, 0.3), (18.0, 135.0, 1.0, 1.5, 0.7)],
            'TY': [(15.0, 133.0, 1.0, 1.5, 0.3), (19.0, 136.0, 1.0, 1.0, 0.7)],
            'STY': [(16.0, 134.0, 1.0, 1.0, 0.3), (20.0, 137.0, 1.0, 1.0, 0.7)]
        },
        # September: Very strong Northern pattern - eastward focus
        9: {
            'TD': [(12.0, 127.0, 1.5, 1.5, 0.2), (17.0, 131.0, 1.5, 1.5, 0.8)],
            'TS': [(14.0, 129.0, 1.2, 1.5, 0.2), (18.0, 133.0, 1.0, 1.5, 0.8)],
            'STS': [(15.0, 131.0, 1.0, 1.5, 0.2), (19.0, 134.0, 1.0, 1.5, 0.8)],
            'TY': [(16.0, 132.0, 1.0, 1.5, 0.2), (20.0, 135.0, 1.0, 1.0, 0.8)],
            'STY': [(17.0, 133.0, 1.0, 1.0, 0.2), (21.0, 136.0, 1.0, 1.0, 0.8)]
        },
        # October: Northern and Eastern pattern - more eastward
        10: {
            'TD': [(10.0, 129.0, 1.5, 1.5, 0.3), (15.0, 132.0, 1.5, 1.5, 0.7)],
            'TS': [(12.0, 131.0, 1.2, 1.5, 0.3), (16.0, 134.0, 1.0, 1.5, 0.7)],
            'STS': [(13.0, 133.0, 1.0, 1.5, 0.3), (17.0, 135.0, 1.0, 1.5, 0.7)],
            'TY': [(14.0, 134.0, 1.0, 1.5, 0.3), (18.0, 136.0, 1.0, 1.0, 0.7)],
            'STY': [(15.0, 135.0, 1.0, 1.0, 0.3), (19.0, 137.0, 1.0, 1.0, 0.7)]
        },
        # November: Shifting back to Central and Eastern - more eastward
        11: {
            'TD': [(8.0, 130.0, 1.5, 1.5, 0.4), (12.0, 133.0, 1.5, 1.5, 0.6)],
            'TS': [(9.0, 132.0, 1.2, 1.5, 0.4), (13.0, 135.0, 1.0, 1.5, 0.6)],
            'STS': [(10.0, 134.0, 1.0, 1.5, 0.4), (14.0, 137.0, 1.0, 1.5, 0.6)],
            'TY': [(11.0, 136.0, 1.0, 1.5, 0.4), (15.0, 138.0, 1.0, 1.0, 0.6)],
            'STY': [(12.0, 137.0, 1.0, 1.0, 0.4), (16.0, 139.0, 1.0, 1.0, 0.6)]
        },
        # December: Eastern pattern - more eastward
        12: {
            'TD': [(7.0, 131.0, 1.5, 1.5, 0.4), (9.0, 134.0, 1.5, 1.5, 0.6)],
            'TS': [(8.0, 133.0, 1.2, 1.5, 0.5), (10.0, 136.0, 1.0, 1.5, 0.5)],
            'STS': [(9.0, 135.0, 1.0, 1.5, 0.5), (11.0, 138.0, 1.0, 1.5, 0.5)],
            'TY': [(10.0, 136.0, 1.0, 1.5, 0.5), (12.0, 139.0, 1.0, 1.0, 0.5)],
            'STY': [(11.0, 137.0, 1.0, 1.0, 0.5), (13.0, 140.0, 1.0, 1.0, 0.5)]
        }
    }
    
    # If no category is specified, determine a default category distribution
    # This allows the function to work with existing code that doesn't specify category
    if category is None:
        if month in _monthly_category_dist:
            # Use the monthly category distribution
            category_probs = _monthly_category_dist[month]
            categories = ['TD', 'TS', 'STS', 'TY', 'STY']
            # Select a random category based on the distribution
            r = random.random() * 100  # Random percentage
            cumulative_prob = 0
            for i, prob in enumerate(category_probs):
                cumulative_prob += prob
                if r <= cumulative_prob:
                    category = categories[i]
                    break
            # Fallback if somehow no category is selected
            if category is None:
                category = 'TD'
        else:
            # Default to TD if no monthly distribution
            category = 'TD'
    
    # Select the genesis regions for this month and category
    try:
        month_regions = genesis_regions.get(month, genesis_regions[7])  # Default to July if month not found
    
        # Enhanced eastern genesis for extreme storms
        if category == 'STY':
            # Force more eastern genesis points for super typhoons
            eastern_regions = [
                (15.0, 138.0, 1.0, 1.0, 0.4),  # Far eastern
                (12.0, 140.0, 1.5, 1.0, 0.3),  # Very far eastern  
                (18.0, 136.0, 1.0, 1.5, 0.3)   # Northern eastern
            ]
        
            # Blend with existing regions but favor eastern
            original_regions = month_regions.get(category, month_regions['TD'])
        
            # 70% eastern, 30% original for STY
            combined_regions = eastern_regions + [(r[0], r[1], r[2], r[3], r[4]*0.3) for r in original_regions]
            category_regions = combined_regions
        
            logging.info(f"STY GENESIS: Using enhanced eastern genesis points")
        
        else:
            category_regions = month_regions.get(category, month_regions['TD'])  # Default to TD if category not found
            
    except (KeyError, TypeError):
        # Fallback to default regions if there's any error
        logging.warning(f"Using default genesis regions for month={month}, category={category}")
        # Use more eastern defaults
        if category == 'STY':
            category_regions = [(13.0, 138.0, 1.5, 1.5, 1.0)]
        elif category == 'TY':
            category_regions = [(12.0, 136.0, 1.5, 1.5, 1.0)]
        elif category in ['STS', 'TS']:
            category_regions = [(10.0, 134.0, 1.5, 1.5, 1.0)]
        else:  # TD
            category_regions = [(8.0, 132.0, 1.5, 1.5, 1.0)]
    
    # Generate the requested number of genesis points
    sampled_points = []
    max_attempts = 50  # Maximum attempts to generate valid points
    total_attempts = 0
    
    while len(sampled_points) < count and total_attempts < max_attempts * count:
        # Randomly select a region based on weights
        weights = [region[4] for region in category_regions]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]  # Normalize weights
        
        selected_region = random.choices(category_regions, weights=weights, k=1)[0]
        center_lat, center_lon, lat_std, lon_std, _ = selected_region
        
        # Generate a point with normal distribution around the center
        perturbed_lat = np.random.normal(center_lat, lat_std)
        perturbed_lon = np.random.normal(center_lon, lon_std)
        
        # Ensure the point is within the buffered PAR region
        if not is_inside_par(perturbed_lat, perturbed_lon, use_buffer=True):
            total_attempts += 1
            continue
            
        # CRITICAL ENHANCEMENT: Check if the point is over land
        if is_over_land(perturbed_lon, perturbed_lat):
            total_attempts += 1
            continue
        
        # If point is valid, add to the list
        sampled_points.append((perturbed_lat, perturbed_lon))
        total_attempts += 1
    
    # If we couldn't generate enough valid points, add points in known ocean regions
    if len(sampled_points) < count:
        logging.warning(f"Could only generate {len(sampled_points)} valid genesis points out of {count} requested")
        # Fill in missing points with safe ocean locations based on category
        missing_count = count - len(sampled_points)
        
        # Safe regions for genesis by category (lat, lon, std)
        safe_regions = {
            'STY': [(17.0, 137.0, 1.0), (13.0, 139.0, 1.0), (10.0, 138.0, 1.0)],
            'TY': [(15.0, 136.0, 1.0), (12.0, 138.0, 1.0), (9.0, 137.0, 1.0)],
            'STS': [(14.0, 135.0, 1.0), (10.0, 136.0, 1.0), (7.0, 134.0, 1.0)],
            'TS': [(12.0, 134.0, 1.0), (8.0, 135.0, 1.0), (6.0, 132.0, 1.0)],
            'TD': [(10.0, 133.0, 1.0), (7.0, 132.0, 1.0), (5.0, 130.0, 1.0)]
        }
        
        safe_list = safe_regions.get(category, safe_regions['TD'])
        
        for _ in range(missing_count):
            # Pick a random safe region
            safe_lat, safe_lon, safe_std = random.choice(safe_list)
            
            # Generate point with some variance
            valid_point = False
            for _ in range(10):  # Try a few times for each missing point
                new_lat = np.random.normal(safe_lat, safe_std)
                new_lon = np.random.normal(safe_lon, safe_std)
                
                # Check if the point is valid (in PAR and not over land)
                if is_inside_par(new_lat, new_lon, use_buffer=True) and not is_over_land(new_lon, new_lat):
                    sampled_points.append((new_lat, new_lon))
                    valid_point = True
                    break
            
            # If all attempts failed, just use the center point directly
            if not valid_point:
                sampled_points.append((safe_lat, safe_lon))
    
    return sampled_points

def generate_track_dynamics(initial_lat, initial_lon, start_month, duration=48, track_coef=None, 
                           track_coef_mod=None, category=None):
    """
    Generate storm track dynamics using derived James-Mason model coefficients with
    enhanced natural variability to mimic real-world tropical cyclone behavior.
    
    Args:
        initial_lat: Starting latitude
        initial_lon: Starting longitude
        start_month: Month of genesis (1-12)
        duration: Track duration in timesteps (each is 3 hours)
        track_coef: Optional track coefficients to override defaults
        track_coef_mod: Optional modifiers for track behavior
        category: Storm category ('TD', 'TS', 'STS', 'TY', 'STY')
        
    Returns:
        Tuple of (track_lats, track_lons)
    """
    # Initialize tracking arrays
    track_lats = [initial_lat]
    track_lons = [initial_lon]
    
    # Use provided coefficients or load the default ones
    coef_dict = track_coef if track_coef is not None else load_track_coefficients()
    
    # Get available latitude bands
    lat_bands = [k for k in coef_dict.keys() if isinstance(k, (int, float))]
    lat_bands = sorted(lat_bands)
    
    # Default modifiers if not provided
    if track_coef_mod is None:
        track_coef_mod = {
            'recurve_bias': 1.0,
            'northward_bias': 1.0,
            'eastward_bias': 1.0,
            'speed_factor': 1.0
        }
    
    # Apply default modifiers based on category if not already set and category is provided
    if category and track_coef_mod.get('recurve_bias', None) is None:
        if category == 'STY':
            track_coef_mod = {
                'recurve_bias': 1.4,    # Higher chance of recurvature
                'northward_bias': 1.2,  # Stronger northward component
                'eastward_bias': 1.3,   # Stronger eastward component at high latitudes
                'speed_factor': 1.15    # Faster movement
            }
        elif category == 'TY':
            track_coef_mod = {
                'recurve_bias': 1.2,
                'northward_bias': 1.1,
                'eastward_bias': 1.1,
                'speed_factor': 1.1
            }
        elif category == 'STS':
            track_coef_mod = {
                'recurve_bias': 1.0,
                'northward_bias': 1.0,
                'eastward_bias': 0.9,
                'speed_factor': 1.0
            }
        elif category == 'TS':
            track_coef_mod = {
                'recurve_bias': 0.8,
                'northward_bias': 0.9,
                'eastward_bias': 0.8,
                'speed_factor': 0.95
            }
        else:  # TD
            track_coef_mod = {
                'recurve_bias': 0.6,
                'northward_bias': 0.8,
                'eastward_bias': 0.7,
                'speed_factor': 0.9
            }
    # Enhanced eastern approach handling for extreme storms
        if category == 'STY' and initial_lon > 130:
            logging.info(f"STY TRACK: Enhancing eastern approach dynamics")
        
            # Modify track coefficients for better eastern approach
            track_coef_mod['westward_persistence'] = 1.3  # Stronger westward component
            track_coef_mod['recurve_delay'] = 1.5          # Delay recurvature
            track_coef_mod['land_attraction'] = 1.2        # Slight attraction to landmasses
    
    # Extract modifiers with defaults
    recurve_bias = track_coef_mod.get('recurve_bias', 1.0)
    northward_bias = track_coef_mod.get('northward_bias', 1.0)
    eastward_bias = track_coef_mod.get('eastward_bias', 1.0)
    speed_factor = track_coef_mod.get('speed_factor', 1.0)
    
    # Seasonal adjustments based on month
    seasonal_adj = {
        # [lat_factor, lon_factor]
        'winter': [0.9, 1.2],   # Dec-Feb: Less northward, more westward
        'spring': [1.0, 1.0],   # Mar-May: Average
        'summer': [1.1, 0.9],   # Jun-Sep: More northward, less westward
        'fall':   [1.0, 1.1]    # Oct-Nov: Average northward, more westward
    }
    
    # Determine season from month
    if start_month in [12, 1, 2]:
        season = 'winter'
    elif start_month in [3, 4, 5]:
        season = 'spring'
    elif start_month in [6, 7, 8, 9]:
        season = 'summer'
    else:
        season = 'fall'
    
    # Get seasonal adjustments
    lat_season_factor, lon_season_factor = seasonal_adj[season]
    
    # Apply category-specific modifiers to seasonal factors
    lat_season_factor *= northward_bias
    lon_season_factor *= (2.0 - eastward_bias)  # Invert for westward movement
    
    # ENHANCEMENT 1: Create storm-specific coefficient variations
    # This ensures each storm behaves differently while still following general patterns
    storm_coef_variation = {
        'a0_factor': np.random.normal(1.0, 0.15),  # Longitude base term
        'a1_factor': np.random.normal(1.0, 0.10),  # Longitude autocorrelation
        'b0_factor': np.random.normal(1.0, 0.15),  # Latitude base term
        'b1_factor': np.random.normal(1.0, 0.10),  # Latitude autocorrelation
        'b2_factor': np.random.normal(1.0, 0.15),  # Latitude curvature
        # Random noise amplification - significantly larger
        'random_scale': np.random.uniform(1.5, 3.0) * speed_factor
    }
    
    # Initialize with statistics from genesis if available
    if 'genesis' in coef_dict:
        # Use month-specific values if available
        if 'genesis_monthly' in coef_dict and start_month in coef_dict['genesis_monthly']:
            g_stats = coef_dict['genesis_monthly'][start_month]
            # Increase standard deviation for initial movement
            dlat0 = np.random.normal(g_stats['initial_dlat_mean'], g_stats['initial_dlat_std'] * 1.5)
            dlon0 = np.random.normal(g_stats['initial_dlon_mean'], g_stats['initial_dlon_std'] * 1.5)
        else:
            # Use overall values
            g_stats = coef_dict['genesis']
            dlat0 = np.random.normal(g_stats['initial_dlat_mean'], g_stats['initial_dlat_std'] * 1.5)
            dlon0 = np.random.normal(g_stats['initial_dlon_mean'], g_stats['initial_dlon_std'] * 1.5)
    else:
        # Default initial movement if no genesis data - increase variability
        dlat0 = np.random.normal(0.1, 0.25)  # Increased standard deviation
        dlon0 = np.random.normal(-0.3, 0.35)  # Increased standard deviation
    
    # Apply eastern location bias for initial step
    if initial_lon > 130:
        dlon0 -= 0.2 * min(1.0, (initial_lon - 130) / 5) * eastward_bias
    
    # Apply category-specific initial motion adjustments
    if category == 'STY' or category == 'TY':
        # Stronger storms tend to have more northward/northeastward initial motion
        dlat0 = max(dlat0, 0.1) * northward_bias  # Ensure some northward component
        dlon0 *= (2.0 - eastward_bias)  # Adjust westward component
    elif category == 'TD':
        # Tropical depressions often have more westward initial motion
        dlat0 *= northward_bias
        dlon0 = min(dlon0, -0.1) * (2.0 - eastward_bias)  # Ensure some westward component
    
    # Apply latitudinal adjustments to initial motion
    if initial_lat < 10:
        # Low-latitude systems tend to move more westward initially
        dlon0 = min(dlon0 * 1.2, -0.2)
    elif initial_lat > 18:
        # High-latitude systems tend to have more poleward/eastward component
        dlat0 = max(dlat0 * 1.2, 0.15)
        dlon0 *= 0.8  # Less westward
    
    # ENHANCEMENT 2: Add initial jitter to prevent parallel tracks
    dlat0 += np.random.normal(0, 0.1)
    dlon0 += np.random.normal(0, 0.15)
    
    # Track if recurvature has occurred
    has_recurved = False
    recurve_progress = 0.0
    
    # ENHANCEMENT 3: Randomize recurvature threshold
    # This prevents all storms from recurving at the same latitude
    # Make it category-dependent
    base_recurve_lat = 17.53
    if category == 'STY':
        # Super typhoons often recurve at higher latitudes
        recurve_adj = 2.0
    elif category == 'TY':
        recurve_adj = 1.0
    elif category == 'STS':
        recurve_adj = 0.0
    elif category == 'TS':
        recurve_adj = -1.0
    else:  # TD
        recurve_adj = -2.0
    
    # Apply seasonal adjustment to recurvature latitude
    if season == 'summer':
        season_recurve_adj = 1.0  # Higher recurvature latitude in summer
    elif season == 'winter':
        season_recurve_adj = -2.0  # Lower recurvature latitude in winter
    else:
        season_recurve_adj = 0.0
    
    # Calculate final threshold with random component
    RECURVATURE_LATITUDE = base_recurve_lat + recurve_adj + season_recurve_adj + np.random.normal(0, 1.5)
    
    # Modify recurvature threshold based on the recurve_bias parameter
    RECURVATURE_LATITUDE = max(12.0, min(22.0, RECURVATURE_LATITUDE / recurve_bias))
    
    # ENHANCEMENT 4: Add wobble parameters
    # This creates small oscillations in the track
    wobble = {
        'active': True,
        'lat_amp': np.random.uniform(0.02, 0.12),  # Amplitude of latitude wobble
        'lon_amp': np.random.uniform(0.03, 0.15),  # Amplitude of longitude wobble
        'lat_period': np.random.uniform(4, 10),    # Period of latitude wobble
        'lon_period': np.random.uniform(4, 10),    # Period of longitude wobble
        'lat_phase': np.random.uniform(0, 2*np.pi),# Phase of latitude wobble
        'lon_phase': np.random.uniform(0, 2*np.pi) # Phase of longitude wobble
    }
    
    # ENHANCEMENT 5: Add random direction shift events
    # This simulates interaction with other weather systems
    direction_shift = {
        'active': False,
        'next_check': np.random.randint(6, 12),  # Steps until next potential shift
        'chance': 0.15,                         # Base chance of shift occurring
        'duration': 0,                          # Current shift duration
        'lat_bias': 0,                          # Current shift bias for latitude
        'lon_bias': 0                           # Current shift bias for longitude
    }
    
    # Generate the track
    for i in range(1, duration):
        # Current position
        current_lat = track_lats[-1]
        current_lon = track_lons[-1]
        
        # Find appropriate latitude band for current position
        current_band = lat_bands[0]
        for band in lat_bands:
            if current_lat >= band:
                current_band = band
        
        # Get coefficients for current position
        a0, a1, b0, b1, b2, lat_mu, lat_std, lon_mu, lon_std = coef_dict[current_band]
        
        # Apply seasonal adjustments
        a0 *= lon_season_factor
        
        # Apply storm-specific coefficient variations
        a0 *= storm_coef_variation['a0_factor']
        a1 *= storm_coef_variation['a1_factor']
        b0 *= storm_coef_variation['b0_factor'] * northward_bias
        b1 *= storm_coef_variation['b1_factor']
        b2 *= storm_coef_variation['b2_factor']
        
        # Apply category-specific adjustments
        if category in ['STY', 'TY'] and current_lat > 15:
            # Stronger northward and eastward components at higher latitudes
            b0 *= northward_bias
            a0 *= (2.0 - eastward_bias * 1.2)  # Reduce westward component
        
        # Handle direction shift events (representing interaction with other weather systems)
        if direction_shift['active']:
            # Continue current shift
            direction_shift['duration'] -= 1
            if direction_shift['duration'] <= 0:
                # End of shift
                direction_shift['active'] = False
                direction_shift['next_check'] = np.random.randint(6, 12)
        else:
            # Check if new shift starts
            direction_shift['next_check'] -= 1
            if direction_shift['next_check'] <= 0:
                # Chance of shift starting (higher near coast and higher latitudes)
                shift_chance = direction_shift['chance']
                if current_lat > 15 or (current_lon < 125 and current_lat > 10):
                    shift_chance *= 1.5
                
                if np.random.random() < shift_chance:
                    # Start a new shift
                    direction_shift['active'] = True
                    direction_shift['duration'] = np.random.randint(2, 6)
                    # Random shift direction
                    direction_shift['lat_bias'] = np.random.normal(0, 0.25)
                    direction_shift['lon_bias'] = np.random.normal(0, 0.3)
                
                # Reset next check
                direction_shift['next_check'] = np.random.randint(6, 12)
        
        # Check for recurvature (more likely at higher latitudes)
        if current_lat > RECURVATURE_LATITUDE and i > duration // 4:
            # Calculate recurvature probability (increases with latitude)
            recurve_prob = 0.01 * min(1.0, (current_lat - RECURVATURE_LATITUDE))
            
            # Apply category-specific recurvature probability adjustments
            recurve_prob *= recurve_bias
            
            # Increase probability based on track stage
            if i > duration // 2:
                recurve_prob *= 2.0
                
            # Check if recurvature starts
            if not has_recurved and np.random.random() < recurve_prob:
                has_recurved = True
                recurve_progress = 0.0
        
        # Apply James-Mason model for forward movement
        # Latitude component: dlat1 = b0 + b1*dlat0 + b2/lat
        dlat1 = b0 + b1 * dlat0
        if current_lat > 0:  # Prevent division by zero
            dlat1 += b2 / current_lat
        
        # Longitude component: dlon1 = a0 + a1*dlon0
        dlon1 = a0 + a1 * dlon0
        
        # Add wobble effect (small oscillations)
        if wobble['active']:
            wobble_lat = wobble['lat_amp'] * np.sin(2*np.pi * i / wobble['lat_period'] + wobble['lat_phase'])
            wobble_lon = wobble['lon_amp'] * np.sin(2*np.pi * i / wobble['lon_period'] + wobble['lon_phase'])
            dlat1 += wobble_lat
            dlon1 += wobble_lon
        
        # Add random components (epsilon terms) with category-specific variability
        rand_scale = storm_coef_variation['random_scale']
        if category == 'STY':
            rand_scale *= 0.8  # Super typhoons tend to have more predictable tracks
        elif category == 'TD':
            rand_scale *= 1.2  # TDs tend to be more erratic
            
        dlat1 += np.random.normal(lat_mu, lat_std * rand_scale)
        dlon1 += np.random.normal(lon_mu, lon_std * rand_scale)
        
        # Add active direction shift biases
        if direction_shift['active']:
            dlat1 += direction_shift['lat_bias']
            dlon1 += direction_shift['lon_bias']
        
        # Apply gradual recurvature effect if activated
        if has_recurved:
            # Increase recurvature progress with variability
            recurve_step = 0.1 * np.random.uniform(0.8, 1.2) * recurve_bias
            recurve_progress = min(1.0, recurve_progress + recurve_step)
            
            # Gradually shift from westward to northward/northeastward with variability
            eastward_factor = min(1.0, (current_lat - RECURVATURE_LATITUDE) / 7.0)
            eastward_factor *= np.random.uniform(0.9, 1.1) * eastward_bias
            
            # Adjust longitude movement (reduce westward/add eastward component)
            dlon1 += recurve_progress * eastward_factor * 0.3 * np.random.uniform(0.9, 1.1)
            
            # Enhance northward movement during recurvature with variability
            dlat1 += recurve_progress * 0.1 * np.random.uniform(0.9, 1.1) * northward_bias
        
        # Track stage adjustments with variability
        if i < duration // 3:  # Early stage
            dlon1 -= 0.15 * np.random.uniform(0.8, 1.2) * (2.0 - eastward_bias)
            dlat1 *= 0.8 * np.random.uniform(0.9, 1.1) * northward_bias
        
        # Eastern PAR adjustment with variability
        if current_lon > 125:
            dlon1 -= 0.2 * min(1.0, (current_lon - 125) / 10) * np.random.uniform(0.8, 1.2) * (2.0 - eastward_bias)
        # Enhanced eastern approach logic for STY
        if category == 'STY' and current_lon > 125 and i < duration // 2:
            # Bias toward continued westward movement during eastern approach
            dlon1 -= 0.15 * np.random.uniform(0.9, 1.1)  # Additional westward bias
            
            # Resist early recurvature during eastern approach
            if current_lat > RECURVATURE_LATITUDE * 0.8:
                dlat1 *= 0.85  # Reduce northward component
        
        # Get data-driven movement suggestions
        data_dlat, data_dlon = track_model.sample_movement(current_lat, current_lon, dlat0, dlon0)

        # Blend data-driven movement with model-generated movement
        # Higher weight to data-driven component at critical latitudes
        if 8.0 <= current_lat <= 12.0:
            # Near the 10°N region, rely more on data-driven pattern
            blend_factor = 0.7  # 70% weight to data-driven model
        else:
            # Elsewhere, give less weight to data-driven model
            blend_factor = 0.4  # 40% weight to data-driven model

        # Apply blending - this combines your mathematical model with the data-driven model
        dlat1_hybrid = blend_factor * data_dlat + (1-blend_factor) * dlat1
        dlon1_hybrid = blend_factor * data_dlon + (1-blend_factor) * dlon1

        # Soften Southward Movement Constraints
        if current_lat < 10 and dlat1_hybrid < 0:
            # Calculate probability from both approaches
            model_prob = (current_lat / 10.0) * 0.8  # Your existing approach
            data_prob = track_model.southward_prob.get(track_model.bin_latitude(current_lat), 0.3)
    
            # Weighted average of both probabilities
            southward_prob = 0.6 * data_prob + 0.4 * model_prob
    
            if np.random.random() > southward_prob:
            # Convert to northward but less severely
                dlat1_hybrid = abs(dlat1_hybrid) * 0.2 * np.random.uniform(0.6, 0.9)
            else:
                # Allow southward but with data-driven limits
                max_southward = -0.08 * (current_lat / 5.0)  # Gentler limit
                dlat1_hybrid = max(dlat1_hybrid, max_southward)

        # Replace original dlat1/dlon1 with the hybrid versions
        dlat1 = dlat1_hybrid
        dlon1 = dlon1_hybrid
        
        # ENHANCEMENT 6: Add occasional larger random deviations (5% chance)
        # This simulates sudden changes due to unexpected atmospheric conditions
        if np.random.random() < 0.05:
            dlat1 += np.random.normal(0, 0.2)
            dlon1 += np.random.normal(0, 0.25)
        
        # ADD THE NEW CODE HERE - Specific randomization near 10°N to break up artificial patterns
        if 9.0 < current_lat < 11.0:  # Near the problematic zone
            # More randomness at exactly 10°N, less as you move away
            random_factor = 1.0 - abs(current_lat - 10.0)
            dlat1 += np.random.normal(0, 0.04 * random_factor)
            dlon1 += np.random.normal(0, 0.04 * random_factor)
        
        # Apply maximum movement constraints with category-specific adjustments
        max_dlat = 1.2 * speed_factor
        max_dlon = 1.8 * speed_factor
        
        dlat1 = np.clip(dlat1, -max_dlat, max_dlat)
        dlon1 = np.clip(dlon1, -max_dlon, max_dlon * 0.8)  # Limit eastward movement more
        
        # Calculate next position
        next_lat = current_lat + dlat1
        next_lon = current_lon + dlon1
        
        # Update for next iteration
        dlat0 = dlat1
        dlon0 = dlon1
        
        # Add to track
        track_lats.append(next_lat)
        track_lons.append(next_lon)
        
        # Check if the storm has exited PAR
        if not is_inside_par(next_lat, next_lon, use_buffer=True):
            if i > duration // 2:  # Only check after first half of duration
                break
    
    return np.array(track_lats), np.array(track_lons)

def generate_synthetic_storm(year, month, historical_positions=None, historical_winds=None, used_ids=None):
    """
    Generate a single synthetic tropical cyclone with realistic category-specific behavior.
    
    Args:
        year: Target year for the storm
        month: Month of genesis (1-12)
        historical_positions: Historical position data for calibration
        historical_winds: Historical wind speed data for calibration
        used_ids: Set of already used storm IDs to avoid duplicates
        
    Returns:
        DataFrame with synthetic storm track
    """
    # Determine storm category first based on monthly distribution
    if month in _monthly_category_dist:
        category_probs = _monthly_category_dist[month]
        r = random.random() * 100  # Random percentage
        cumulative_prob = 0
        category_idx = 0
        for i, prob in enumerate(category_probs):
            cumulative_prob += prob
            if r <= cumulative_prob:
                category_idx = i
                break
        
        # Map index to category string (0=TD, 1=TS, 2=STS, 3=TY, 4=STY)
        category_names = ['TD', 'TS', 'STS', 'TY', 'STY']
        storm_category = category_names[category_idx]
    else:
        # Default to TD if no distribution available
        storm_category = 'TD'
    
    # Generate genesis point based on category and month
    if historical_positions is not None and not historical_positions.empty:
        # Sample genesis from historical data with category-specific perturbation
        genesis_points = generate_genesis_points(
            historical_positions, year, month, 1, category=storm_category
        )
        if genesis_points:
            initial_lat, initial_lon = genesis_points[0]
        else:
            # Fallback to default genesis points
            logging.warning(f"No valid genesis points for {year}-{month}, category {storm_category}, using defaults")
            # Use category-specific defaults
            if storm_category == 'STY':
                initial_lat = np.random.uniform(12, 18)
                initial_lon = np.random.uniform(132, 138)
            elif storm_category == 'TY':
                initial_lat = np.random.uniform(10, 16)
                initial_lon = np.random.uniform(128, 135)
            elif storm_category in ['STS', 'TS']:
                initial_lat = np.random.uniform(8, 14)
                initial_lon = np.random.uniform(125, 132)
            else:  # TD
                initial_lat = np.random.uniform(6, 12)
                initial_lon = np.random.uniform(122, 130)
        
        # ENHANCEMENT: Bias towards eastern genesis for better PAR entry
        # Add 50% chance to shift eastward for all storms
        if np.random.random() < 0.5:
            initial_lon = min(140, initial_lon + np.random.uniform(2, 5))
            logging.debug(f"Applied eastward genesis bias: lon shifted to {initial_lon}")
    
    else:
        # Default genesis point generation with category specificity
        if storm_category == 'STY':
            initial_lat = np.random.uniform(12, 18)
            initial_lon = np.random.uniform(132, 138)
        elif storm_category == 'TY':
            initial_lat = np.random.uniform(10, 16)
            initial_lon = np.random.uniform(128, 135)
        elif storm_category in ['STS', 'TS']:
            initial_lat = np.random.uniform(8, 14)
            initial_lon = np.random.uniform(125, 132)
        else:  # TD
            initial_lat = np.random.uniform(6, 12)
            initial_lon = np.random.uniform(122, 130)
    
    # Generate track dynamics with category-specific settings
    track_coef_mod = None
    
    # Category-specific track coefficient modifiers
    if storm_category == 'STY':
        # Super Typhoons tend to recurve more and move faster
        track_coef_mod = {
            'recurve_bias': 1.4,    # Higher chance of recurvature
            'northward_bias': 1.2,  # Stronger northward component
            'eastward_bias': 1.3,   # Stronger eastward component at high latitudes
            'speed_factor': 1.15    # Faster movement
        }
    elif storm_category == 'TY':
        # Typhoons also tend to recurve but not as strongly as STYs
        track_coef_mod = {
            'recurve_bias': 1.2,
            'northward_bias': 1.1,
            'eastward_bias': 1.1,
            'speed_factor': 1.1
        }
    elif storm_category == 'STS':
        # Severe Tropical Storms - more varied tracks
        track_coef_mod = {
            'recurve_bias': 1.0,
            'northward_bias': 1.0,
            'eastward_bias': 0.9,
            'speed_factor': 1.0
        }
    elif storm_category == 'TS':
        # Tropical Storms - more westward movement
        track_coef_mod = {
            'recurve_bias': 0.8,
            'northward_bias': 0.9,
            'eastward_bias': 0.8,
            'speed_factor': 0.95
        }
    else:  # TD
        # Tropical Depressions - slower, more westward/WNW movement
        track_coef_mod = {
            'recurve_bias': 0.6,
            'northward_bias': 0.8,
            'eastward_bias': 0.7,
            'speed_factor': 0.9
        }
    
    # Apply monthly adjustments to track_coef_mod
    if month in [7, 8, 9]:  # Peak season
        if track_coef_mod:
            track_coef_mod['northward_bias'] *= 1.1
            track_coef_mod['recurve_bias'] *= 1.1
    elif month in [11, 12, 1, 2]:  # Winter season
        if track_coef_mod:
            track_coef_mod['eastward_bias'] *= 1.2
            track_coef_mod['northward_bias'] *= 0.9
    
    # Initial track segment (first few points)
    initial_duration = 8  # Generate first 24 hours (8 × 3-hour steps)
    logging.info(f"Generating initial {initial_duration} timesteps ({initial_duration*3} hours) for {storm_category} in {month}/{year}")

    # Set duration based on month and category (stronger storms & peak season = longer duration)
    base_duration = 0
    if month in [7, 8, 9, 10]:  # Peak season
        base_duration = 48  # 6 days (at 3-hour intervals)
    elif month in [5, 6, 11]:  # Transition months
        base_duration = 40  # 5 days
    else:
        base_duration = 32  # 4 days

    # Category adjustment
    if storm_category == 'STY':
        # Significantly longer duration for super typhoons
        base_duration = max(56, base_duration)  # Minimum 7 days
        duration = np.random.randint(base_duration + 24, base_duration + 48)  # 10-13 days
        logging.info(f"EXTREME STORM: Extended duration to {duration} timesteps ({duration*3} hours)")
    elif storm_category == 'TY':
        duration = np.random.randint(base_duration + 8, base_duration + 24)
    elif storm_category == 'STS':
        duration = np.random.randint(base_duration + 0, base_duration + 16)
    elif storm_category == 'TS':
        duration = np.random.randint(base_duration - 8, base_duration + 8)
    else:  # TD
        duration = np.random.randint(base_duration - 16, base_duration)

    # Ensure minimum duration
    duration = max(duration, 16)  # Minimum 2 days
    
    initial_track_lats, initial_track_lons = generate_track_dynamics(
        initial_lat, initial_lon, month, duration, track_coef=track_coefficients,
        track_coef_mod=track_coef_mod, category=storm_category
    )

    # Generate initial wind speeds
    if historical_winds is not None and len(historical_winds) > 10:
        #initial_wind_speeds = weibull_gev_blend(len(initial_track_lats), historical_winds, month, region="PAR")
        initial_wind_speeds = weibull_gev_blend(len(initial_track_lats), historical_winds, month, region="PAR", year=year)
    else:
        initial_wind_speeds = weibull_gev_blend(len(initial_track_lats), None, month, region="PAR")
    
    initial_wind_speeds = validate_wind_speeds(initial_wind_speeds, caller="generate_synthetic_storm-initial")

    # Initialize full track arrays with the initial segment
    track_lats = initial_track_lats.tolist()
    track_lons = initial_track_lons.tolist()
    wind_speeds = initial_wind_speeds.tolist()
    
    if storm_category == 'STY':
        # Ensure at least some points reach true STY intensity
        max_current = max(wind_speeds)
        if max_current < 106:
            # Force peak to STY level
            peak_idx = np.argmax(wind_speeds)
            target_peak = np.random.uniform(106, min(max_current * 1.2, 125))
             
            # Boost peak and surrounding points
            for boost_idx in range(max(0, peak_idx-2), min(len(wind_speeds), peak_idx+3)):
                distance_factor = 1.0 - abs(boost_idx - peak_idx) * 0.1
                wind_speeds[boost_idx] = max(wind_speeds[boost_idx], target_peak * distance_factor)
            
            logging.info(f"STY PEAK BOOST: Enhanced peak from {max_current:.1f}kt to {max(wind_speeds):.1f}kt")
            
    # Create timestamp start point
    start_day = np.random.randint(1, 28)  # Avoid potential issues with month lengths
    start_date = datetime(year, month, start_day)

    # Keep track of consecutive low wind points
    consecutive_low_count = 0
    max_duration = 48  # Set a maximum possible duration as safety
    current_duration = initial_duration

    # Keep track of consecutive low wind points
    consecutive_low_count = 0
    max_duration = 48  # Set a maximum possible duration as safety
    current_duration = initial_duration

    # Track ERC state
    experiencing_erc = False
    erc_duration = 0  # Track how long the ERC has been happening
    max_erc_duration = 4  # Maximum segments an ERC can last (about 12 hours)
    
    # Track landfall status
    has_made_landfall = False
    over_land_now = False

    # Continue generating the track incrementally, checking winds at each step
    while current_duration < max_duration:
        # Generate next segment (4 timesteps = 12 hours)
        segment_duration = 4
        next_segment_start_lat = track_lats[-1]
        next_segment_start_lon = track_lons[-1]
    
        # Generate next track segment
        next_lats, next_lons = generate_track_dynamics(
            next_segment_start_lat, next_segment_start_lon, month, 
            segment_duration + 1,  # +1 because first point overlaps with previous segment
            track_coef=track_coefficients,
            track_coef_mod=track_coef_mod,
            category=storm_category
        )
    
        # Skip the first point (overlaps with previous segment)
        next_lats = next_lats[1:]
        next_lons = next_lons[1:]
    
        # Generate wind speeds for this segment WITH CONTINUITY
        if historical_winds is not None and len(historical_winds) > 10:
            next_winds = weibull_gev_blend(len(next_lats), historical_winds, month, region="PAR")
        else:
            next_winds = weibull_gev_blend(len(next_lats), None, month, region="PAR")
                
        next_winds = validate_wind_speeds(next_winds, caller="generate_synthetic_storm-segment")
        
        try:
            # Process each wind value in the segment
            for i in range(len(next_winds)):
                # ALWAYS ensure scalar before any operation
                current_wind = ensure_scalar(next_winds[i])
                current_lat = ensure_scalar(next_lats[i] if i < len(next_lats) else 15.0)
                current_lon = ensure_scalar(next_lons[i] if i < len(next_lons) else 125.0)
        
                # Check if over land
                if is_over_land(current_lon, current_lat):
                    # Use safe_comparison for all conditionals
                    if safe_comparison(current_wind, '>=', 106) and safe_comparison(current_lon, '>=', 122.0):
                        # Eastern areas - preserve STY intensity
                        consecutive_land = 0
                        for j in range(max(0, i-3), i+1):
                            if j < len(next_lats) and is_over_land(
                                ensure_scalar(next_lons[j]), 
                                ensure_scalar(next_lats[j])
                            ):
                                consecutive_land += 1
                
                        if consecutive_land <= 1:  # First 3 hours over land
                            logging.info(f"STY KISSING SHORE: {current_wind:.1f}kt maintained for first contact")
                            continue
            
                    # Apply regional intensity caps
                    if safe_comparison(current_wind, '>=', 106):
                        if safe_comparison(current_lon, '>=', 127.0):
                            # Eastern areas - preserve
                            logging.info(f"EASTERN STY PRESERVED: {current_wind:.1f}kt at {current_lon:.1f}°E")
                        elif (safe_comparison(current_lat, '>=', 14.0) and 
                            safe_comparison(current_lat, '<=', 19.5) and
                            safe_comparison(current_lon, '>=', 121.5) and 
                            safe_comparison(current_lon, '<=', 123.0)):
                            # Luzon - preserve
                            logging.info(f"LUZON STY PRESERVED: {current_wind:.1f}kt")
                        else:
                            # Other areas - cap
                            next_winds[i] = min(95.0, current_wind * 0.80)
                            logging.info(f"STY LAND CAP: {current_wind:.1f}kt -> {next_winds[i]:.1f}kt")
            
                    elif safe_comparison(current_wind, '>', 90.0):
                        next_winds[i] = min(90.0, current_wind * 0.8)

        except Exception as e:
            logging.error(f"Error in wind processing: {e}")
            # Ensure all winds are valid scalars
            next_winds = np.array([ensure_scalar(w) for w in next_winds])
        
        # Keep these variables for code compatibility
        experiencing_erc = False
        erc_duration = 0 
        # Keep these variables for code compatibility
        experiencing_erc = False
        erc_duration = 0

        # Natural intensity variability instead of dramatic ERC
        if len(wind_speeds) >= 8:
            # Add small random intensity fluctuations for realism
            # Only for storms over water and above TS strength
            current_lat = track_lats[-1] if track_lats else 15.0
            current_lon = track_lons[-1] if track_lons else 125.0
            over_water = not is_over_land(current_lon, current_lat)
    
            if over_water and len(next_winds) > 0 and max(next_winds) > 34:
                # Apply subtle intensity fluctuations (±2-5%)
                for i in range(len(next_winds)):
                    if next_winds[i] > 34:  # Only for TS and above
                        fluctuation = np.random.uniform(0.98, 1.02)  # ±2% variation
                        next_winds[i] *= fluctuation
        
                # Validate the adjusted winds
                next_winds = validate_wind_speeds(next_winds, caller="natural_variability")
        
        # End of Option C: Replace ERC with Natural Intensity Variability
        
        # Check each point in the new segment for termination condition
        segment_valid = True
        termination_idx = None
        
        # Check each point in the new segment for termination condition
        segment_valid = True
        termination_idx = None
    
        for i, wind in enumerate(next_winds):
            # ENSURE WIND IS A SCALAR VALUE:
            wind = ensure_scalar(wind)
                
            # Get the position for this point
            point_lat = ensure_scalar(next_lats[i] if i < len(next_lats) else 15.0)
            point_lon = ensure_scalar(next_lons[i] if i < len(next_lons) else 125.0)
    
            # Check if point is over land using DEM
            over_land_now = is_over_land(point_lon, point_lat)
    
            # Detect landfall (transition from sea to land)
            if over_land_now and not has_made_landfall:
                has_made_landfall = True
                logging.info(f"Storm made landfall at ({point_lat:.2f}, {point_lon:.2f})")
            
            # Check termination conditions using safe_comparison
            if safe_comparison(wind, '<', 22.0):
                if has_made_landfall:
                    # One-strike rule after landfall
                    termination_idx = i
                    segment_valid = False
                    logging.info(f"Post-landfall ONE STRIKE: Terminated at position {i}")
                    break
                else:
                    # Two-strike rule over water
                    consecutive_low_count += 1
                    if consecutive_low_count >= 2:
                        termination_idx = i
                        segment_valid = False
                        logging.info(f"Two-strike termination at position {i}")
                        break
            else:
                consecutive_low_count = 0
    
        # Check if the point is in eastern PAR
        in_eastern_par = is_inside_par(point_lat, point_lon) and point_lon > 130.0
        print(f"DEBUG TERM1: About to compare has_made_landfall={has_made_landfall} and wind={type(wind)} value={wind}")
    
        # ONE STRIKE RULE - Hard enforcement
        print(f"DEBUG TERM2: About to compare wind={type(wind)} value={wind}")
        if has_made_landfall and wind < 22.0:
            # Terminate immediately at first point below threshold after landfall
            termination_idx = i
            segment_valid = False
            logging.info(f"ENFORCED ONE STRIKE: Hard termination at position {i}")
            # Make sure to include this point as the last one
            next_lats = next_lats[:i+1]
            next_lons = next_lons[:i+1]
            next_winds = next_winds[:i+1]
            break
    
        # Enhanced termination checks with strict two-strike rule
        wind_scalar = ensure_scalar(wind)
        if wind_scalar < 22.0:
            if has_made_landfall:
                # ONE STRIKE RULE: Immediately terminate after landfall if below threshold
                termination_idx = i
                segment_valid = False
                logging.info(f"Post-landfall ONE STRIKE: Track terminated at first point below 22kt")
                break
            else:
                # Apply two-strike rule uniformly regardless of location or conditions
                consecutive_low_count += 1
                if consecutive_low_count >= 2:
                    termination_idx = i
                    segment_valid = False
                    logging.info(f"Track terminated after 2 consecutive points below 22kt (two-strike rule)")
                    break
        else:
            consecutive_low_count = 0  # Reset counter
    
        # Add the valid portion of the segment to the track
        if termination_idx is not None:
            # Include the termination point and one additional point (to show track ending)
            valid_segment_length = min(termination_idx + 1, len(next_lats))
            track_lats.extend(next_lats[:valid_segment_length])
            track_lons.extend(next_lons[:valid_segment_length])
            wind_speeds.extend(next_winds[:valid_segment_length])
    
            # IMMEDIATE LAND CAP - ADD THIS LINE:
            wind_speeds = gradual_land_cap(track_lats, track_lons, wind_speeds)
    
            break  # End track generation
        else:
            # Full segment is valid, add it all
            track_lats.extend(next_lats)
            track_lons.extend(next_lons)
            wind_speeds.extend(next_winds)
    
            # IMMEDIATE LAND CAP - ADD THIS LINE:
            wind_speeds = gradual_land_cap(track_lats, track_lons, wind_speeds)
    
        # Fix oscillations in the full track periodically
        if len(wind_speeds) > 10:  # Every few segments
            wind_speeds = fix_category_threshold_oscillations(np.array(wind_speeds))
            wind_speeds = wind_speeds.tolist()  # Convert back to list
    
        # Update current duration
        current_duration += segment_duration
    
        # Check if the storm has moved outside the extended generation area
        if not is_inside_par(track_lats[-1], track_lons[-1], use_buffer=True):
            # Instead of immediate termination, add gradual weakening for storms exiting PAR
            # Only apply if still above TD strength
            if wind_speeds[-1] >= 22.0:
                # Add 3-6 points of gradual weakening
                decay_steps = np.random.randint(3, 7)
                start_wind = wind_speeds[-1]
        
                for decay_step in range(1, decay_steps + 1):
                    # Calculate decayed wind with exponential decay
                    # More rapid decay for stronger storms
                    decay_factor = 0.7 ** (decay_step / (3 if start_wind < 64 else 2))
                    decay_wind = max(21.0, start_wind * decay_factor)
            
                    # Generate position continuing in same direction but slowing
                    direction_lat = track_lats[-1] - track_lats[-2]
                    direction_lon = track_lons[-1] - track_lons[-2]
                    movement_factor = 0.9 ** decay_step  # Slowing movement
            
                    next_lat = track_lats[-1] + direction_lat * movement_factor
                    next_lon = track_lons[-1] + direction_lon * movement_factor
            
                    # Add to track
                    track_lats.append(next_lat)
                    track_lons.append(next_lon)
                    wind_speeds.append(decay_wind)
            
                    # Stop if below TD threshold (this ensures proper termination)
                    if decay_wind < 22.0:
                        # Add one more point to satisfy two-strike rule
                        final_lat = next_lat + direction_lat * movement_factor * 0.8
                        final_lon = next_lon + direction_lon * movement_factor * 0.8
                        track_lats.append(final_lat)
                        track_lons.append(final_lon)
                        wind_speeds.append(20.0)  # Clearly below threshold
                        break
                
            logging.info(f"Track extended with decay points after exiting extended generation area")
            break

        # Convert back to numpy arrays
        track_lats = np.array(track_lats)
        track_lons = np.array(track_lons)
        wind_speeds = np.array(wind_speeds)

        # TARGETED: Fix segmentation only around category thresholds
        try:
            # Focus only on the STY threshold where purple segmentation occurs
            for i in range(1, len(wind_speeds)-1):
                # If point oscillates across STY threshold, smooth it
                if ((wind_speeds[i-1] >= 100 and wind_speeds[i] < 100 and wind_speeds[i+1] >= 100) or
                    (wind_speeds[i-1] < 100 and wind_speeds[i] >= 100 and wind_speeds[i+1] < 100)):
                    # Simple average to eliminate threshold crossing
                    wind_speeds[i] = (wind_speeds[i-1] + wind_speeds[i+1]) / 2
        
            logging.info(f"Applied threshold-specific smoothing to eliminate segmentation")
            
        except Exception as e:
            logging.error(f"Error in threshold smoothing: {e}")
        
    # Preservation of STY category
    # Calculate if this storm should be an STY based on monthly distribution
    if storm_category == 'STY' or (month in _monthly_category_dist and 
                                    random.random() < _monthly_category_dist[month][4]/100):
        max_wind = max(wind_speeds)
    
        # If maximum wind is close to STY threshold but not quite there
        if 95 <= max_wind < 100:
            # Find peak intensity point
            peak_idx = np.argmax(wind_speeds)
        
            # Boost just enough to cross threshold
            boost_factor = 101 / max_wind
        
            # Apply gradual boost around peak
            window = 3  # Points before and after peak
            for i in range(max(0, peak_idx-window), min(len(wind_speeds), peak_idx+window+1)):
                # Calculate distance from peak 
                distance = abs(i - peak_idx) / window
                # Apply stronger boost near peak, weaker at edges
                local_boost = 1.0 + (boost_factor - 1.0) * (1 - distance)
                wind_speeds[i] *= local_boost
        
            # Validate adjusted winds
            wind_speeds = validate_wind_speeds(wind_speeds, caller="sty_threshold_boost")
        
            logging.info(f"Applied STY threshold boost to storm in category {storm_category}")
        
    # Create timestamps for the final track
    track_times = [start_date + timedelta(hours=3*i) for i in range(len(track_lats))]

    logging.info(f"Final track length: {len(track_lats)} timesteps ({len(track_lats)*3} hours)")
    
    # Calculate pressure, RMW, and other parameters
    pressures = calculate_pressure_from_wind(wind_speeds, env_pressure=1010.0, lat=track_lats)
    rmw_values = [calculate_rmw(wind, lat) for wind, lat in zip(wind_speeds, track_lats)]
    
    # Preservation of STY category - can be toggled on/off
    if ENABLE_STY_THRESHOLD_BOOST:
        # Calculate if this storm should be an STY based on monthly distribution
        if storm_category == 'STY' or (month in _monthly_category_dist and 
                                        random.random() < _monthly_category_dist[month][4]/100):
            max_wind = max(wind_speeds)

            # If maximum wind is close to STY threshold but not quite there
            if 95 <= max_wind < 100:
                # Find peak intensity point
                peak_idx = np.argmax(wind_speeds)
        
                # Boost just enough to cross threshold
                boost_factor = 101 / max_wind
        
                # Apply gradual boost around peak
                window = 3  # Points before and after peak
                for i in range(max(0, peak_idx-window), min(len(wind_speeds), peak_idx+window+1)):
                    # Calculate distance from peak 
                    distance = abs(i - peak_idx) / window
                    # Apply stronger boost near peak, weaker at edges
                    local_boost = 1.0 + (boost_factor - 1.0) * (1 - distance)
                    wind_speeds[i] *= local_boost
        
                # Validate adjusted winds
                wind_speeds = validate_wind_speeds(wind_speeds, caller="sty_threshold_boost")
        
                logging.info(f"Applied STY threshold boost to storm in category {storm_category}")
    
    # Create storm category labels
    categories = []
    for wind in wind_speeds:
        if wind >= 100:
            categories.append("Super Typhoon")
        elif wind >= 64:
            categories.append("Typhoon")
        elif wind >= 48:
            categories.append("Severe Tropical Storm")
        elif wind >= 34:
            categories.append("Tropical Storm")
        elif wind >= 22:
            categories.append("Tropical Depression")
        else:
            categories.append("Remnant Low")  # New category for sub-TD winds
        
    # Create unique storm ID
    if used_ids is None:
        used_ids = set()

    # Ensure we create a unique ID
    storm_id = None
    attempts = 0
    while storm_id is None or storm_id in used_ids:
        storm_id = f"SYN_{year}{month:02d}_{np.random.randint(1, 999):03d}"
        attempts += 1
        if attempts > 100:  # Avoid infinite loop
            # Use timestamp for guaranteed uniqueness
            import time
            storm_id = f"SYN_{year}{month:02d}_{int(time.time())%100000}"
            break

    used_ids.add(storm_id)
    
    # Create dataframe
    storm_df = pd.DataFrame({
        'SID': storm_id,
        'YEAR': year,
        'MONTH': month,
        'ISO_TIME': track_times,
        'LAT': track_lats,
        'LON': track_lons,
        'WIND': wind_speeds,
        'PRES': pressures,
        'RMW': rmw_values,
        'CATEGORY': categories
    })
    
    # Apply the DEM-based landfall wind decay if storm intersects land
    if any([is_over_land(lon, lat) for lon, lat in zip(track_lons, track_lats)]):
        try:
            storm_df = apply_dem_based_decay(storm_df, dem_path=DEM_PATH)
            logging.info(f"Applied DEM-based wind decay to storm {storm_id}")
            
            storm_df = preserve_eastern_approaches(storm_df)
            
            # Boundary decay function call here
            storm_df = apply_boundary_decay(storm_df)
            logging.info(f"Applied boundary decay to storm {storm_id}")
            
            logging.info(f"Using gradual land intensity degradation instead of strict rules")
        
            # IMPORTANT: Add post-decay validation here
            # First check if any winds dropped below threshold after land decay
            if any(storm_df['WIND'] < 22.0):
                # Find the index of the first occurrence of winds below 22kt
                below_threshold = storm_df['WIND'] < 22.0  # Create boolean mask of sub-threshold points
                first_low_idx = below_threshold[below_threshold].index[0]  # Get index of first True value
            
                # Keep only points up to and including the first sub-threshold point
                # This enforces the one-strike rule by truncating the track
                storm_df = storm_df.loc[:first_low_idx]
                logging.info(f"Post-decay validation: Terminated track at first point below 22kt")
                
            # Diagnostic: Check if STY survived over land
            if storm_df['WIND'].max() >= 100:
                land_points = [is_over_land(lon, lat) for lon, lat in zip(storm_df['LON'], storm_df['LAT'])]
                if any(land_points):
                    # Get wind values over land
                    land_winds = storm_df.loc[land_points, 'WIND']
                    max_land_wind = land_winds.max() if not land_winds.empty else 0
                
                    # Only log CRITICAL if land winds are actually too high
                    if max_land_wind > 90:
                        logging.warning(f"CRITICAL: Storm {storm_id} has maximum wind of {max_land_wind:.1f}kt over land")
                
                        # Force cap winds over land
                        storm_df.loc[land_points, 'WIND'] = np.minimum(storm_df.loc[land_points, 'WIND'], 90.0)
                        logging.info(f"Forcibly capped winds over land to 90kt maximum")
                    else:
                        # Log success instead of false alarm
                        logging.info(f"Storm {storm_id} correctly weakened to {max_land_wind:.1f}kt over land (max water: {storm_df['WIND'].max():.1f}kt)")
                
        except Exception as e:
            logging.error(f"Error applying DEM-based decay to storm {storm_id}: {e}")
    
    # Apply western intensity cap for all storms
    try:
        storm_df = enforce_western_intensity_cap(storm_df)
        logging.info(f"Applied western intensity cap to storm {storm_id}")
    except Exception as e:
        logging.error(f"Error applying western intensity cap to storm {storm_id}: {e}")
        
    # Create storm category labels
    categories = []
    for wind in wind_speeds:
        if wind >= 100:
            categories.append("Super Typhoon")
        elif wind >= 64:
            categories.append("Typhoon")
        elif wind >= 48:
            categories.append("Severe Tropical Storm")
        elif wind >= 34:
            categories.append("Tropical Storm")
        elif wind >= 22:
            categories.append("Tropical Depression")
        else:
            categories.append("Remnant Low")  # New category for sub-TD winds
        
    # Create unique storm ID
    if used_ids is None:
        used_ids = set()
    
    # Apply orographic effects for extreme storms
    try:
        # Apply orographic effects (primarily for extreme storms)
        if storm_df['WIND'].max() >= 100:  # Only for Super Typhoons
            storm_df = apply_orographic_effects(storm_df, dem_path=DEM_PATH)
            logging.info(f"Applied orographic effects to extreme storm {storm_id}")
    except Exception as e:
        logging.error(f"Error applying orographic effects to storm {storm_id}: {e}")
    
    # Final safety check - Remove any points after a sub-22kt point post-landfall
    landfall_detected = False
    for i, row in storm_df.iterrows():
        if is_over_land(row['LON'], row['LAT']):
            landfall_detected = True
        if landfall_detected and row['WIND'] < 22.0:
            # Keep only up to this point
            storm_df = storm_df.loc[:i]
            logging.info("FINAL CHECK: Truncated track at first point below 22kt after landfall")
            break
    
    # Gradual land intensity degradation
    for i, row in storm_df.iterrows():
        if is_over_land(row['LON'], row['LAT']):
            current_wind = row['WIND']
        
            # Find how many consecutive land points we've had
            land_duration = 0
            for j in range(max(0, i-10), i+1):  # Look back up to 10 timesteps
                if j < len(storm_df) and is_over_land(storm_df.iloc[j]['LON'], storm_df.iloc[j]['LAT']):
                    land_duration += 1
                else:
                    land_duration = 0  # Reset if we hit water
        
            # Gradual degradation based on time over land
            if current_wind > 90:
                if land_duration == 1:  # First timestep over land (t+0h)
                    current_wind_scalar = ensure_scalar(current_wind)
                    if current_wind_scalar >= 106:
                        new_wind = current_wind
                    else:
                        new_wind = max(90, current_wind * 0.95)  # 5% reduction
                elif land_duration == 2:  # Second timestep (t+3h)  
                    new_wind = max(90, current_wind * 0.90)  # 10% total reduction
                elif land_duration == 3:  # Third timestep (t+6h)
                    new_wind = 90  # Hard cap at 90kt after 6 hours
                else:  # 4+ timesteps (t+9h and beyond)
                    new_wind = 90  # Maintain cap
            
                storm_df.loc[i, 'WIND'] = new_wind
            
                if new_wind != current_wind:
                    logging.info(f"GRADUAL LAND CAP: Point {i} after {land_duration*3}h over land: {current_wind:.1f}kt -> {new_wind:.1f}kt")
    
    # ADD THIS RIGHT BEFORE "return storm_df":
    for i, row in storm_df.iterrows():
        if row['LON'] < 120.0 and row['WIND'] >= 106:
            storm_df.loc[i, 'WIND'] = 95.0
            logging.info(f"FINAL WEST CAP: Storm at {row['LON']:.1f}°E capped at 95kt")
        
    return storm_df
    
def safe_storm_generation(year: int, month: int, used_ids: set, max_attempts: int = 10) -> Optional[pd.DataFrame]:
    """Generate storm with comprehensive error handling"""
    
    for attempt in range(max_attempts):
        try:
            storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, set())
            
            # Validate generated storm
            if storm is None or storm.empty:
                raise ValueError("Generated empty storm")
                
            if len(storm) < 2:
                raise ValueError("Storm track too short")
                
            # Check for data consistency
            required_cols = ['LAT', 'LON', 'WIND', 'SID']
            missing_cols = set(required_cols) - set(storm.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
                
            return storm
            
        except Exception as e:
            logging.warning(f"Storm generation attempt {attempt + 1} failed: {e}")
            if attempt == max_attempts - 1:
                logging.error(f"Failed to generate storm after {max_attempts} attempts")
                return None
    
    return None
    
def validate_wind_threshold(storm_df):
    """Ensure no storm continues after two consecutive points with winds below 22 knots"""
    storm_ids = storm_df['SID'].unique()
    valid_storms = []
    
    for sid in storm_ids:
        storm = storm_df[storm_df['SID'] == sid].copy()
        consecutive_low = 0
        termination_idx = None
        
        for i, row in storm.iterrows():
            if row['WIND'] < 22.0:
                consecutive_low += 1
                if consecutive_low >= 2:  # Two strikes rule
                    termination_idx = i - consecutive_low + 1
                    break
            else:
                consecutive_low = 0  # Reset counter
                
        if termination_idx is not None:
            # Keep exactly 2 points below threshold
            storm = storm.iloc[:termination_idx+2]
        
        valid_storms.append(storm)
    
    return pd.concat(valid_storms, ignore_index=True) if valid_storms else pd.DataFrame()

def filter_storms_for_realism(yearly_storms, min_par_points=1):
    """
    Filter storm tracks to ensure they realistically represent historical patterns
    and remove any obvious outliers or unrealistic tracks.
    
    Args:
        yearly_storms: DataFrame with all synthetic storms for a year
        min_par_points: Minimum number of points that must be inside PAR
        
    Returns:
        DataFrame with filtered storms
    """
    # Validate input
    if yearly_storms is None or not isinstance(yearly_storms, pd.DataFrame) or yearly_storms.empty:
        logging.warning("Empty or invalid DataFrame provided to filter_storms_for_realism")
        return pd.DataFrame()
    
    # Check required columns
    required_columns = {'SID', 'LAT', 'LON'}
    missing_columns = required_columns - set(yearly_storms.columns)
    if missing_columns:
        logging.error(f"DataFrame missing required columns: {missing_columns}")
        return pd.DataFrame()
        
    filtered_storms = []
    
    # Group by storm ID
    for sid in yearly_storms['SID'].unique():
        storm_data = yearly_storms[yearly_storms['SID'] == sid]
        
        # Count points inside actual PAR
        par_points = 0
        for _, row in storm_data.iterrows():
            if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                par_points += 1
        
        # Filter condition 1: Must have minimum points inside PAR
        if par_points < min_par_points:
            continue
            
        # Filter condition 2: Check for unrealistic zonal movement
        # Calculate the longitude span
        lon_min = storm_data['LON'].min()
        lon_max = storm_data['LON'].max()
        lon_span = lon_max - lon_min
        
        # Calculate the latitude span
        lat_min = storm_data['LAT'].min()
        lat_max = storm_data['LAT'].max()
        lat_span = lat_max - lat_min
        
        # Reject tracks that move almost purely zonally (very little latitude change)
        # This is rare in nature, especially for long tracks
        if lon_span > 10 and lat_span < 2:
            continue
            
        # Filter condition 3: Check for unrealistic track shape
        # Calculate track sinuosity (length / direct distance)
        positions = list(zip(storm_data['LAT'], storm_data['LON']))
        if len(positions) > 2:
            track_length = 0
            for i in range(1, len(positions)):
                track_length += calculate_distance(
                    positions[i-1][0], positions[i-1][1],
                    positions[i][0], positions[i][1]
                )
                
            direct_distance = calculate_distance(
                positions[0][0], positions[0][1],
                positions[-1][0], positions[-1][1]
            )
            
            # Avoid division by zero
            if direct_distance > 0:
                sinuosity = track_length / direct_distance
                
                # Reject extremely straight or extremely sinuous tracks
                if sinuosity < 1.05 or sinuosity > 3.0:
                    continue
        
        # Filter condition 4: Check for too many Remnant Low points
        remnant_low_points = (storm_data['WIND'] < 22.0).sum()
        total_points = len(storm_data)
        
        # If more than 10% of the track is Remnant Low, reject it
        if total_points > 0 and remnant_low_points / total_points > 0.10:
            continue
        
        # Add storm to filtered list if it passes all criteria
        filtered_storms.append(storm_data)
    
    # Combine all filtered storms
    if filtered_storms:
        combined_df = pd.concat(filtered_storms, ignore_index=True)
        # Apply two strikes validation to ensure no tracks continue after 3 consecutive points below 22kt
        validated_df = validate_wind_threshold(combined_df)
        return validated_df
    else:
        return pd.DataFrame()

def post_process_synthetic_tracks(yearly_storms, year):
    """
    Apply post-processing to synthetic tracks to enhance category-specific patterns
    and ensure tracks match historical hotspot patterns with improved eastern coverage.
    
    Args:
        yearly_storms: DataFrame with synthetic storms
        year: Target year for labeling
        
    Returns:
        DataFrame with post-processed storm tracks
    """
    if yearly_storms.empty:
        return yearly_storms
    
    processed_storms = []
    
    # Count storms by genesis region to ensure balanced distribution
    east_region_count = 0
    west_region_count = 0
    
    # First pass: count genesis points by region
    for sid in yearly_storms['SID'].unique():
        storm_data = yearly_storms[yearly_storms['SID'] == sid]
        if storm_data.empty or len(storm_data) < 2:
            continue
            
        genesis_lon = storm_data['LON'].iloc[0]
        if genesis_lon >= 126:
            east_region_count += 1
        else:
            west_region_count += 1
    
    # Calculate east-west ratio
    total_storms = east_region_count + west_region_count
    east_ratio = east_region_count / total_storms if total_storms > 0 else 0
    
    # Target ratio - we want at least 60% eastern genesis for better coverage
    target_east_ratio = 0.60
    
    # Process each storm individually
    for sid in yearly_storms['SID'].unique():
        storm_data = yearly_storms[yearly_storms['SID'] == sid]
        
        # Skip if too few points
        if len(storm_data) < 5:
            processed_storms.append(storm_data)
            continue
            
        # Get storm details
        month = storm_data['MONTH'].iloc[0]
        
        # Get maximum wind speed and corresponding category
        max_wind = storm_data['WIND'].max()
        
        if max_wind >= 100:
            category = 'STY'
        elif max_wind >= 64:
            category = 'TY'
        elif max_wind >= 48:
            category = 'STS'
        elif max_wind >= 34:
            category = 'TS'
        else:
            category = 'TD'
        
        # Get genesis and track points
        genesis_lat = storm_data['LAT'].iloc[0]
        genesis_lon = storm_data['LON'].iloc[0]
        
        # Apply category and month-specific adjustments
        modified_data = storm_data.copy()
        
        # If we have too few eastern genesis storms, shift more storms eastward
        if east_ratio < target_east_ratio and genesis_lon < 130:
            # Higher chance of shifting for certain categories
            shift_chance = 0.7  # Base chance
            
            if category in ['STY', 'TY']:
                shift_chance = 0.85  # Higher chance for stronger storms
            elif month in [7, 8, 9, 10]:
                shift_chance = 0.80  # Higher chance during peak months
                
            if np.random.random() < shift_chance:
                # Shift eastward by a random amount
                lon_shift = np.random.uniform(3, 7)
                modified_data['LON'] = modified_data['LON'] + lon_shift
        
        # 1. Super Typhoons - ensure they follow the typical eastern corridor pattern
        if category == 'STY':
            # Shift tracks that are too far west
            if genesis_lon < 130 and month in [7, 8, 9, 10]:
                # Shift eastward by a random amount
                lon_shift = np.random.uniform(3, 6)
                modified_data['LON'] = modified_data['LON'] + lon_shift
        
        # 2. Tropical Depressions - ensure proper concentration east of Visayas
        elif category == 'TD':
            # If genesis is in northern PAR but should be in central/southern
            if genesis_lat > 15 and month in [12, 1, 2, 3, 4]:
                # Shift southward
                lat_shift = np.random.uniform(-3, -6)
                modified_data['LAT'] = modified_data['LAT'] + lat_shift
        
        # 3. Ensure proper monthly pattern for July-September storms
        if month in [7, 8, 9] and genesis_lat < 10:
            # Shift northern storms north during peak months
            lat_shift = np.random.uniform(3, 7)
            modified_data['LAT'] = modified_data['LAT'] + lat_shift
        
        # 4. Adjust February tracks to match the southeastern pattern
        if month == 2 and genesis_lon < 125:
            # Shift eastward
            lon_shift = np.random.uniform(5, 10)
            modified_data['LON'] = modified_data['LON'] + lon_shift
        
        # 5. NEW: Ensure Super Typhoons in peak season have enough eastern genesis
        if category == 'STY' and month in [8, 9, 10] and genesis_lon < 132:
            # Shift eastward substantially
            lon_shift = np.random.uniform(5, 8)
            modified_data['LON'] = modified_data['LON'] + lon_shift
        
        # 6. NEW: Improve far eastern coverage for all categories
        if np.random.random() < 0.15:  # 15% chance for any storm
            if genesis_lon < 135:
                # Create some far eastern genesis points
                lon_shift = np.random.uniform(3, 7)
                modified_data['LON'] = modified_data['LON'] + lon_shift
        
        processed_storms.append(modified_data)
    
    # Combine all processed storms
    if processed_storms:
        combined_storms = pd.concat(processed_storms, ignore_index=True)
        
        # breakup horizontal banding
        combined_storms = break_up_horizontal_banding(combined_storms)
        
        return combined_storms
    else:
        return yearly_storms        

def process_yearly_storms(yearly_storms):
    """Process yearly storm data with enhanced termination rules."""
    # Apply improvements to each storm individually
    processed_storms = []
    
    for storm_id in yearly_storms['SID'].unique():
        storm_data = yearly_storms[yearly_storms['SID'] == storm_id]
        
        # Apply enhanced three-strike rule
        storm_data = apply_enhanced_termination_rules(storm_data)
        
        # Apply aging-based decay
        storm_data = apply_aging_decay(storm_data)
        
        # Add to collection
        processed_storms.append(storm_data)
    
    # Combine all processed storms
    if processed_storms:
        processed_df = pd.concat(processed_storms, ignore_index=True)
        
        # Apply post-process filtering for anomalous terminations
        processed_df = filter_anomalous_terminations(processed_df)
        
        # Wind Speed Tracking DIAGNOSTIC
        max_wind_by_storm = {}
        for storm_id in processed_df['SID'].unique():
            storm_data = processed_df[processed_df['SID'] == storm_id]
            max_wind = storm_data['WIND'].max()
            max_wind_by_storm[storm_id] = max_wind

        max_winds = list(max_wind_by_storm.values())
        logging.info(f"Maximum wind distribution: min={min(max_winds):.1f}, mean={np.mean(max_winds):.1f}, max={max(max_winds):.1f}")
        logging.info(f"Storm category counts by maximum intensity:")
        logging.info(f"  TD (22-33kt): {sum(1 for w in max_winds if 22 <= w < 34)}")
        logging.info(f"  TS (34-47kt): {sum(1 for w in max_winds if 34 <= w < 48)}")
        logging.info(f"  STS (48-63kt): {sum(1 for w in max_winds if 48 <= w < 64)}")
        logging.info(f"  TY (64-99kt): {sum(1 for w in max_winds if 64 <= w < 100)}")
        logging.info(f"  STY (>=100kt): {sum(1 for w in max_winds if w >= 100)}")
        
        return processed_df
    else:
        return pd.DataFrame()

def clean_remnant_low_tracks(storms_df):
    """
    Post-processing function to clean up tracks with Remnant Low points.
    """
    clean_df = storms_df.copy()
    
    # Get all unique storm IDs
    storm_ids = clean_df['SID'].unique()
    
    # Create a list to hold clean storm segments
    clean_storms = []
    
    for sid in storm_ids:
        # Get data for this storm
        storm_data = clean_df[clean_df['SID'] == sid]
        
        # Check for TWO consecutive points below 22 knots
        consecutive_low_limit = 2
        below_22_count = 0
        termination_idx = None
        
        for i, row in storm_data.iterrows():
            if row['WIND'] < 22.0:
                below_22_count += 1
                if below_22_count >= consecutive_low_limit:
                    # We found 3 consecutive low points
                    termination_idx = i - below_22_count + 1
                    break
            else:
                below_22_count = 0
        
        # If we found a termination point, trim the storm
        if termination_idx is not None:
            # Find the index in the original DataFrame
            indices = storm_data.index.tolist()
            try:
                idx_position = indices.index(termination_idx)
                # Keep the track up to and including the 3 points below threshold
                if idx_position + consecutive_low_limit <= len(indices):
                    clean_storms.append(storm_data.iloc[:idx_position+consecutive_low_limit])
                else:
                    clean_storms.append(storm_data)
            except ValueError:
                # If index not found, keep the whole track
                clean_storms.append(storm_data)
        else:
            # No termination needed, keep the whole track
            clean_storms.append(storm_data)
    
    # ADD THIS LINE HERE - Right before the return statement
    logging.info(f"Cleanup found {len([s for s in clean_storms if len(s) < len(clean_df[clean_df['SID'] == s['SID'].iloc[0]])])} storms with Remnant Low segments")
    
    # Combine all clean storms
    if clean_storms:
        return pd.concat(clean_storms, ignore_index=True)
    else:
        return pd.DataFrame()

def break_up_horizontal_banding(storm_tracks_df):
    """
    Post-process storm tracks to eliminate artificial horizontal banding.
    
    Args:
        storm_tracks_df: DataFrame with storm track positions
        
    Returns:
        Modified DataFrame with more realistic distribution
    """
    # Identify problem latitude band
    problem_band = (storm_tracks_df['LAT'] >= 9.5) & (storm_tracks_df['LAT'] <= 15.5)
    
    # Count points in the problem band
    band_points = problem_band.sum()
    total_points = len(storm_tracks_df)
    band_percentage = band_points / total_points * 100
    
    logging.info(f"Found {band_points} points ({band_percentage:.1f}%) in the 9.5-15.5°N band")
    
    # If percentage is too high (over 30%), apply correction
    if band_percentage > 30:
        # Make a copy to avoid modifying the original during iteration
        modified_df = storm_tracks_df.copy()
        
        # Group by storm ID
        for storm_id in modified_df['SID'].unique():
            storm_mask = modified_df['SID'] == storm_id
            storm_data = modified_df.loc[storm_mask]
            
            # Check if this storm has multiple points in the problem band
            band_points_in_storm = problem_band & storm_mask
            if band_points_in_storm.sum() >= 3:
                # Calculate the percentage of this storm's points in the band
                storm_band_percentage = band_points_in_storm.sum() / len(storm_data) * 100
                
                # Only modify storms with excessive time in the band (>50%)
                if storm_band_percentage > 50:
                    # Find the indices in the problem band
                    band_indices = np.where(band_points_in_storm)[0]
                    
                    # Only modify some of the points (random selection)
                    points_to_modify = np.random.choice(
                        band_indices, 
                        size=int(len(band_indices) * 0.7),  # Modify 70% of points
                        replace=False
                    )
                    
                    # Apply perturbations
                    for idx in points_to_modify:
                        # Determine direction (up or down) based on position in band
                        lat = modified_df.loc[idx, 'LAT']
                        
                        # Stronger displacement near center of band
                        center_dist = abs(lat - 12.5)
                        max_displacement = 0.8 * np.exp(-0.5 * (center_dist / 1.5)**2)
                        
                        # Use larger displacements where concentration is highest
                        if 10.5 <= lat <= 14.5:
                            # Decide direction - biased away from center
                            if lat < 12.5:
                                direction = -1  # Push south
                            else:
                                direction = 1   # Push north
                                
                            # Add random factor to avoid creating new bands
                            if np.random.random() < 0.3:
                                direction *= -1  # Sometimes push opposite direction
                                
                            # Apply displacement
                            displacement = direction * np.random.uniform(0.3, max_displacement)
                            modified_df.loc[idx, 'LAT'] += displacement
                            
                            # Add some longitudinal variation to maintain realistic tracks
                            modified_df.loc[idx, 'LON'] += np.random.normal(0, 0.3)
        
        # Verify the effect
        new_band_points = ((modified_df['LAT'] >= 9.5) & (modified_df['LAT'] <= 15.5)).sum()
        new_percentage = new_band_points / len(modified_df) * 100
        logging.info(f"After correction: {new_band_points} points ({new_percentage:.1f}%) in the 9.5-15.5°N band")
        
        return modified_df
    else:
        # No correction needed
        return storm_tracks_df
        
def generate_yearly_storms(year, total_storms=None, par_only=True):
    """
    Generate a monthly distribution of storms closely matching historical PAR data.
    
    Args:
        year: Target year
        total_storms: Total number of storms (if None, use forecast or historical average)
    
    Returns:
        DataFrame with all synthetic storms for the year
    """
    # Load historical data if not already loaded
    load_historical_data()
    
    # Updated with actual point counts from provided summary
    
    # A (1923-2023, PAR) with future trends approach
    historical_distribution = {
        1: 46,  # January (stable, 0)
        2: 29,   # February (stable, 0)
        3: 31,   # March (increasing, +1)
        4: 57,  # April (stable, 0)
        5: 92,  # May (increasing, +1) 
        6: 167,  # June (stable, 0)
        7: 356,  # July (decreasing, -2)
        8: 404,  # August (decreasing, -1)
        9: 377,  # September (decreasing, -1)
        10: 300, # October (increasing, +1)
        11: 222, # November (stable, 0)
        12: 135  # December (stable, 0)
    }
    
    # B (1977-2023, PAR)
    #historical_distribution = {
    #    1: 23,  # January
    #    2: 13,   # February
    #    3: 20,   # March
    #    4: 27,  # April
    #    5: 51,  # May
    #    6: 88,  # June
    #    7: 167,  # July
    #    8: 183,  # August
    #    9: 178,  # September
    #    10: 157, # October
    #    11: 112, # November
    #    12: 70  # December
    #}
    
    # Determine total storms
    if total_storms is None:
        # Check forecast first
        if _df_forecast is not None and not _df_forecast.empty:
            year_forecast = _df_forecast[_df_forecast['YEAR'] == year]
            if not year_forecast.empty and 'Storm_Count' in year_forecast.columns:
                total_storms = int(round(year_forecast['Storm_Count'].iloc[0]))
                logging.info(f"Using forecast storm count for year {year}: {total_storms}")
            else:
                # If this specific year is not in the forecast, use historical average
                logging.warning(f"Year {year} not found in forecast data")
                total_storms = 18  # Historical average is around 18
                logging.info(f"Using historical average for year {year}: {total_storms}")
    
        # If no forecast available, use historical average
        if total_storms is None or total_storms <= 0:
            total_storms = 18
            logging.warning(f"No storm count forecast available, using historical average: {total_storms}")
    
    # Calculate storm counts for each month based on historical proportions
    month_counts = {}
    total_historical_points = sum(historical_distribution.values())
    
    for month, historical_count in historical_distribution.items():
        # Calculate proportion and apply to current year's total
        proportion = historical_count / total_historical_points
        month_count = max(1, round(total_storms * proportion))
        
        # Additional factor: months with few storms historically should 
        # sometimes have no storms in certain years
        if month in [1, 2, 3] and random.random() < 0.3:  # 30% chance of no storms in winter months
            month_count = 0
        
        month_counts[month] = month_count
    
    # Adjust to ensure total matches desired storm count
    current_total = sum(month_counts.values())
    if current_total != total_storms and total_storms > 0:
        difference = total_storms - current_total
        
        # Calculate monthly weights for adjustment
        weights = {}
        for month, count in month_counts.items():
            # Higher weights for peak season months
            if month in [7, 8, 9, 10]:
                weights[month] = 5
            elif month in [5, 6, 11]:
                weights[month] = 3
            else:
                weights[month] = 1
                
            # Don't add storms to months with zero count
            if count == 0:
                weights[month] = 0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for month in weights:
                weights[month] = weights[month] / total_weight
        
        # Distribute difference according to weights
        remaining = difference
        while abs(remaining) > 0 and any(weights.values()):
            # Choose a month based on weights
            months = list(weights.keys())
            probs = [weights[m] for m in months]
            if sum(probs) == 0:
                break
                
            month = np.random.choice(months, p=[w/sum(probs) for w in probs])
            
            if remaining > 0:
                month_counts[month] += 1
                remaining -= 1
            else:
                if month_counts[month] > 0:  # Don't reduce below zero
                    month_counts[month] -= 1
                    remaining += 1
    
    logging.info(f"Generating {sum(month_counts.values())} storms for year {year}")
    
    # Generate storms for each month
    storm_dfs = []
    
    # Keep track of used storm IDs to avoid duplicates
    used_ids = set()
    
    # Generate storms for each month
    for month, count in month_counts.items():
        month_storms = []
        # INCREASE attempts significantly
        max_attempts = count * 10 if par_only else count * 2  # Increased from 3x to 10x
        actual_attempts = 0
        consecutive_failures = 0
        max_consecutive_failures = 20  # Add a consecutive failure limit
    
        while len(month_storms) < count and actual_attempts < max_attempts:
            actual_attempts += 1
            try:
                #storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, used_ids)
                storm = safe_storm_generation(year, month, used_ids, max_attempts=3)
                if storm is None:
                    consecutive_failures += 1
                    continue
            
                # If par_only, check if storm passes through PAR
                if par_only:
                    passes_par = False
                    for _, row in storm.iterrows():
                        if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                            passes_par = True
                            break
                
                    if passes_par:
                        month_storms.append(storm)
                        used_ids.add(storm['SID'].iloc[0])
                        consecutive_failures = 0  # Reset on success
                    else:
                        consecutive_failures += 1
                        # If too many consecutive failures, try with eastern genesis bias
                        if consecutive_failures >= max_consecutive_failures:
                            logging.warning(f"Too many consecutive failures for {year}-{month}, switching to eastern genesis strategy")
                            # Force eastern genesis for better PAR entry
                            initial_lat = np.random.uniform(10, 18)
                            initial_lon = np.random.uniform(133, 140)
                            # Continue with the loop
                else:
                    month_storms.append(storm)
                    used_ids.add(storm['SID'].iloc[0])
                    consecutive_failures = 0                    
                
            except Exception as e:
                logging.error(f"Error generating storm for {year}-{month}: {e}")
                consecutive_failures += 1
    
        storm_dfs.extend(month_storms)
        # Be more aggressive if we're short
        if len(month_storms) < count:
            shortfall = count - len(month_storms)
            logging.warning(f"Month {month}: Only generated {len(month_storms)}/{count} storms after {actual_attempts} attempts")
            logging.info(f"Attempting emergency generation for {shortfall} storms...")
            
            # Emergency generation with more favorable parameters
            emergency_attempts = 0
            while len(month_storms) < count and emergency_attempts < shortfall * 20:
                emergency_attempts += 1
                try:
                    # Use more favorable genesis locations
                    if month in [7, 8, 9, 10]:  # Peak season
                        lat = np.random.uniform(12, 20)
                        lon = np.random.uniform(130, 138)
                    else:
                        lat = np.random.uniform(8, 16)
                        lon = np.random.uniform(128, 135)
                    # Generate with specific genesis point
                    storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, used_ids)
                    # Manually adjust genesis if needed
                    if storm is not None and not storm.empty:
                        storm.loc[storm.index[0], 'LAT'] = lat
                        storm.loc[storm.index[0], 'LON'] = lon
                        
                        # Re-check PAR entry
                        passes_par = any(is_inside_par(r['LAT'], r['LON'], use_buffer=False) 
                                       for _, r in storm.iterrows())
                        
                        if passes_par:
                            month_storms.append(storm)
                            storm_dfs.append(storm)
                            used_ids.add(storm['SID'].iloc[0])
                            logging.info(f"Emergency generation successful, now have {len(month_storms)}/{count}")
            
                except Exception as e:
                    if emergency_attempts % 10 == 0:
                        logging.error(f"Emergency generation error: {e}")
                        
        logging.info(f"Month {month}: Generated {len(month_storms)}/{count} storms after {actual_attempts} attempts")
    
    # Combine all storms into a single DataFrame
    if storm_dfs:
        yearly_storms = pd.concat(storm_dfs, ignore_index=True)
        logging.info(f"Generated {yearly_storms['SID'].nunique()} storms with {len(yearly_storms)} points for year {year}")
        
        ## Free memory used by individual storm dataframes after they're combined
        #del storm_dfs
        #import gc
        #gc.collect()
    
        # Apply the enhanced filtering for realism
        filtered_storms = filter_storms_for_realism(yearly_storms, min_par_points=1 if par_only else 2)
        
        if not filtered_storms.empty:
            logging.info(f"After filtering, kept {filtered_storms['SID'].nunique()} storms with {len(filtered_storms)} points for year {year}")
            
            # If we have fewer than the forecast number, keep generating until we reach it
            actual_count = filtered_storms['SID'].nunique()
            max_attempts = 50  # INCREASED: More attempts from 20 to 50 for better success rate
            attempts = 0
            progress_made = True  # Track if we're making progress

            while actual_count < total_storms and attempts < max_attempts and progress_made:
                attempts += 1
                logging.info(f"Need {total_storms - actual_count} more storms for year {year}, generating... (attempt {attempts}/{max_attempts})")
    
                # Track the count before this attempt to see if we make progress
                previous_count = actual_count
    
                # Generate more storms, distributed across months to match historical patterns
                additional_storms = []
    
                # Choose months based on weighted probability from historical distribution
                month_probs = {m: historical_distribution[m] / total_historical_points for m in historical_distribution}
                # Request more storms than needed to improve chances of success
                num_to_generate = min(10, (total_storms - actual_count) * 2)  # Generate at most 10 at once
                months_to_generate = np.random.choice(
                    list(month_probs.keys()),
                    size=num_to_generate,
                    p=list(month_probs.values())
                )
                
                for month in months_to_generate:
                    try:
                        storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, used_ids)
                        try:
                            # Make sure we properly handle the storm dataframe
                            if isinstance(storm, pd.DataFrame) and not storm.empty:
                                # Ensure we have a proper DataFrame for filtering
                                if len(storm.shape) > 2:  # Check for more than 2 dimensions
                                    logging.warning(f"Storm has unexpected dimensions: {storm.shape}. Attempting to fix.")
                                    storm_df = pd.DataFrame([storm])  # Wrap in list
                                else:
                                    storm_df = storm  # Use as is
                
                                filtered_storm = filter_storms_for_realism(storm_df)
                                if not filtered_storm.empty:
                                    additional_storms.append(storm)
                                    used_ids.add(storm['SID'].iloc[0])
                        except Exception as e:
                            logging.error(f"Error filtering storm for {year}-{month}: {e}")
                    except Exception as e:
                        logging.error(f"Error generating additional storm for {year}-{month}: {e}")
                
                # Check if we made any progress in this attempt
                if additional_storms:
                    # Add new storms to our filtered list
                    filtered_storms = pd.concat([filtered_storms] + additional_storms, ignore_index=True)  # FIXED: Update filtered_storms
                    actual_count = filtered_storms['SID'].nunique()
                    logging.info(f"Added {len(additional_storms)} more storms, now have {actual_count}")
                    
                    # Check if we made progress
                    progress_made = (actual_count > previous_count)
                else:
                    progress_made = False
                    logging.warning(f"Failed to generate any additional storms in attempt {attempts}")

            if actual_count < total_storms:
                shortfall = total_storms - actual_count
                if shortfall <= 2:  # Accept small shortfalls (1-2 storms)
                    logging.info(f"Generated {actual_count} out of {total_storms} storms for year {year} (shortfall of {shortfall} is acceptable)")
                else:
                    logging.warning(f"Could only generate {actual_count} out of {total_storms} storms for year {year} after {attempts} attempts")
                    logging.info(f"Proceeding with {actual_count} storms (shortfall of {shortfall})")
            
            # Enforce cap to prevent exceeding the forecast count
            if filtered_storms['SID'].nunique() > total_storms:
                excess = filtered_storms['SID'].nunique() - total_storms
                logging.warning(f"Overshooting detected: trimming {excess} storms to match forecast of {total_storms}")
                to_keep = filtered_storms.drop_duplicates('SID').sample(n=total_storms, random_state=42)['SID']
                filtered_storms = filtered_storms[filtered_storms['SID'].isin(to_keep)]
        
            return filtered_storms  # FIXED: Always return filtered_storms if we have some
        
        else:
            logging.warning(f"All storms were filtered out for year {year}, regenerating...")
            # Try again with different random seed
            return generate_yearly_storms(year, total_storms)
    else:
        logging.warning(f"No storms generated for year {year}")
        return pd.DataFrame()

def generate_multi_year_storms(start_year, end_year, output_directory, target_storm_count=None):
    """
    Generate synthetic tropical cyclones for a range of years.
    
    Args:
        start_year: First year to generate storms for
        end_year: Last year to generate storms for
        output_directory: Directory to save yearly storm files
        target_storm_count: Exact number of storms to generate (if specified)
        
    Returns:
        Total number of storms generated
    """
    # If targeting exact storm count, prepare to collect all storms
    # But don't override each year's forecast count
    if target_storm_count is not None:
        all_storms = []
        
        # Calculate total from forecast to compare with target
        forecast_total = 0
        if _df_forecast is not None and not _df_forecast.empty:
            years_in_forecast = _df_forecast[(_df_forecast['YEAR'] >= start_year) & 
                                          (_df_forecast['YEAR'] <= end_year)]
            if not years_in_forecast.empty and 'Storm_Count' in years_in_forecast.columns:
                forecast_total = int(round(years_in_forecast['Storm_Count'].sum()))
                
        logging.info(f"Targeting {target_storm_count} storms over {end_year-start_year+1} years. Forecast total: {forecast_total}")
    else:
        total_storm_count = 0
    
    # First phase: Generate storms based on yearly forecasts
    for year in range(start_year, end_year + 1):
        logging.info(f"Processing year {year}...")
        
        # Always use the year's forecast count
        year_storms = generate_yearly_storms(year)
        
        if not year_storms.empty:
            # Apply enhanced termination processing
            year_storms = process_yearly_storms(year_storms)
            logging.info(f"Applied enhanced termination processing for {year}. Storm count: {year_storms['SID'].nunique()}")
    
        if not year_storms.empty:
            if target_storm_count is not None:
                all_storms.append(year_storms)
            else:
                # Original behavior: save each year immediately
                year_file = os.path.join(output_directory, f"synthetic_storms_{year}.csv")
                save_storms_to_csv(year_storms, year_file)
                
                storm_count = year_storms['SID'].nunique()
                total_storm_count += storm_count
                logging.info(f"Generated and saved {storm_count} storms for year {year}")
            
            # Free memory when not collecting all storms
            if target_storm_count is None:
                del year_storms
                import gc
                gc.collect()
    
    # Second phase: Adjust to meet exact target count if specified
    if target_storm_count is not None and all_storms:
        combined_storms = pd.concat(all_storms, ignore_index=True)
        unique_storm_ids = combined_storms['SID'].unique()
        current_count = len(unique_storm_ids)
        
        logging.info(f"Initially generated {current_count} storms, target is {target_storm_count}")
        
        # Final adjustment to match target EXACTLY
        if current_count != target_storm_count:
            if current_count > target_storm_count:
                # Too many storms, remove some
                storms_to_remove = current_count - target_storm_count
                storm_ids_to_remove = np.random.choice(unique_storm_ids, storms_to_remove, replace=False)
                combined_storms = combined_storms[~combined_storms['SID'].isin(storm_ids_to_remove)]
                logging.info(f"Removed {storms_to_remove} storms to match target count of exactly {target_storm_count}")
            
            elif current_count < target_storm_count:
                # Too few storms, generate more
                storms_to_add = target_storm_count - current_count
                logging.info(f"Need to generate {storms_to_add} additional storms to reach exactly {target_storm_count}")
                
                # Get years with the fewest storms relative to their forecast to balance distribution
                current_by_year = combined_storms.groupby('YEAR')['SID'].nunique().to_dict()
                
                # Calculate deficit for each year (how far below forecast)
                year_deficit = {}
                for year in range(start_year, end_year + 1):
                    # Get forecast count for this year
                    forecast_count = 0
                    if _df_forecast is not None and not _df_forecast.empty:
                        year_forecast = _df_forecast[_df_forecast['YEAR'] == year]
                        if not year_forecast.empty and 'Storm_Count' in year_forecast.columns:
                            forecast_count = int(round(year_forecast['Storm_Count'].iloc[0]))
                    
                    # Get current count for this year
                    current_count = current_by_year.get(year, 0)
                    
                    # Calculate deficit
                    deficit = max(0, forecast_count - current_count)
                    if deficit > 0:
                        year_deficit[year] = deficit
                
                # If we have deficits, use them to guide where to add storms
                if year_deficit:
                    # Prioritize years with biggest deficits
                    deficit_years = sorted(year_deficit.keys(), key=lambda y: year_deficit[y], reverse=True)
                    
                    # Cycle through deficit years
                    years_to_add = []
                    for year in deficit_years:
                        # Add this year deficit times
                        years_to_add.extend([year] * year_deficit[year])
                    
                    # If we need more years, repeat the cycle
                    while len(years_to_add) < storms_to_add:
                        for year in deficit_years:
                            if len(years_to_add) < storms_to_add:
                                years_to_add.append(year)
                            else:
                                break
                    
                    # Trim if needed
                    years_to_add = years_to_add[:storms_to_add]
                else:
                    # Fall back to distributing among all years proportionally to forecast
                    year_weights = {}
                    for year in range(start_year, end_year + 1):
                        if _df_forecast is not None and not _df_forecast.empty:
                            year_forecast = _df_forecast[_df_forecast['YEAR'] == year]
                            if not year_forecast.empty and 'Storm_Count' in year_forecast.columns:
                                year_weights[year] = int(round(year_forecast['Storm_Count'].iloc[0]))
                            else:
                                year_weights[year] = 18  # Default
                        else:
                            year_weights[year] = 18  # Default
                    
                    # Normalize weights
                    total_weight = sum(year_weights.values())
                    for year in year_weights:
                        year_weights[year] = year_weights[year] / total_weight
                    
                    # Select years based on weights
                    years = list(year_weights.keys())
                    weights = [year_weights[y] for y in years]
                    years_to_add = np.random.choice(years, size=storms_to_add, p=weights).tolist()
                
                # Generate the additional storms
                additional_storms = []
                used_ids = set(combined_storms['SID'])
                for year in years_to_add:
                    # Use peak months for better chances of entering PAR
                    month = np.random.choice([7, 8, 9, 10])
                    
                    # Try to generate a storm that enters PAR
                    attempts = 0
                    while attempts < 10:
                        try:
                            storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, used_ids)
                            # Check if the storm enters PAR
                            in_par = False
                            for _, row in storm.iterrows():
                                if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                                    in_par = True
                                    break
                            
                            if in_par:
                                additional_storms.append(storm)
                                used_ids.add(storm['SID'].iloc[0])
                                break
                        except Exception as e:
                            logging.error(f"Error generating additional storm for {year}-{month}: {e}")
                        
                        attempts += 1
                
                if additional_storms:
                    combined_storms = pd.concat([combined_storms] + additional_storms, ignore_index=True)
                    final_count = combined_storms['SID'].nunique()
                    logging.info(f"Added {len(additional_storms)} storms, final count: {final_count}")
        
        # Save each year separately
        for year in range(start_year, end_year + 1):
            year_storms = combined_storms[combined_storms['YEAR'] == year]
            if not year_storms.empty:
                year_file = os.path.join(output_directory, f"synthetic_storms_{year}.csv")
                save_storms_to_csv(year_storms, year_file)
        
        total_storm_count = combined_storms['SID'].nunique()
        
        # Free memory
        del combined_storms, all_storms
        import gc
        gc.collect()
    
    logging.info(f"Generated {total_storm_count} storms across {end_year-start_year+1} years")
    return total_storm_count

def storm_passes_through_par(storm_data):
    """
    Check if a storm passes through the PAR at any point in its lifetime.
    
    Args:
        storm_data: DataFrame with storm track positions
        
    Returns:
        bool: True if the storm passes through PAR, False otherwise
    """
    if storm_data is None or storm_data.empty:
        return False
        
    for _, row in storm_data.iterrows():
        if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
            return True
    
    return False

def generate_additional_par_storms(count, start_year, end_year, existing_ids=None):
    """
    Generate additional storms that are guaranteed to pass through PAR.
    
    Args:
        count: Number of additional storms needed
        start_year: Start year for generation
        end_year: End year for generation
        existing_ids: Set of existing storm IDs to avoid duplicates
        
    Returns:
        List of DataFrames with generated storms
    """
    if existing_ids is None:
        existing_ids = set()
        
    additional_storms = []
    attempts = 0
    max_attempts = count * 5  # Allow up to 5 attempts per needed storm
    
    while len(additional_storms) < count and attempts < max_attempts:
        attempts += 1
        
        # Use strategic genesis parameters to increase PAR hit probability
        year = np.random.randint(start_year, end_year + 1)
        month = np.random.choice([7, 8, 9, 10, 6, 11])  # Peak and shoulder months
        
        try:
            # Generate a storm
            storm = safe_storm_generation(year, month, existing_ids, max_attempts=3)
            if storm is None:
                continue
            
            # Check if it passes through PAR
            passes_par = False
            for _, row in storm.iterrows():
                if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                    passes_par = True
                    break
            
            if passes_par:
                # Apply basic realism checks
                lon_span = storm['LON'].max() - storm['LON'].min()
                lat_span = storm['LAT'].max() - storm['LAT'].min()
                
                if not (lon_span > 10 and lat_span < 2):  # Not purely zonal
                    additional_storms.append(storm)
                    existing_ids.add(storm['SID'].iloc[0])
                    
                    if len(additional_storms) % 10 == 0:
                        logging.info(f"Generated {len(additional_storms)}/{count} additional PAR-passing storms")
        except Exception as e:
            if attempts % 50 == 0:
                logging.error(f"Error generating additional storm ({attempts} attempts): {e}")
    
    if len(additional_storms) < count:
        logging.warning(f"Could only generate {len(additional_storms)}/{count} additional PAR-passing storms after {attempts} attempts")
    
    return additional_storms

def plot_storm_tracks(storms_df, output_path=None, show_plot=True):
    """
    Create a visualization of storm tracks with segments colored by intensity category.
    
    Args:
        storms_df: DataFrame with storm tracks
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        None
    """
    plt.figure(figsize=(15, 10))
    
    # Set up map projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    
    # Define colors for different storm categories
    color_dict = {
        'Remnant Low': 'green',  # Added new category
        'Tropical Depression': 'blue',
        'Tropical Storm': 'yellow',
        'Severe Tropical Storm': 'orange',
        'Typhoon': 'red',
        'Super Typhoon': 'purple'
    }
    
    # Draw the PAR boundary
    par_coords = list(zip(*PAR_BOUNDS)) + list(zip(*[PAR_BOUNDS[0]]))  # Close the polygon
    ax.plot(par_coords[1], par_coords[0], 'k-', linewidth=1.5, transform=ccrs.PlateCarree())
    
    # Function to determine category from wind speed
    def get_category(wind_speed):
        if wind_speed >= 100:
            return 'Super Typhoon'
        elif wind_speed >= 64:
            return 'Typhoon'
        elif wind_speed >= 48:
            return 'Severe Tropical Storm'
        elif wind_speed >= 34:
            return 'Tropical Storm'
        elif wind_speed >= 22:
            return 'Tropical Depression'
        else:
            return 'Remnant Low'
    
    # Plot all storm tracks segment by segment
    for storm_id in storms_df['SID'].unique():
        storm_data = storms_df[storms_df['SID'] == storm_id]
        
        # Sort by time to ensure correct ordering
        if 'ISO_TIME' in storm_data.columns:
            storm_data = storm_data.sort_values('ISO_TIME')
        
        # Get wind column
        wind_col = 'WIND' if 'WIND' in storm_data.columns else 'TOK_WIND'
        
        # Pre-check for two strikes rule before plotting
        valid_point_indices = []
        consecutive_low_count = 0

        for i in range(len(storm_data)):
            wind_speed = storm_data.iloc[i][wind_col]
    
            if wind_speed < 22.0:
                consecutive_low_count += 1
                # Add this point to valid indices
                valid_point_indices.append(i)
        
                # Check if we've hit two consecutive low points
                if consecutive_low_count >= 2:
                    # We've reached our limit, so stop checking more points
                    break
            else:
                consecutive_low_count = 0  # Reset counter
                valid_point_indices.append(i)

        # If there are valid points to plot, continue
        if len(valid_point_indices) < 2:  # Need at least 2 points to draw a line
            continue
    
        # Plot only the valid segments
        for j in range(len(valid_point_indices) - 1):
            i1 = valid_point_indices[j]
            i2 = valid_point_indices[j+1]
    
            # Only plot consecutive points
            if i2 != i1 + 1:
                continue
        
            wind_speed = storm_data.iloc[i1][wind_col]
            category = get_category(wind_speed)
    
            # Plot segment with color based on current intensity
            ax.plot([storm_data.iloc[i1]['LON'], storm_data.iloc[i2]['LON']], 
                    [storm_data.iloc[i1]['LAT'], storm_data.iloc[i2]['LAT']], 
                    color=color_dict.get(category, 'gray'), 
                    linewidth=1, alpha=0.6,
                    transform=ccrs.PlateCarree())

        # Plot genesis point with marker
        genesis_wind = storm_data.iloc[0][wind_col]
        genesis_category = get_category(genesis_wind)
        ax.plot(storm_data['LON'].iloc[0], storm_data['LAT'].iloc[0], 
                'o', color=color_dict.get(genesis_category, 'gray'), 
                markersize=4, alpha=0.8,
                transform=ccrs.PlateCarree())
    
    # Set map extent to PAR
    ax.set_extent([114, 142, 4, 26])
    
    # Add title and legend
    plt.title('Synthetic Tropical Cyclone Tracks in Philippine Area of Responsibility')
    
    # Create custom legend
    legend_elements = [Line2D([0], [0], color=color, lw=2, label=cat) 
                      for cat, color in color_dict.items()]
    
    ax.legend(handles=legend_elements, loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add context info
    plt.annotate(f"Total storms: {storms_df['SID'].nunique()}\n" +
                f"Date generated: {datetime.now().strftime('%Y-%m-%d')}", 
                xy=(0.02, 0.02), xycoords='figure fraction',
                fontsize=10)
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Plot saved to {output_path}")
    
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_monthly_distribution(storms_df, output_path=None, show_plot=True):
    """
    Plot monthly distribution of synthetic storms.
    
    Args:
        storms_df: DataFrame with storm tracks
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        None
    """
    # Group by month and count unique storms
    monthly_counts = storms_df.groupby([storms_df['MONTH']])['SID'].nunique()
    
    # Ensure all months are present
    all_months = pd.Series(0, index=range(1, 13))
    monthly_counts = monthly_counts.add(all_months, fill_value=0)
    
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    plt.bar(months, monthly_counts, color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Number of Storms')
    plt.title('Monthly Distribution of Synthetic Tropical Cyclones')
    plt.xticks(months, month_names)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add counts as text
    for i, count in enumerate(monthly_counts):
        plt.text(i+1, count+0.5, str(int(count)), ha='center')
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Monthly distribution plot saved to {output_path}")
    
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_intensity_distribution(storms_df, output_path=None, show_plot=True):
    """
    Plot intensity distribution of synthetic storms with color-coded categories.
    
    Parameters:
    -----------
    storms_df : pandas DataFrame
        DataFrame containing synthetic storm data
    output_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import logging
    
    # Define storm category colors with ESRI color scheme
    CATEGORY_COLORS = {
        'TD': '#4E96CE',      # Tropical Depression (22-33 kt)
        'TS': '#FFDA00',      # Tropical Storm (34-47 kt)
        'STS': '#F5821F',     # Severe Tropical Storm (48-63 kt)
        'TY': '#C03A38',      # Typhoon (64-99 kt)
        'STY': '#834696'      # Super Typhoon (>=100 kt)
    }
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Get wind speeds
    wind_col = 'WIND' if 'WIND' in storms_df.columns else 'TOK_WIND'
    wind_data = storms_df[wind_col][storms_df[wind_col] >= 22]  # Filter to include only tropical cyclone values (TD and above)
    
    # Create histogram with integer-aligned bins
    bins = np.arange(21.5, 126.5, 1)  # Creates bins centered on integers 22,23,24...126
    n, bins, patches = plt.hist(
        wind_data,             # CHANGED: Use filtered data instead of storms_df[wind_col]
        bins=30,             # Use integer-aligned bins
        edgecolor='black', 
        alpha=1.0
    )
    
    # Highlight the previously problematic range
    plt.axvspan(115, 120, color='lightgray', alpha=0.2, zorder=0)
    
    # Function to determine the correct color based on bin range
    def get_bin_color(bin_min, bin_max):
        """Get the appropriate color for a bin based on its range"""
        # Check if bin is entirely in one category
        if bin_max <= 33:
            return CATEGORY_COLORS['TD']
        elif bin_min >= 34 and bin_max <= 47:
            return CATEGORY_COLORS['TS']
        elif bin_min >= 48 and bin_max <= 63:
            return CATEGORY_COLORS['STS']
        elif bin_min >= 64 and bin_max <= 99:
            return CATEGORY_COLORS['TY']
        elif bin_min >= 100:
            return CATEGORY_COLORS['STY']
        
        # For bins that span categories, use the category that contains the bin center
        bin_center = (bin_min + bin_max) / 2
        if bin_center <= 33:
            return CATEGORY_COLORS['TD']
        elif bin_center <= 47:
            return CATEGORY_COLORS['TS']
        elif bin_center <= 63:
            return CATEGORY_COLORS['STS']
        elif bin_center <= 99:
            return CATEGORY_COLORS['TY']
        else:
            return CATEGORY_COLORS['STY']
    
    # Color the histogram bars based on their bin ranges
    for i, patch in enumerate(patches):
        bin_min = bins[i]
        bin_max = bins[i+1]
        color = get_bin_color(bin_min, bin_max)
        patch.set_facecolor(color)
    
    # Explicitly set the x-axis limits
    plt.xlim(20, 130)
    
    # Set labels and title
    plt.xlabel('Wind Speed (knots)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Intensity Distribution of Synthetic Tropical Cyclones', fontsize=14)
    
    # Add vertical lines at category boundaries with improved styling
    #plt.axvline(x=33, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    #plt.axvline(x=47, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    #plt.axvline(x=63, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    #plt.axvline(x=99, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    for boundary in [33, 47, 63, 99]:
        plt.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    # Create custom legend for storm categories
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS['TD'], label='TD (22-33 kt)'),
        mpatches.Patch(color=CATEGORY_COLORS['TS'], label='TS (34-47 kt)'),
        mpatches.Patch(color=CATEGORY_COLORS['STS'], label='STS (48-63 kt)'),
        mpatches.Patch(color=CATEGORY_COLORS['TY'], label='TY (64-99 kt)'),
        mpatches.Patch(color=CATEGORY_COLORS['STY'], label='STY (>=100 kt)')
    ]
    plt.legend(handles=legend_patches, title='Storm Category', loc='upper right')
    
    # Add grid
    #plt.grid(linestyle='--', alpha=0.4)
    plt.grid(True, linestyle=':', alpha=0.4)
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Color-coded intensity distribution plot saved to {output_path}")
    
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_storm_density_map(storms_df, output_path=None, show_plot=True):
    """
    Create a density map of storm positions.
    
    Args:
        storms_df: DataFrame with storm tracks
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        
    Returns:
        None
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Set up map projection
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.COASTLINE)
        
        # Draw the PAR boundary
        par_coords = list(zip(*PAR_BOUNDS)) + list(zip(*[PAR_BOUNDS[0]]))  # Close the polygon
        ax.plot(par_coords[1], par_coords[0], 'k-', linewidth=1.5, transform=ccrs.PlateCarree())
        
        # Create 2D histogram
        lon_range = (110, 145)
        lat_range = (0, 30)
        
        # Filter points to those within the range
        valid_points = storms_df[(storms_df['LON'] >= lon_range[0]) & (storms_df['LON'] <= lon_range[1]) &
                               (storms_df['LAT'] >= lat_range[0]) & (storms_df['LAT'] <= lat_range[1])]
        
        if len(valid_points) > 0:
            # Adjusted bin calculation to ensure correct dimensions
            lat_bins = np.linspace(lat_range[0], lat_range[1], 31)
            lon_bins = np.linspace(lon_range[0], lon_range[1], 36)
            
            h, xedges, yedges = np.histogram2d(
                valid_points['LON'], valid_points['LAT'],
                bins=[lon_bins, lat_bins]
            )
            
            # Normalize and smooth
            h = h / h.max() if h.max() > 0 else h
            
            # Plot density map
            pcm = ax.pcolormesh(
                xedges, yedges, h.T, 
                cmap='viridis', 
                norm=colors.PowerNorm(gamma=0.5),  # Adjust gamma to highlight lower-density areas
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                shading='flat'  # Explicitly set shading
            )
            
            # Add colorbar
            cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02)
            cbar.set_label('Normalized Density')
        else:
            logging.warning("No valid storm points found for density map")
            plt.text(0.5, 0.5, 'No Storm Data', 
                     horizontalalignment='center', 
                     verticalalignment='center', 
                     transform=ax.transAxes)
        
        # Set map extent
        ax.set_extent([110, 145, 0, 30])
        
        # Add title and grid
        plt.title('Density Map of Synthetic Tropical Cyclones')
        
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
        gl.top_labels = False
        gl.right_labels = False
        
        # Save figure if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Density map saved to {output_path}")
        
        # Show or close
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        logging.error(f"Error creating storm density map: {e}")
        plt.close()

def verify_plot_termination_rules(storms_df):
    """
    Verify that the plot_storm_tracks function's termination rules match generation termination rules.
    This function analyzes storm tracks to check for consistent application of termination criteria.
    
    Args:
        storms_df: DataFrame with storm tracks
        
    Returns:
        Dictionary with verification results
    """
    verification_results = {
        "total_storms": 0,
        "proper_termination": 0,
        "improper_termination": 0,
        "boundary_termination": 0,
        "termination_details": []
    }
    
    wind_col = 'WIND' if 'WIND' in storms_df.columns else 'TOK_WIND'
    verification_results["total_storms"] = storms_df['SID'].nunique()
    
    for storm_id in storms_df['SID'].unique():
        storm_data = storms_df[storms_df['SID'] == storm_id]
        
        # Sort by time to ensure correct ordering
        if 'ISO_TIME' in storm_data.columns:
            storm_data = storm_data.sort_values('ISO_TIME')
        
        # Check termination condition
        final_points = storm_data.iloc[-2:] if len(storm_data) >= 2 else storm_data
        final_winds = final_points[wind_col].values
        
        # Check for boundary termination
        final_point = storm_data.iloc[-1]
        if (final_point['LON'] > 140 or final_point['LON'] < 115 or 
            final_point['LAT'] > 25 or final_point['LAT'] < 5):
            verification_results["boundary_termination"] += 1
            termination_type = "boundary"
        # If last two points are below 22kt, proper termination
        elif len(final_winds) >= 2 and all(wind < 22.0 for wind in final_winds):
            verification_results["proper_termination"] += 1
            termination_type = "proper"
        # If single final point is below 22kt (possible one-strike post-landfall)
        elif final_winds[-1] < 22.0:
            # Check if over land
            if 'ELEVATION' in final_point and final_point['ELEVATION'] > 0.5:
                verification_results["proper_termination"] += 1
                termination_type = "one-strike"
            else:
                verification_results["improper_termination"] += 1
                termination_type = "incomplete"
        else:
            verification_results["improper_termination"] += 1
            termination_type = "premature"
        
        # Store detailed information for this storm
        verification_results["termination_details"].append({
            "storm_id": storm_id,
            "final_wind": final_winds[-1],
            "termination_type": termination_type,
            "final_lat": final_point['LAT'],
            "final_lon": final_point['LON']
        })
    
    # Calculate percentages
    total = verification_results["total_storms"]
    for key in ["proper_termination", "improper_termination", "boundary_termination"]:
        verification_results[f"{key}_percent"] = (verification_results[key] / total * 100) if total > 0 else 0
    
    return verification_results
    
def save_storms_to_csv(storms_df, output_path):
    """
    Save synthetic storms to CSV file.
    
    Args:
        storms_df: DataFrame with storm tracks
        output_path: Path to save the CSV file
        
    Returns:
        None
    """
    # Make a copy to avoid modifying the original
    output_df = storms_df.copy()
    
    # NOW convert winds to integers for final output
    if 'WIND' in output_df.columns:
        output_df['WIND'] = np.round(output_df['WIND']).astype(np.int32)
        
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    storms_df.to_csv(output_path, index=False)
    logging.info(f"Storms saved to {output_path}")

def check_input_files():
    """Enhanced file validation with content checks"""
    
    required_files = {
        POSITIONS_PATH: {
            'required_columns': ['SID', 'SEASON', 'LAT', 'LON', 'ISO_TIME'],
            'min_rows': 1000
        },
        WIND_PATH: {
            'required_columns': ['TOK_WIND'],
            'min_rows': 100
        },
        FORECAST_PATH: {
            'required_columns': ['YEAR', 'Storm_Count'],
            'min_rows': 10
        }
    }
    
    for file_path, requirements in required_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path, nrows=1)  # Read header only first
            missing_cols = set(requirements['required_columns']) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns in {file_path}: {missing_cols}")
            
            # Check minimum row count
            full_df = pd.read_csv(file_path)
            if len(full_df) < requirements['min_rows']:
                raise ValueError(f"Insufficient data in {file_path}: {len(full_df)} < {requirements['min_rows']}")
                
        except Exception as e:
            raise ValueError(f"Error validating {file_path}: {e}")
    
    return True

def combine_yearly_files(directory, output_path):
    """
    Combine all yearly storm CSV files in a directory into a single file.
    
    Args:
        directory: Directory containing yearly storm files
        output_path: Path for the combined output file
    """
    import glob
    yearly_files = glob.glob(os.path.join(directory, "synthetic_storms_*.csv"))
    
    if not yearly_files:
        logging.warning("No yearly files found to combine")
        return
    
    # Read and combine all files
    all_data = []
    for file in yearly_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                all_data.append(df)
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        logging.info(f"Combined {len(yearly_files)} files into {output_path}")
    else:
        logging.error("No valid data found to combine")

def generate_storm_ensemble(start_year, end_year, ensemble_size=50, base_seed=42, target_storms=1824):
    """
    Generate an ensemble of synthetic storm sets for the specified period.
    Enhanced to guarantee exact storm counts for each ensemble member.
    
    Args:
        start_year: First year to generate storms for
        end_year: Last year to generate storms for
        ensemble_size: Number of ensemble members to generate
        base_seed: Base random seed to derive ensemble member seeds
        target_storms: Target number of storms per ensemble member
        
    Returns:
        Dictionary with paths to ensemble member files
    """
    # Add seed control to generators if not already done
    add_seed_control_to_generators()
    
    ensemble_results = {}
    
    # Run through each ensemble member
    for ensemble_id in range(1, ensemble_size+1):
        # Create unique seed for this ensemble member
        member_seed = base_seed + ensemble_id * 1000
        
        # Create output directory for this ensemble member
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_dir = os.path.join(OUTPUT_DIR, "ENSEMBLE", f"ensemble_member_{ensemble_id}_{timestamp}")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        logging.info(f"Generating ensemble member {ensemble_id}/{ensemble_size} with seed {member_seed}")
        
        all_storms = []
        used_ids = set()  # Track used storm IDs across all years
        
        # First phase: Generate storms based on yearly forecasts
        for year in range(start_year, end_year + 1):
            # Use a deterministic seed derived from member seed and year
            year_seed = member_seed + (year - start_year)
            
            logging.info(f"Processing year {year} for ensemble member {ensemble_id}...")
            
            # Generate storms with the seed
            year_storms = generate_yearly_storms(year, None, True, random_seed=year_seed)
            
            # Apply post-processing with the same seed
            np.random.seed(year_seed)
            random.seed(year_seed)
            year_storms = post_process_synthetic_tracks(year_storms, year)
            
            # Add termination processing here
            if not year_storms.empty:
                year_storms = process_yearly_storms(year_storms)
                logging.info(f"Applied enhanced termination processing for ensemble {ensemble_id}, year {year}")
            
            # Filter with the same seed
            np.random.seed(year_seed)
            random.seed(year_seed) 
            year_storms = filter_storms_for_realism(year_storms, min_par_points=1)
            
            if not year_storms.empty:
                # Save year file
                year_file = os.path.join(ensemble_dir, f"synthetic_storms_{year}.csv")
                save_storms_to_csv(year_storms, year_file)
                
                # Add to collection and track storm IDs
                all_storms.append(year_storms)
                for sid in year_storms['SID'].unique():
                    used_ids.add(sid)
                
                storm_count = year_storms['SID'].nunique()
                logging.info(f"Generated {storm_count} storms for year {year} in ensemble {ensemble_id}")
        
        # Combine all storms for this ensemble member
        if all_storms:
            combined_storms = pd.concat(all_storms, ignore_index=True)
            
            # Apply final wind scaling to achieve exact target mean
            combined_storms = scale_winds_to_target_mean(combined_storms, target_mean=58.0)
    
            # Compensate for land losses
            combined_storms = compensate_for_land_losses(combined_storms, target_sty_percent=5.49)
    
            # CRITICAL: Final validation of wind speeds before saving
            if 'WIND' in combined_storms.columns:
                # Create a temporary copy to avoid SettingWithCopyWarning
                wind_copy = combined_storms['WIND'].values
                validated_winds = validate_wind_speeds(wind_copy, caller="generate_storm_ensemble-final")
                combined_storms['WIND'] = validated_winds
            
                # Report any extreme values
                if np.any(wind_copy > 200):
                    extreme_count = np.sum(wind_copy > 200)
                    extreme_max = np.max(wind_copy)
                    extreme_indices = np.where(wind_copy > 200)[0]
                    logging.error(f"CRITICAL: Found {extreme_count} extreme wind values before final validation! Max: {extreme_max}")
                
                    # Optional: Log detailed information for debugging
                    if len(extreme_indices) > 0:
                        for idx in extreme_indices[:5]:  # Log details for first 5 extreme values
                            row = combined_storms.iloc[idx]
                            logging.error(f"Extreme wind at index {idx}: SID={row['SID']}, "
                                         f"WIND={row['WIND']}, LAT={row['LAT']}, LON={row['LON']}")
                                     
            total_count = combined_storms['SID'].nunique()
            
            # ADD HERE: Generate monthly distribution plot
            try:
                # Create monthly distribution plot for this ensemble member
                monthly_plot_path = os.path.join(ensemble_dir, f"monthly_distribution_ensemble_{ensemble_id}.png")
                plot_monthly_distribution(combined_storms, monthly_plot_path, show_plot=False)
                logging.info(f"Generated monthly distribution plot for ensemble {ensemble_id}")
            except Exception as e:
                logging.error(f"Error creating monthly distribution plot for ensemble {ensemble_id}: {e}")
            
            # Adjust to exact target count if needed
            if total_count != target_storms:
                if total_count > target_storms:
                    # Too many storms, remove some
                    storms_to_remove = total_count - target_storms
                    all_sids = combined_storms['SID'].unique()
                    np.random.seed(member_seed)  # Ensure reproducibility
                    sids_to_remove = np.random.choice(all_sids, storms_to_remove, replace=False)
                    combined_storms = combined_storms[~combined_storms['SID'].isin(sids_to_remove)]
                    logging.info(f"Removed {storms_to_remove} excess storms for ensemble {ensemble_id}")
                else:
                    # Too few storms, keep generating until we reach exactly the target count
                    storms_to_add = target_storms - total_count
                    logging.info(f"Need to generate {storms_to_add} additional storms for ensemble {ensemble_id}")
                    
                    # Try harder to generate the missing storms with increased attempts
                    max_patching_attempts = storms_to_add * 10  # Much more aggressive attempts
                    patching_attempts = 0
                    
                    while combined_storms['SID'].nunique() < target_storms and patching_attempts < max_patching_attempts:
                        # Cycle through years to distribute the additional storms
                        year = start_year + (patching_attempts % (end_year - start_year + 1))
                        # Use peak months for better PAR entry chances
                        month = np.random.choice([7, 8, 9, 10, 6, 11])
                        
                        # Generate a storm with eastern genesis to increase PAR hit probability
                        patching_seed = member_seed + 50000 + patching_attempts  # Unique seed for each attempt
                        np.random.seed(patching_seed)
                        random.seed(patching_seed)
                        
                        try:
                            # Create a storm with eastern genesis bias for higher PAR entry chance
                            storm = safe_storm_generation(year, month, used_ids, max_attempts=3)
                            if storm is None:
                                continue
                            
                            # Check if it passes through PAR
                            par_points = 0
                            for _, row in storm.iterrows():
                                if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                                    par_points += 1
                                    if par_points >= 1:  # Only need 1 point in PAR
                                        break
                            
                            if par_points > 0:
                                # Add the storm
                                combined_storms = pd.concat([combined_storms, storm], ignore_index=True)
                                used_ids.add(storm['SID'].iloc[0])
                                
                                # Log progress periodically
                                if combined_storms['SID'].nunique() % 10 == 0:
                                    current_count = combined_storms['SID'].nunique()
                                    logging.info(f"Now at {current_count}/{target_storms} storms for ensemble {ensemble_id}")
                        except Exception as e:
                            if patching_attempts % 50 == 0:
                                logging.error(f"Error generating additional storm: {e}")
                        
                        patching_attempts += 1
                    
                    # Final count check
                    final_count = combined_storms['SID'].nunique()
                    if final_count < target_storms:
                        # If still short after all attempts, duplicate some existing storms with new IDs
                        # as a last resort to reach the exact target count
                        still_missing = target_storms - final_count
                        logging.warning(f"After {patching_attempts} attempts, still short {still_missing} storms. Using duplicates with new IDs.")
                        
                        # Get all existing complete storms
                        all_storm_ids = combined_storms['SID'].unique()
                        
                        for i in range(still_missing):
                            # Select a random storm to duplicate
                            source_id = np.random.choice(all_storm_ids)
                            source_storm = combined_storms[combined_storms['SID'] == source_id].copy()
                            
                            # Create a new unique ID
                            new_id = f"SYN_PATCH_{ensemble_id}_{i+1:03d}"
                            while new_id in used_ids:
                                new_id = f"SYN_PATCH_{ensemble_id}_{np.random.randint(1, 999):03d}"
                            
                            # Replace the ID and add minor position variations
                            source_storm['SID'] = new_id
                            # Add small random variations to positions to make it unique
                            jitter = 0.05  # Small positional jitter in degrees
                            source_storm['LAT'] = source_storm['LAT'] + np.random.uniform(-jitter, jitter, len(source_storm))
                            source_storm['LON'] = source_storm['LON'] + np.random.uniform(-jitter, jitter, len(source_storm))
                            
                            # Add to the collection
                            combined_storms = pd.concat([combined_storms, source_storm], ignore_index=True)
                            used_ids.add(new_id)
                        
                        logging.info(f"Added {still_missing} patched storms to reach exactly {target_storms} storms")
            
            # Save combined file
            combined_path = os.path.join(ensemble_dir, f"synthetic_storms_{start_year}_{end_year}.csv")
            save_storms_to_csv(combined_storms, combined_path)
            
            # Verify final count
            final_count = combined_storms['SID'].nunique()
            if final_count != target_storms:
                logging.error(f"ERROR: Ensemble {ensemble_id} final count {final_count} does not match target {target_storms}!")
                logging.error(f"Shortfall: {target_storms - final_count} storms")
            else:
                logging.info(f"Successfully generated exactly {target_storms} storms for ensemble {ensemble_id}")
            
            # Always report the actual count
            logging.info(f"ACTUAL STORM COUNT for ensemble {ensemble_id}: {final_count} storms")
            
            # Comprehensive track termination and boundary diagnostics
            logging.info(f"Performing diagnostic analysis for ensemble {ensemble_id}")

            # 1. Analyze track termination categories
            termination_categories = {}
            for storm_id in combined_storms['SID'].unique():
                storm_data = combined_storms[combined_storms['SID'] == storm_id]
                final_category = storm_data.iloc[-1]['CATEGORY']
                termination_categories[final_category] = termination_categories.get(final_category, 0) + 1

            total_storms = len(combined_storms['SID'].unique())
            logging.info("Track termination statistics:")
            for category, count in sorted(termination_categories.items()):
                percentage = (count / total_storms) * 100
                logging.info(f"  {category}: {count} tracks ({percentage:.1f}%)")

            # 2. Analyze geographic termination patterns
            boundary_terminations = 0
            land_terminations = 0
            open_water_terminations = 0

            for storm_id in combined_storms['SID'].unique():
                storm_data = combined_storms[combined_storms['SID'] == storm_id]
                final_point = storm_data.iloc[-1]
    
                # Check domain boundary termination
                if (final_point['LON'] > 140 or final_point['LON'] < 115 or 
                    final_point['LAT'] > 25 or final_point['LAT'] < 5):
                    boundary_terminations += 1
                # Check land termination
                elif 'ELEVATION' in final_point and final_point['ELEVATION'] > 0.5:
                    land_terminations += 1
                else:
                    open_water_terminations += 1

            logging.info("Geographic termination statistics:")
            logging.info(f"  Domain boundary: {boundary_terminations} storms ({(boundary_terminations/total_storms)*100:.1f}%)")
            logging.info(f"  Over land: {land_terminations} storms ({(land_terminations/total_storms)*100:.1f}%)")
            logging.info(f"  Open water: {open_water_terminations} storms ({(open_water_terminations/total_storms)*100:.1f}%)")

            # 3. Analyze final wind speed distribution
            final_winds = []
            for storm_id in combined_storms['SID'].unique():
                storm_data = combined_storms[combined_storms['SID'] == storm_id]
                final_winds.append(storm_data.iloc[-1]['WIND'])

            wind_stats = {
                "min": min(final_winds),
                "max": max(final_winds),
                "mean": sum(final_winds) / len(final_winds),
                "median": sorted(final_winds)[len(final_winds) // 2],
                "below_22kt": sum(1 for w in final_winds if w < 22.0) / len(final_winds) * 100
            }

            logging.info("Final wind speed statistics:")
            logging.info(f"  Minimum: {wind_stats['min']:.1f} knots")
            logging.info(f"  Maximum: {wind_stats['max']:.1f} knots")
            logging.info(f"  Mean: {wind_stats['mean']:.1f} knots")
            logging.info(f"  Median: {wind_stats['median']:.1f} knots")
            logging.info(f"  Percent below 22kt: {wind_stats['below_22kt']:.1f}%")

            # 4. Analyze track segment intensity distribution
            intensity_counts = {
                "Remnant Low": 0,
                "Tropical Depression": 0, 
                "Tropical Storm": 0,
                "Severe Tropical Storm": 0,
                "Typhoon": 0,
                "Super Typhoon": 0
            }

            for _, row in combined_storms.iterrows():
                if row['WIND'] < 22.0:
                    intensity_counts["Remnant Low"] += 1
                elif row['WIND'] < 34.0:
                    intensity_counts["Tropical Depression"] += 1
                elif row['WIND'] < 48.0:
                    intensity_counts["Tropical Storm"] += 1
                elif row['WIND'] < 64.0:
                    intensity_counts["Severe Tropical Storm"] += 1
                elif row['WIND'] < 100.0:
                    intensity_counts["Typhoon"] += 1
                else:
                    intensity_counts["Super Typhoon"] += 1

            total_points = len(combined_storms)
            logging.info("Track segment intensity distribution:")
            for category, count in intensity_counts.items():
                percentage = (count / total_points) * 100
                logging.info(f"  {category}: {count} points ({percentage:.1f}%)")
                
            # Verify plot termination rules
            logging.info(f"Verifying ensemble {ensemble_id} plot termination rules...")
            verification_results = verify_plot_termination_rules(combined_storms)

            # Log verification results
            logging.info(f"Termination rule verification:")
            logging.info(f"  Proper termination: {verification_results['proper_termination']} storms ({verification_results['proper_termination_percent']:.1f}%)")
            logging.info(f"  Improper termination: {verification_results['improper_termination']} storms ({verification_results['improper_termination_percent']:.1f}%)")
            logging.info(f"  Boundary termination: {verification_results['boundary_termination']} storms ({verification_results['boundary_termination_percent']:.1f}%)")

            # If any improper terminations found, log details for further investigation
            if verification_results['improper_termination'] > 0:
                logging.info("Examples of improper terminations:")
                improper_examples = [d for d in verification_results['termination_details'] if d['termination_type'] == "premature"][:5]
                for example in improper_examples:
                    logging.info(f"  Storm {example['storm_id']}: final wind {example['final_wind']:.1f}kt at ({example['final_lat']:.2f}N, {example['final_lon']:.2f}E)")

            # Create visualizations
            try:
                # Plot tracks for this ensemble member
                plot_path = os.path.join(ensemble_dir, f"storm_tracks_ensemble_{ensemble_id}.png")
                plot_storm_tracks(combined_storms, plot_path, show_plot=False)
    
                # Generate intensity distribution plot for this ensemble member
                intensity_plot_path = os.path.join(ensemble_dir, f"intensity_distribution_ensemble_{ensemble_id}.png")
                plot_intensity_distribution(combined_storms, intensity_plot_path, show_plot=False)
    
            except Exception as e:
                logging.error(f"Error creating visualizations for ensemble {ensemble_id}: {e}")

            ensemble_results[ensemble_id] = combined_path
            
            # After completing the first ensemble member, ask if we should continue
            if ensemble_id == 1:
                logging.info("First ensemble member completed.")
                print("\n")  # Add some space for visibility
                print("="*80)
                print(f"First ensemble member (ID: {ensemble_id}) completed and saved to:")
                print(f"  {combined_path}")
                print("="*80)
                
                while True:
                    try:
                        response = input("\nContinue with remaining ensemble members? (Y/N): ").strip().upper()
                        if response == 'Y':
                            logging.info("Continuing with remaining ensemble members...")
                            break  # Exit the prompt loop and continue
                        elif response == 'N':
                            logging.info("Stopping after first ensemble member as requested.")
                            return ensemble_results  # Return early with just the first result
                        else:
                            print("Please enter Y or N.")
                    except KeyboardInterrupt:
                        print("\nProcess interrupted by user.")
                        return ensemble_results
                    except Exception as e:
                        print(f"Error reading input: {e}")
                        print("Please try again.")
                        
            # Free memory
            del combined_storms, all_storms
            import gc
            gc.collect()
        else:
            logging.error(f"No storms were generated for ensemble member {ensemble_id}!")
    
    logging.info(f"Completed generation of {len(ensemble_results)} ensemble members")
    return ensemble_results

def add_seed_control_to_generators():
    """
    Modify the key generation functions to accept random seeds.
    Add this to the top of your main() function.
    """
    # Save the original function for reference
    original_generate_yearly_storms = globals()['generate_yearly_storms']
    
    # Create the enhanced version with seed control
    def seeded_generate_yearly_storms(year, total_storms=None, par_only=True, random_seed=None):
        # Set random seed if provided for reproducibility
        if random_seed is not None:
            prev_state = np.random.get_state()
            prev_random_state = random.getstate()
            
            # Set both numpy and Python random generators
            np.random.seed(random_seed)
            random.seed(random_seed)
            logging.info(f"Using random seed {random_seed} for year {year}")
            
            try:
                # Call the original function
                result = original_generate_yearly_storms(year, total_storms, par_only)
                return result
            finally:
                # Restore previous random state
                np.random.set_state(prev_state)
                random.setstate(prev_random_state)
        else:
            # Just call the original function if no seed specified
            return original_generate_yearly_storms(year, total_storms, par_only)
    
    # Replace the global function with the seeded version
    globals()['generate_yearly_storms'] = seeded_generate_yearly_storms
    logging.info("Added seed control to generator functions")

def create_ensemble_hotspot_map(ensemble_dirs, output_path=None, show_plot=True, subset_years=None):
    """
    Create a hotspot map based on multiple ensemble members.
    
    Args:
        ensemble_dirs: Dictionary mapping ensemble IDs to paths of ensemble combined files
        output_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
        subset_years: Optional tuple (start_year, end_year) to filter data by year range
        
    Returns:
        None
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    import pandas as pd
    import logging
    
    plt.figure(figsize=(15, 10))
    
    # Set up map projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    
    # Draw the PAR boundary
    par_coords = list(zip(*PAR_BOUNDS)) + list(zip(*[PAR_BOUNDS[0]]))  # Close the polygon
    ax.plot(par_coords[1], par_coords[0], 'k-', linewidth=1.5, transform=ccrs.PlateCarree())
    
    # Create 2D histogram grid
    lon_range = (110, 145)
    lat_range = (0, 30)
    
    # More bins for higher resolution
    lon_bins = np.linspace(lon_range[0], lon_range[1], 351)  # 0.1 degree resolution
    lat_bins = np.linspace(lat_range[0], lat_range[1], 301)  # 0.1 degree resolution
    
    # Create an empty grid to accumulate storm point counts
    combined_grid = np.zeros((len(lon_bins)-1, len(lat_bins)-1))
    
    # Track number of ensemble members processed
    ensemble_count = 0
    
    # Process each ensemble member
    for ensemble_id, ensemble_path in ensemble_dirs.items():
        try:
            # Load the ensemble member data
            ensemble_df = pd.read_csv(ensemble_path)
            
            # Filter by year range if specified
            if subset_years is not None:
                start_year, end_year = subset_years
                if 'YEAR' in ensemble_df.columns:
                    ensemble_df = ensemble_df[(ensemble_df['YEAR'] >= start_year) & 
                                             (ensemble_df['YEAR'] <= end_year)]
            
            # Filter points to those within the range
            valid_points = ensemble_df[(ensemble_df['LON'] >= lon_range[0]) & 
                                     (ensemble_df['LON'] <= lon_range[1]) &
                                     (ensemble_df['LAT'] >= lat_range[0]) & 
                                     (ensemble_df['LAT'] <= lat_range[1])]
            
            if not valid_points.empty:
                # Create 2D histogram for this ensemble member
                hist, _, _ = np.histogram2d(
                    valid_points['LON'], valid_points['LAT'],
                    bins=[lon_bins, lat_bins]
                )
                
                # Add to combined grid
                combined_grid += hist
                ensemble_count += 1
                logging.info(f"Processed ensemble member {ensemble_id} with {len(valid_points)} points")
            else:
                logging.warning(f"No valid points found in ensemble member {ensemble_id}")
                
        except Exception as e:
            logging.error(f"Error processing ensemble member {ensemble_id}: {e}")
    
    # If no ensemble members processed successfully, show error message
    if ensemble_count == 0:
        logging.error("No valid ensemble members processed")
        plt.text(0.5, 0.5, 'No Valid Ensemble Data', 
                 horizontalalignment='center', 
                 verticalalignment='center', 
                 transform=ax.transAxes)
    else:
        # Normalize by number of ensemble members
        combined_grid = combined_grid / ensemble_count
        
        # Apply Gaussian smoothing for a cleaner hotspot map
        from scipy.ndimage import gaussian_filter
        smoothed_grid = gaussian_filter(combined_grid, sigma=2.0)
        
        # Create mesh grid for plotting
        lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
        lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        lon_mesh, lat_mesh = np.meshgrid(lon_centers, lat_centers)
        
        # Plot hotspot map with enhanced color scheme and better normalization
        # Use a custom color normalization to highlight both common and rare areas
        norm = colors.PowerNorm(gamma=0.5, vmin=smoothed_grid.min(), vmax=smoothed_grid.max())
        
        # Plot the hotspot map
        pcm = ax.pcolormesh(
            lon_bins[:-1], lat_bins[:-1], smoothed_grid.T, 
            cmap='viridis', 
            norm=norm,
            alpha=0.7,
            transform=ccrs.PlateCarree(),
            shading='auto'
        )
        
        # Add colorbar with better formatting
        cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.02, 
                           format='%.1f', label='Average Storm Frequency')
        
        # Add contour lines for key thresholds
        # Calculate percentiles for contour levels
        levels = np.percentile(smoothed_grid[smoothed_grid > 0], 
                              [50, 75, 90, 95, 99])
        
        contour = ax.contour(
            lon_mesh, lat_mesh, smoothed_grid.T, 
            levels=levels,
            colors=['#FFFFFF'],
            linewidths=0.8,
            alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        
        # Add contour labels for key percentiles
        plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # Set map extent
    ax.set_extent([110, 145, 0, 30])
    
    # Add title and grid
    plt.title(f'Ensemble Tropical Cyclone Hotspot Map ({ensemble_count} Members)')
    
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add text with ensemble information
    plt.annotate(
        f"Based on {ensemble_count} ensemble members\n"
        f"Date created: {datetime.now().strftime('%Y-%m-%d')}",
        xy=(0.02, 0.02), xycoords='figure fraction',
        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    # Save figure if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Ensemble hotspot map saved to {output_path}")
    
    # Show or close
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return None


def run_ensemble_analysis(ensemble_results, start_year, end_year, output_dir=None):
    """
    Run comprehensive ensemble analysis on the generated ensemble members.
    
    Args:
        ensemble_results: Dictionary with paths to ensemble member files
        start_year: Start year of the simulation
        end_year: End year of the simulation
        output_dir: Directory to save analysis outputs
        
    Returns:
        None
    """
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, f"ensemble_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Running ensemble analysis, outputs will be saved to {output_dir}")
    
    # Create hotspot map for entire period
    hotspot_path = os.path.join(output_dir, f"ensemble_hotspot_map_{start_year}_{end_year}.png")
    create_ensemble_hotspot_map(ensemble_results, hotspot_path, show_plot=False)
    
    # Create decade-specific hotspot maps
    for decade_start in range(start_year, end_year, 10):
        decade_end = min(decade_start + 9, end_year)
        decade_path = os.path.join(output_dir, f"ensemble_hotspot_map_{decade_start}_{decade_end}.png")
        create_ensemble_hotspot_map(ensemble_results, decade_path, show_plot=False, 
                                   subset_years=(decade_start, decade_end))
    
    # Create category-specific hotspot maps
    # This would require extending the hotspot function to filter by storm category
    
    # Add more ensemble analysis functions here:
    # Create combined intensity distribution plot for all ensembles
    logging.info("Generating combined intensity distribution plot...")
    try:
        # Create a combined DataFrame from all ensemble members
        combined_df = []
        for ensemble_id, file_path in ensemble_results.items():
            try:
                # Load this ensemble member's data
                ensemble_df = pd.read_csv(file_path)
                combined_df.append(ensemble_df)
            except Exception as e:
                logging.error(f"Error loading ensemble {ensemble_id} for combined plot: {e}")
    
        # Concatenate all DataFrames if we have any
        if combined_df:
            all_ensembles_df = pd.concat(combined_df, ignore_index=True)
        
            # Generate the combined intensity distribution plot
            combined_plot_path = os.path.join(output_dir, "intensity_distribution_all_ensembles.png")
            plot_intensity_distribution(
                storms_df=all_ensembles_df,
                output_path=combined_plot_path,
                show_plot=False
            )
            logging.info(f"Generated combined intensity distribution for all {len(ensemble_results)} ensembles")
        
            # Free memory
            del all_ensembles_df, combined_df
            import gc
            gc.collect()
        else:
            logging.warning("No valid ensemble data to create combined plot")
    except Exception as e:
        logging.error(f"Error creating combined intensity distribution plot: {e}")
    
    # - Return period analysis
    # - Maximum intensity analysis
    # - Landfall analysis
    # - etc.
    
    logging.info("Ensemble analysis completed successfully")
    return None
    
def main():
    """
    Main function to generate synthetic storms and create visualizations.
    Can run in single-run mode or ensemble mode based on the     flag.
    """
    
    # Check if required input files exist before proceeding
    
    # Add global validation to pandas operations for wind speeds
    # This will catch issues when manipulating DataFrames
    
    original_setitem = pd.core.indexing._LocIndexer.__setitem__
    
    def validated_setitem(self, key, value):
        # When setting values in a DataFrame or Series
        if isinstance(key, str) and key == 'WIND':
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Validate wind arrays when assigning to DataFrame
                value = validate_wind_speeds(value, caller="pandas_setitem")
            elif isinstance(value, (int, float)) and value > 0:
                # Validate single wind values
                value = float(validate_wind_speeds(np.array([value]), caller="pandas_setitem_scalar")[0])
        return original_setitem(self, key, value)
    
    # Apply the monkey patch
    pd.core.indexing._LocIndexer.__setitem__ = validated_setitem
    
    if not check_input_files():
        logging.error("Required input files missing. Exiting.")
        return
        
    # Load historical data
    load_historical_data()
    
   # Initialize the DEM for storm calculations
    logging.info("Initializing ArcMap 10.2.x resampled DEM...")
    try:
        if initialize_dem(DEM_PATH):
            logging.info("DEM ready for storm calculations")
        else:
            raise Exception("Failed to initialize primary DEM")
    except Exception as e:
        logging.warning(f"Failed to load primary DEM: {e}")
        logging.info("Attempting to load fallback DEM (phl_dem.tif)...")
    
        # Try fallback DEM
        fallback_success = create_fallback_philippines_dem()
        if fallback_success:
            logging.info("Fallback DEM loaded successfully - results will have lower resolution")
        else:
            logging.warning("Fallback DEM also failed - using artificial DEM")
    
    # Access the global variable
    #global track_model
    
    # Initialize configuration
    config = StormConfig()
    config.initialize(DATA_DIR)
    
    # MAKE TRACK_MODEL GLOBAL AGAIN (temporary fix)
    global track_model
    track_model = config.track_model if config.track_model else DataDrivenTrackModel(POSITIONS_PATH)
    logging.info("Initialized data-driven track model from historical data")
    
    # Define parameters
    start_year = 2024
    end_year = 2124
    target_storms = 1824  # Exact number of storms to generate
    
    # Add ensemble parameters
    ensemble_size = 35    # Number of ensemble members to generate (i.e., ensemble_size * 101 years)
    base_seed = 42        # Base random seed for reproducibility
    
    # Add seed control to key generator functions
    add_seed_control_to_generators()
    
    # Create ensemble mode selection with default single run
    ensemble_mode = True  # Set to True to run ensemble generation, False for single run
    
    if not ensemble_mode:
        # Original single-run storm generation code
        # Create output directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(OUTPUT_DIR, "NEWCSV", f"synthetic_storms_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Create output directory for NEWSTORMS CSV
        output_directory = r"G:\2025\GEVNEW\SOURCE\NEWSTORMS\CSV"
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate synthetic storms with exact count that pass through PAR
        logging.info(f"Generating exactly {target_storms} synthetic storms for {start_year}-{end_year} that pass through PAR...")
        all_storms = []
        
        # Forecast-based storm distribution
        total_forecast_storms = _df_forecast[
            (_df_forecast['YEAR'] >= start_year) & 
            (_df_forecast['YEAR'] <= end_year)
        ]['Storm_Count'].sum()
        
        if total_forecast_storms != target_storms:
            logging.warning(f"Forecast total {total_forecast_storms} differs from target {target_storms}")
        
        # Track the total valid storms (passing through PAR)
        valid_storm_count = 0
        
        # First generate storms based on yearly forecasts
        for year in range(start_year, end_year + 1):
            # Skip if we've already reached the target
            if valid_storm_count >= target_storms:
                break
                
            year_forecast = _df_forecast[_df_forecast['YEAR'] == year]
            if not year_forecast.empty:
                year_target = int(round(year_forecast['Storm_Count'].iloc[0]))
            else:
                # Fallback to 19 if no forecast for that year
                year_target = 19
            
            logging.info(f"Processing year {year}...")
            
            # Generate storms for this year
            yearly_storms = generate_yearly_storms(year, year_target, par_only=True)
            
            # Apply post-processing for better category-specific patterns
            yearly_storms = post_process_synthetic_tracks(yearly_storms, year)
            
            # Apply enhanced termination processing
            if not yearly_storms.empty:
                yearly_storms = process_yearly_storms(yearly_storms)
                logging.info(f"Applied enhanced termination processing for {year}. Storm count: {yearly_storms['SID'].nunique()}")
            
            # Apply PAR filtering with min_par_points=1 to ensure all pass through PAR
            yearly_storms = filter_storms_for_realism(yearly_storms, min_par_points=1)
            
            if not yearly_storms.empty:
                # Save individual year file
                year_file = os.path.join(run_dir, f"synthetic_storms_{year}.csv")
                save_storms_to_csv(yearly_storms, year_file)
                
                # Add to collection for combined file
                all_storms.append(yearly_storms)
                
                # Update valid storm count
                current_year_valid_storms = yearly_storms['SID'].nunique()
                valid_storm_count += current_year_valid_storms
                logging.info(f"Generated and saved {current_year_valid_storms} valid storms for year {year}")
                logging.info(f"Total valid storms so far: {valid_storm_count}/{target_storms}")
            else:
                logging.warning(f"No valid storms generated for year {year}")
        
        # Combine all years into a single file
        if all_storms:
            combined_storms = pd.concat(all_storms, ignore_index=True)
            #--------------------------------------------------
            # Test overall category distribution
            category_counts = {
                'TD': 0, 'TS': 0, 'STS': 0, 'TY': 0, 'STY': 0
            }

            for _, row in combined_storms.iterrows():
                wind = row['WIND']
                if wind >= 100:
                    category_counts['STY'] += 1
                elif wind >= 64:
                    category_counts['TY'] += 1
                elif wind >= 48:
                    category_counts['STS'] += 1
                elif wind >= 34:
                    category_counts['TS'] += 1
                else:
                    category_counts['TD'] += 1

            total = sum(category_counts.values())
            percentages = {k: (v/total)*100 for k, v in category_counts.items()}
            logging.info(f"Overall category counts: {category_counts}")
            logging.info(f"Overall category percentages: {percentages}")

            # Monthly breakdown - very useful for comparing with target distribution
            for month in range(1, 13):
                month_data = combined_storms[combined_storms['MONTH'] == month]
        
                if month_data.empty:
                    logging.info(f"No data for month {month}")
                    continue
            
                month_counts = {
                    'TD': 0, 'TS': 0, 'STS': 0, 'TY': 0, 'STY': 0
                }
        
                for _, row in month_data.iterrows():
                    wind = row['WIND']
                    if wind >= 100:
                        month_counts['STY'] += 1
                    elif wind >= 64:
                        month_counts['TY'] += 1
                    elif wind >= 48:
                        month_counts['STS'] += 1
                    elif wind >= 34:
                        month_counts['TS'] += 1
                    else:
                        month_counts['TD'] += 1
        
                month_total = sum(month_counts.values())
                if month_total > 0:  # Avoid division by zero
                    month_percentages = {k: (v/month_total)*100 for k, v in month_counts.items()}
            
                    # Format comparison with expected values
                    expected = _monthly_category_dist.get(month, [0, 0, 0, 0, 0])
                    expected_formatted = [f"{v:.1f}%" for v in expected]
                    actual_formatted = [f"{month_percentages.get('TD', 0):.1f}%", 
                                       f"{month_percentages.get('TS', 0):.1f}%", 
                                       f"{month_percentages.get('STS', 0):.1f}%", 
                                       f"{month_percentages.get('TY', 0):.1f}%", 
                                       f"{month_percentages.get('STY', 0):.1f}%"]
            
                    logging.info(f"Month {month} category counts: {month_counts}")
                    logging.info(f"Month {month} category percentages: {month_percentages}")
                    logging.info(f"Month {month} comparison:")
                    logging.info(f"  Expected: TD={expected_formatted[0]}, TS={expected_formatted[1]}, STS={expected_formatted[2]}, TY={expected_formatted[3]}, STY={expected_formatted[4]}")
                    logging.info(f"  Actual:   TD={actual_formatted[0]}, TS={actual_formatted[1]}, STS={actual_formatted[2]}, TY={actual_formatted[3]}, STY={actual_formatted[4]}")
            #--------------------------------------------------
            total_count = combined_storms['SID'].nunique()
            
            logging.info(f"Initial generation produced {total_count}/{target_storms} valid storms")
            
            # Adjust to exact target count
            if total_count != target_storms:
                # Track used storm IDs
                used_ids = set(combined_storms['SID'])
                
                if total_count > target_storms:
                    # Too many storms, remove some
                    storms_to_remove = total_count - target_storms
                    all_sids = combined_storms['SID'].unique()
                    sids_to_remove = np.random.choice(all_sids, storms_to_remove, replace=False)
                    combined_storms = combined_storms[~combined_storms['SID'].isin(sids_to_remove)]
                    logging.info(f"Removed {storms_to_remove} excess storms")
                else:
                    # Too few storms, generate more until we reach the target
                    storms_to_add = target_storms - total_count
                    logging.info(f"Need to generate {storms_to_add} additional storms that pass through PAR")
                    
                    # Generate the needed additional storms
                    additional_storms = generate_additional_par_storms(
                        storms_to_add, 
                        start_year, 
                        end_year, 
                        existing_ids=used_ids
                    )
                    
                    # Add the new storms to the combined dataset
                    if additional_storms:
                        combined_storms = pd.concat([combined_storms] + additional_storms, ignore_index=True)
                        new_total = combined_storms['SID'].nunique()
                        logging.info(f"Added {len(additional_storms)} more storms, now have {new_total}/{target_storms}")
                    
                    # If still short, make one final attempt with different approach
                    if combined_storms['SID'].nunique() < target_storms:
                        final_shortfall = target_storms - combined_storms['SID'].nunique()
                        logging.warning(f"Still short by {final_shortfall} storms. Making final attempt with eastern genesis approach.")
                        
                        # Generate storms with eastern genesis points for higher PAR entry probability
                        final_additions = []
                        final_attempts = 0
                        
                        while len(final_additions) < final_shortfall and final_attempts < 300:
                            final_attempts += 1
                            year = np.random.randint(start_year, end_year + 1)
                            month = np.random.choice([7, 8, 9, 10, 6, 11])  # Peak and shoulder months
                            
                            try:
                                # Generate a storm with higher chance of entering PAR
                                storm = generate_synthetic_storm(year, month, _df_positions, _historical_wind_speeds, used_ids)
                                
                                # Check if it passes through PAR
                                par_points = 0
                                for _, row in storm.iterrows():
                                    if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                                        par_points += 1
                                        break
                                
                                if par_points > 0:
                                    final_additions.append(storm)
                                    used_ids.add(storm['SID'].iloc[0])
                                    
                                    if len(final_additions) % 10 == 0:
                                        logging.info(f"Generated {len(final_additions)}/{final_shortfall} final addition storms")
                            except Exception as e:
                                if final_attempts % 50 == 0:
                                    logging.error(f"Error in final generation attempt: {e}")
                        
                        if final_additions:
                            combined_storms = pd.concat([combined_storms] + final_additions, ignore_index=True)
                            final_count = combined_storms['SID'].nunique()
                            logging.info(f"Final count: {final_count}/{target_storms} after all attempts")
            
            # Verify final count and log results
            final_valid_count = 0
            for sid in combined_storms['SID'].unique():
                storm_data = combined_storms[combined_storms['SID'] == sid]
                par_hit = False
                for _, row in storm_data.iterrows():
                    if is_inside_par(row['LAT'], row['LON'], use_buffer=False):
                        par_hit = True
                        break
                if par_hit:
                    final_valid_count += 1
                    
            logging.info(f"Final verification: {final_valid_count} storms pass through PAR out of {combined_storms['SID'].nunique()} total storms")
            
            # Ensure no minimum wind intensity lower than 22 knots
            #combined_storms.loc[combined_storms['WIND'] < 22.0, 'WIND'] = 22.0
            
            # Apply post-processing to clean up remnant low tracks
            logging.info("Applying post-processing to clean up remnant low tracks...")
            combined_storms = clean_remnant_low_tracks(combined_storms)
            logging.info(f"Clean-up complete: {combined_storms['SID'].nunique()} storms remain")
            
            # Apply final wind scaling to achieve exact target mean
            combined_storms = scale_winds_to_target_mean(combined_storms, target_mean=58.0)

            # Perform final validation to ensure the two strikes rule is enforced
            logging.info("Performing final two strikes validation...")
            combined_storms = clean_remnant_low_tracks(combined_storms)
            logging.info(f"Final validation complete: {combined_storms['SID'].nunique()} storms remaining")
            
            # Apply final wind scaling to achieve exact target mean
            combined_storms = scale_winds_to_target_mean(combined_storms, target_mean=58.0)

            # Save combined file
            combined_path = os.path.join(run_dir, f"synthetic_storms_{start_year}_{end_year}.csv")
            save_storms_to_csv(combined_storms, combined_path)
            
            # Copy to output directory
            import shutil
            csv_output_path = os.path.join(output_directory, f"synthetic_storms_{start_year}_{end_year}.csv")
            try:
                shutil.copy2(combined_path, csv_output_path)
                logging.info(f"Copied combined file to {csv_output_path}")
            except Exception as e:
                logging.error(f"Error copying combined file: {e}")
            
            # Create visualizations
            logging.info("Creating visualizations...")
            
            try:
                # Plot all storm tracks
                plot_path = os.path.join(run_dir, "storm_tracks.png")
                plot_storm_tracks(combined_storms, plot_path, show_plot=False)
                
                # Plot monthly distribution
                month_plot_path = os.path.join(run_dir, "monthly_distribution.png")
                plot_monthly_distribution(combined_storms, month_plot_path, show_plot=False)
                
                # Plot intensity distribution
                intensity_plot_path = os.path.join(run_dir, "intensity_distribution.png")
                plot_intensity_distribution(combined_storms, intensity_plot_path, show_plot=False)
                
                # Create density map
                density_plot_path = os.path.join(run_dir, "storm_density_map.png")
                create_storm_density_map(combined_storms, density_plot_path, show_plot=False)
                
                logging.info("Visualizations created successfully")
            except Exception as e:
                logging.error(f"Error creating visualizations: {e}")
            
            # Free memory
            del combined_storms, all_storms
            import gc
            gc.collect()
        else:
            logging.error("No storms were generated!")
        
        logging.info(f"All outputs saved to {run_dir}")
        logging.info("Process completed successfully!")
    
    else:
        # New ensemble generation code
        logging.info(f"Running in ENSEMBLE MODE with {ensemble_size} members")
        
        # Create output directory for this ensemble run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_base_dir = os.path.join(OUTPUT_DIR, "ENSEMBLE", f"tc_ensemble_{timestamp}")
        os.makedirs(ensemble_base_dir, exist_ok=True)
        
        # Set up additional logging for this ensemble run
        ensemble_log_file = os.path.join(ensemble_base_dir, "ensemble_generation.log")
        file_handler = logging.FileHandler(ensemble_log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Generate the ensemble members
        ensemble_results = generate_storm_ensemble(
            start_year=start_year, 
            end_year=end_year, 
            ensemble_size=ensemble_size, 
            base_seed=base_seed, 
            target_storms=target_storms
        )
        
        # Save ensemble summary
        ensemble_summary_path = os.path.join(ensemble_base_dir, "ensemble_summary.csv")
        with open(ensemble_summary_path, 'w') as f:
            f.write("ensemble_id,file_path,storm_count\n")
            for ensemble_id, file_path in ensemble_results.items():
                try:
                    df = pd.read_csv(file_path)
                    storm_count = df['SID'].nunique()
                    f.write(f"{ensemble_id},{file_path},{storm_count}\n")
                except Exception as e:
                    logging.error(f"Error reading storm count for ensemble {ensemble_id}: {e}")
                    f.write(f"{ensemble_id},{file_path},error\n")
        
        # Run ensemble analysis
        analysis_dir = os.path.join(ensemble_base_dir, "analysis")
        run_ensemble_analysis(ensemble_results, start_year, end_year, analysis_dir)
        
        # Prepare data for ArcMap
        arcmap_dir = os.path.join(ensemble_base_dir, "arcmap_data")
        os.makedirs(arcmap_dir, exist_ok=True)
        
        # Export ensemble data for ArcMap use
        try:
            for ensemble_id, file_path in ensemble_results.items():
                df = pd.read_csv(file_path)
                df['ENSEMBLE_ID'] = ensemble_id
                df['GLOBAL_SID'] = f"E{ensemble_id:02d}_" + df['SID']
                arcmap_file = os.path.join(arcmap_dir, f"ensemble_{ensemble_id:02d}_arcmap.csv")
                df.to_csv(arcmap_file, index=False)
            logging.info(f"Prepared ArcMap data in {arcmap_dir}")
        except Exception as e:
            logging.error(f"Error preparing ArcMap data: {e}")
        
        logging.info(f"Ensemble generation complete. Results in {ensemble_base_dir}")


# Update the main code to call the enhanced functions
if __name__ == "__main__":
    main()