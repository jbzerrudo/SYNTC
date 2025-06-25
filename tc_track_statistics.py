import numpy as np
import pandas as pd
from scipy import stats
import os
import logging

class DataDrivenTrackModel:
    """
    A data-driven model for tropical cyclone track movements based on historical patterns.
    Replaces hardcoded track constraints with statistical patterns learned from data.
    """
    
    def __init__(self, historical_data_path=None, bin_size=2.0, smooth_factor=0.5):
        """
        Initialize the track model using historical data.
        
        Args:
            historical_data_path: Path to CSV with historical track data
            bin_size: Size of latitude/longitude bins in degrees
            smooth_factor: Smoothing parameter for distributions
        """
        self.bin_size = bin_size
        self.smooth_factor = smooth_factor
        self.lat_bins = np.arange(0, 30 + bin_size, bin_size)
        self.lon_bins = np.arange(110, 150 + bin_size, bin_size)
        
        # Distributions for dlat/dlon by latitude bin
        self.lat_movement_dist = {}  # Key: lat_bin, Value: KDE for dlat
        self.lon_movement_dist = {}  # Key: lat_bin, Value: KDE for dlon
        
        # Transition probabilities
        self.southward_prob = {}  # Key: lat_bin, Value: probability of southward movement
        
        # Load and process historical data if provided
        if historical_data_path and os.path.exists(historical_data_path):
            self.load_historical_data(historical_data_path)
        else:
            logging.warning("No historical data provided. Model needs to be trained before use.")
    
    def bin_latitude(self, lat):
        """Get the latitude bin for a given latitude."""
        bin_idx = int(lat / self.bin_size)
        return bin_idx * self.bin_size
    
    def load_historical_data(self, data_path):
        """
        Load and process historical track data to build movement distributions.
        
        Args:
            data_path: Path to CSV file with historical track data
        """
        logging.info(f"Loading historical track data from {data_path}")
        
        try:
            # Load historical data
            historical_df = pd.read_csv(data_path)
            
            # Ensure required columns exist
            required_cols = ['SID', 'LAT', 'LON']
            if not all(col in historical_df.columns for col in required_cols):
                raise ValueError(f"Historical data missing required columns: {required_cols}")
            
            # Process each storm track to extract movement patterns
            storm_ids = historical_df['SID'].unique()
            
            # Data containers for movement statistics
            lat_changes = {lat_bin: [] for lat_bin in self.lat_bins}
            lon_changes = {lat_bin: [] for lat_bin in self.lat_bins}
            southward_counts = {lat_bin: 0 for lat_bin in self.lat_bins}
            total_counts = {lat_bin: 0 for lat_bin in self.lat_bins}
            
            # Process each storm track
            for storm_id in storm_ids:
                # Get storm track sorted by time
                storm_track = historical_df[historical_df['SID'] == storm_id].sort_values('ISO_TIME' if 'ISO_TIME' in historical_df.columns else 'SID')
                
                if len(storm_track) < 2:
                    continue  # Skip tracks with only one point
                
                # Calculate changes between consecutive points
                for i in range(1, len(storm_track)):
                    prev_lat = storm_track['LAT'].iloc[i-1]
                    prev_lon = storm_track['LON'].iloc[i-1]
                    curr_lat = storm_track['LAT'].iloc[i]
                    curr_lon = storm_track['LON'].iloc[i]
                    
                    dlat = curr_lat - prev_lat
                    dlon = curr_lon - prev_lon
                    
                    # Get latitude bin for the previous position
                    lat_bin = self.bin_latitude(prev_lat)
                    
                    # Store movement data
                    lat_changes[lat_bin].append(dlat)
                    lon_changes[lat_bin].append(dlon)
                    
                    # Track southward movement
                    if dlat < 0:
                        southward_counts[lat_bin] += 1
                    total_counts[lat_bin] += 1
            
            # Build statistical models for each latitude bin
            for lat_bin in self.lat_bins:
                # Skip bins with insufficient data
                if len(lat_changes[lat_bin]) < 10:
                    continue
                
                # Calculate probability of southward movement
                if total_counts[lat_bin] > 0:
                    self.southward_prob[lat_bin] = southward_counts[lat_bin] / total_counts[lat_bin]
                else:
                    # Default to global average if no data for this bin
                    total_southward = sum(southward_counts.values())
                    total_movements = sum(total_counts.values())
                    self.southward_prob[lat_bin] = total_southward / total_movements if total_movements > 0 else 0.3
                
                # Create kernel density estimators for dlat and dlon
                try:
                    # Use Gaussian KDE for smooth distributions
                    self.lat_movement_dist[lat_bin] = stats.gaussian_kde(
                        lat_changes[lat_bin], 
                        bw_method=self.smooth_factor
                    )
                    
                    self.lon_movement_dist[lat_bin] = stats.gaussian_kde(
                        lon_changes[lat_bin], 
                        bw_method=self.smooth_factor
                    )
                except Exception as e:
                    logging.warning(f"Could not create KDE for lat_bin {lat_bin}: {e}")
                    # Fall back to normal distribution based on sample statistics
                    lat_mean = np.mean(lat_changes[lat_bin])
                    lat_std = np.std(lat_changes[lat_bin]) or 0.1
                    lon_mean = np.mean(lon_changes[lat_bin])
                    lon_std = np.std(lon_changes[lat_bin]) or 0.1
                    
                    # Create simple parametric distributions
                    self.lat_movement_dist[lat_bin] = lambda size=1: np.random.normal(lat_mean, lat_std, size)
                    self.lon_movement_dist[lat_bin] = lambda size=1: np.random.normal(lon_mean, lon_std, size)
            
            logging.info(f"Successfully built movement models for {len(self.lat_movement_dist)} latitude bins")
            
            # Fill in gaps for bins with no data using nearest neighbor interpolation
            self._fill_distribution_gaps()
            
        except Exception as e:
            logging.error(f"Error processing historical data: {e}")
    
    def _fill_distribution_gaps(self):
        """Fill in missing latitude bins by interpolating from nearby bins."""
        # Get bins with data
        bins_with_data = sorted(self.lat_movement_dist.keys())
        
        if not bins_with_data:
            logging.error("No valid movement distributions found")
            return
        
        # For each missing bin, find nearest neighbors
        for lat_bin in self.lat_bins:
            if lat_bin not in self.lat_movement_dist:
                # Find nearest bin with data
                distances = [abs(lat_bin - bin) for bin in bins_with_data]
                nearest_idx = np.argmin(distances)
                nearest_bin = bins_with_data[nearest_idx]
                
                # Copy distributions from nearest bin
                self.lat_movement_dist[lat_bin] = self.lat_movement_dist[nearest_bin]
                self.lon_movement_dist[lat_bin] = self.lon_movement_dist[nearest_bin]
                
                # For southward probability, use a more conservative approach
                if lat_bin < 5.0:
                    # Very low latitude: rarely move southward
                    self.southward_prob[lat_bin] = 0.05
                elif lat_bin < nearest_bin:
                    # Lower than nearest: reduce southward probability
                    self.southward_prob[lat_bin] = max(0.05, self.southward_prob.get(nearest_bin, 0.3) * 0.7)
                elif lat_bin > nearest_bin:
                    # Higher than nearest: increase southward probability
                    self.southward_prob[lat_bin] = min(0.5, self.southward_prob.get(nearest_bin, 0.3) * 1.3)
                else:
                    # Same as nearest (shouldn't happen)
                    self.southward_prob[lat_bin] = self.southward_prob.get(nearest_bin, 0.3)
    
    def sample_movement(self, current_lat, current_lon, previous_dlat=None, previous_dlon=None):
        """
        Sample realistic latitude/longitude changes based on current position.
        This replaces the hardcoded constraints in the original model.
        
        Args:
            current_lat: Current latitude
            current_lon: Current longitude
            previous_dlat: Previous change in latitude (for autocorrelation)
            previous_dlon: Previous change in longitude (for autocorrelation)
        
        Returns:
            Tuple of (dlat, dlon) - changes to apply to current position
        """
        # Get appropriate latitude bin
        lat_bin = self.bin_latitude(current_lat)
        
        # Default values if no distributions available
        if not self.lat_movement_dist or lat_bin not in self.lat_movement_dist:
            # Very basic defaults based on general TC behavior
            dlat = np.random.normal(0.15, 0.1)  # Slight northward bias
            dlon = np.random.normal(-0.3, 0.15)  # Westward bias
            return dlat, dlon
        
        # Determine if movement should be southward based on latitude-specific probability
        allow_southward = np.random.random() < self.southward_prob.get(lat_bin, 0.3)
        
        # Sample from learned distributions
        try:
            dlat_sample = self.lat_movement_dist[lat_bin](1)[0]
            dlon_sample = self.lon_movement_dist[lat_bin](1)[0]
            
            # Apply autocorrelation with previous movement if available
            if previous_dlat is not None and previous_dlon is not None:
                # Blend with previous movement (autocorrelation)
                autocorr_factor = 0.7  # How much previous movement influences current
                dlat_sample = autocorr_factor * previous_dlat + (1 - autocorr_factor) * dlat_sample
                dlon_sample = autocorr_factor * previous_dlon + (1 - autocorr_factor) * dlon_sample
            
            # If southward movement not allowed, ensure dlat is positive
            if not allow_southward and dlat_sample < 0:
                # Instead of forcing northward, reduce the southward component
                # This creates a more natural pattern than simply reversing direction
                dlat_sample = abs(dlat_sample) * 0.3  # Reduced magnitude
            
            # Special handling for very low latitudes (physical constraints near equator)
            if current_lat < 5.0 and dlat_sample < 0:
                # Stronger constraint near equator, but still data-driven
                equator_factor = current_lat / 5.0  # 0 at equator, 1 at 5°N
                dlat_sample = max(dlat_sample, -0.05 * equator_factor)
            
            return dlat_sample, dlon_sample
            
        except Exception as e:
            logging.warning(f"Error sampling movement: {e}, using defaults")
            # Fallback to reasonable defaults
            dlat = np.random.normal(0.1, 0.1)
            dlon = np.random.normal(-0.2, 0.15)
            return dlat, dlon