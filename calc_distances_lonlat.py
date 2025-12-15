import numpy as np
import xarray as xr

def calculate_haversine_numpy(lats1, lons1, lats2, lons2):
    """Calculates the great-circle distance between corresponding points in 
    (potentially 2D) NumPy arrays."""
    R = 6371.0  # Radius of the Earth in kilometers
    lats1, lons1, lats2, lons2 = map(np.radians, [lats1, lons1, lats2, lons2])
    dlon = lons2 - lons1
    dlat = lats2 - lats1
    a = np.sin(dlat / 2.0)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def analyze_2d_grid(lat_grid, lon_grid):
    """
    Computes distances between adjacent cells in a 2D lat/lon grid.

    :param lat_grid: A 2D NumPy array of latitudes (rows, columns).
    :param lon_grid: A 2D NumPy array of longitudes (rows, columns).
    :return: A dictionary containing 2D arrays of distances in km.
    """
    # Calculate North-South distances (between adjacent rows)
    # Compare each row with the next one
    distances_ns = calculate_haversine_numpy(
        lat_grid[:-1, :], lon_grid[:-1, :], # All rows except the last
        lat_grid[1:, :], lon_grid[1:, :]   # All rows except the first
    )

    # Calculate East-West distances (between adjacent columns)
    # Compare each column with the next one
    distances_ew = calculate_haversine_numpy(
        lat_grid[:, :-1], lon_grid[:, :-1], # All columns except the last
        lat_grid[:, 1:], lon_grid[:, 1:]   # All columns except the first
    )
    
    # Calculate mean cell sizes
    mean_ns_cell_size = np.mean(distances_ns)
    mean_ew_cell_size = np.mean(distances_ew)

    return {
        "N-S Distances (km)": distances_ns,
        "E-W Distances (km)": distances_ew,
        "Mean N-S Cell Size (km)": mean_ns_cell_size,
        "Mean E-W Cell Size (km)": mean_ew_cell_size,
        "Approximate Average Cell Area (km²)": mean_ns_cell_size * mean_ew_cell_size
    }

# --- Example Usage ---

path_runs='/oscar/data/deeps/private/chorvat/santanarc/n/southern/runs/' # ''~/'
run='50km_bbm4roms_20130101'

#Grid information
#run=runs[expt[0]-1] # 'data_glorys'
data = xr.open_dataset(path_runs+run+'/output/Moorings_2015m01.nc')
lon_2d = data.longitude #sit.to_masked_array() # Extract a given variable
lat_2d = data.latitude #sit.to_masked_array() # Extract a given variable



# Create sample 2D grid data (e.g., 4x5 grid points)
#rows, cols = 4, 5
#lats_1d = np.linspace(34.0, 35.0, rows)
#lons_1d = np.linspace(-118.0, -117.0, cols)

# Convert 1D arrays to a 2D meshgrid
#lon_2d, lat_2d = np.meshgrid(lons_1d, lats_1d)

print(f"Shape of Latitude grid: {lat_2d.shape}")
print(f"Shape of Longitude grid: {lon_2d.shape}")

grid_distances = analyze_2d_grid(lat_2d, lon_2d)

print("\nGrid Analysis Results:")
print(f"* N-S Distances shape: {grid_distances['N-S Distances (km)'].shape}")
print(f"* E-W Distances shape: {grid_distances['E-W Distances (km)'].shape}")
print(f"* Mean N-S Cell Size: {grid_distances['Mean N-S Cell Size (km)']:.4f} km")
print(f"* Mean E-W Cell Size: {grid_distances['Mean E-W Cell Size (km)']:.4f} km")
print(f"* Approximate Average Cell Area: {grid_distances['Approximate Average Cell Area (km²)']:.4f} km²")

