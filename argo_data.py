# %% Download raw data if not exist yet
from argopy import DataFetcher as ArgoDataFetcher
import xarray as xr
import pandas as pd
import numpy as np
from argopy import set_options
import os
import random

set_options(src="argovis")

# Define your region [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max]
region_coords = [-75, -45, 30, 45, 0, 500]  # Gulf stream
years = range(2010, 2027)

loader = ArgoDataFetcher()

os.makedirs("data/Argo_raw", exist_ok=True)

print("Starting batch download for February (2010-2026)...")

for year in years:
    output_file = f"data/Argo_raw/argo_february_{year}.csv"
    
    # Check if file exists to allow resuming if the script crashes
    if os.path.exists(output_file):
        print(f"Skipping {year}, file already exists.")
        continue

    try:
        t1 = f"{year}-02-01"
        t2 = f"{year}-03-01"
        
        print(f"Fetching: {year}-02...")
        
        # 3. Use the region + time constraints
        # Removed .load() here because .to_xarray() does the fetching.
        ds_year = loader.region(region_coords + [t1, t2]).to_xarray()
        ds_year_csv = np.stack((
            ds_year['LONGITUDE'].data,
            ds_year['LATITUDE'].data,
            ds_year['PRES'].data, 
            ds_year['TIME'].data.dayofyear, 
            ds_year['TIME'].data.year, 
            ds_year['PLATFORM_NUMBER'].data,
            ds_year['CYCLE_NUMBER'].data,
            ds_year['TEMP'].data,
            ), axis=-1)
        
        # 4. Critical: Check if the dataset is empty before saving
        if ds_year_csv.shape[0] > 0:
            df = pd.DataFrame(ds_year_csv, columns=['longitude', 'latitude', "pressure", "dayofyear", "year", "platform_number", "cycle_number", "temperature"])
            df.to_csv(output_file, index=False)
            print(f"Success: Saved {year}-02")
        else:
            print(f"No data found for {year}-02")
            
    except Exception as e:
        print(f"Error in {year}: {e}")

# %% Restrict number of responses per float
def subsample_per_float(dataset, n_obs_per_float=100):
    """
    Subsamples the dataset based on unique values in the 6th column.
    
    Args:
        dataset (pd.DataFrame): The input pandas DataFrame.
        n_obs_per_float (int): Target number of rows per unique value.
        
    Returns:
        pd.DataFrame: The subsampled dataset.
    """
    # Identify the name of the grouping column
    group_col = 'platform_number'
    
    # Define a helper function for the sampling logic
    def sample_logic(group):
        if len(group) <= n_obs_per_float:
            return group
        return group.sample(n=n_obs_per_float)
    
    # Group by the 6th column and apply the sampling logic
    subsampled_index = dataset.groupby(group_col, group_keys=False).apply(
        sample_logic, include_groups=False).index.tolist()
    
    return dataset.loc[subsampled_index]

# %% set seed
random.seed(123)
np.random.seed(123)
data_subset_list = []
# %% random split into training and testing
for k in range(len(years)):
    year = years[k]
    fn = f"data/Argo_raw/argo_february_{year}.csv"
    data = pd.read_csv(fn, sep=',', header=0)
    data.dropna(inplace=True)
    data['platform_number'] = data['platform_number'].astype(int)
    data['cycle_number'] = data['cycle_number'].astype(int)
    data['dayofyear'] = data['dayofyear'].astype(int)
    data['year'] = data['year'].astype(int)
    data_subset = subsample_per_float(data, 200)
    data_subset_list.append(data_subset)
full_data_subset = pd.concat(data_subset_list, axis=0)
global_min = full_data_subset.min()
global_max = full_data_subset.max()
range_denom = global_max - global_min
def _normalize_df(df, g_min, g_denom):
    # Align global stats with the columns present in the specific dataframe
    cols = df.columns
    return (df - g_min[cols]) / g_denom[cols]


for k, data_subset in enumerate(data_subset_list):
    fn_out = f"data/Argo/seed_{k}/"
    os.makedirs(fn_out + "train/", exist_ok=True)
    os.makedirs(fn_out + "test/", exist_ok=True)
    data_subset_normalized = _normalize_df(data_subset, global_min, range_denom)
    data_train = data_subset_normalized.sample(frac=0.8, random_state=123)
    data_test = data_subset_normalized.drop(data_train.index)
    x_train = data_train.iloc[:, :4]
    y_train = data_train.iloc[:, -1:]
    x_test = data_test.iloc[:, :4]
    y_test = data_test.iloc[:, -1:]
    x_train.to_csv(fn_out + "train/x.csv", index=False, header=False, mode='w')
    y_train.to_csv(fn_out + "train/y.csv", index=False, header=False, mode='w')
    x_test.to_csv(fn_out + "test/x.csv", index=False, header=False, mode='w')
    y_test.to_csv(fn_out + "test/y.csv", index=False, header=False, mode='w')
    