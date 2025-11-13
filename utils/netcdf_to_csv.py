import xarray as xr
import pandas as pd

# Input your NetCDF file path here
netcdf_file = "/Users/raghavsharma/Desktop/Float_Chat/data/D20250602_prof_0.nc"   # <-- replace with your file
csv_file = "data/sample_argo5.csv"

# Open the NetCDF file
ds = xr.open_dataset(netcdf_file)

# Convert to pandas DataFrame
df = ds.to_dataframe().reset_index()

# Save as CSV
df.to_csv(csv_file, index=False)

print(f"✅ Converted {netcdf_file} → {csv_file}")
print("\n--- Preview ---")
print(df.head())
