import xarray as xr

netcdf_file = "/Users/raghavsharma/Desktop/Float_Chat/data/D20250602_prof_0.nc"
xrds = xr.open_dataset(netcdf_file)

dimensions = xrds.dims
coords = xrds.coords
temperature = xrds.data_vars['TEMP'].values
temp_var_attrs = xrds.data_vars['TEMP'].attrs['standard_name']

print(coords)
print(dimensions)
print(temperature)
print(temp_var_attrs)
