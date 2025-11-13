import xarray as xr
import numpy as np
import datetime as dt

file = "/Users/raghavsharma/Desktop/Float_Chat/data/nodc_D1900975_339.nc"

ds = xr.open_dataset(file)
df = ds.to_dataframe()
df.to_csv("/Users/raghavsharma/Desktop/Float_Chat/data_tocsv/data.csv")