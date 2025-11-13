import matplotlib.pyplot as plt
import xarray as xr

file = '/Users/raghavsharma/Desktop/Float_Chat/data/D20250730_prof_0.nc'

xrds = xr.open_dataset(file)
xrds['HISTORY_PARAMETER'].plot()
plt.show()