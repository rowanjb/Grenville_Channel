import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

def closest_grid(nav_lon, nav_lat, lon, lat):
    abslat = np.abs(nav_lat-lat)
    abslon = np.abs(nav_lon-lon)

    c = (abslon**2 + abslat**2)**0.5
    return np.where(c == np.min(c))

ctd_file = '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/CTD/grc1_20190809_20200801_0020m_L2.ctd.nc'

cf = xr.open_dataset(ctd_file, chunks={'time': 50})

print(cf)

cf['sea_water_temperature'].plot(label='ctd')

#now read in the model data
#paths for model files are in filepaths/

grc_filepaths_txt = 'filepaths/' + 'grc100_filepaths_1h_grid_T.csv'
grc_filepaths = list(pd.read_csv(grc_filepaths_txt))
grc_filepaths.remove('/mnt/storage3/tahya/DFO/grc100_model_results/2019060100_000/NEMO_RPN_1h_grid_T.nc') #file is bad??
gs = xr.open_mfdataset(grc_filepaths, chunks={'time_counter':50})

print(gs)

#get the closest grid cell to the mooring
y, x = closest_grid(gs['nav_lon'], gs['nav_lat'], cf['longitude'], cf['latitude'])

temp_ts = gs['thetao'].isel(y=y, x=x)
print(temp_ts)

#now need to figure out the model depth
#should get the range the ctd sits at
#and take the closest model depth to the average of that

avg_ctd_depth = cf['depth'].mean()
print(avg_ctd_depth)

temp_ts = temp_ts.sel(deptht=avg_ctd_depth, method='nearest')
print(temp_ts)

temp_ts[:].plot(label='grc100')
plt.legend()
plt.savefig('figs/mooring_20m_temp.png')

cf.close()
gs.close()
