#Comparing tide gauges with model(s)
#This is pretty hard coded and "one-off"; I think that makes it a bit more readable and simple
#Rowan Brown
#11 Feb 2024

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date
import matplotlib.pyplot as plt
import pytide

def find_nearest_model_ids():
	'''Searches for model x and y coords that are nearest the tide gauges'''

	#observation coords (i.e., where the tide gauges were)			Coords from Google:
	hartley_bay_coords = {'lat': 53.424213, 'lon': -129.251877} 	#53.4239N 129.2535W
	lowe_inlet_coords = {'lat': 53.55, 'lon': -129.583333} 			#53.5564N 129.5875W
	prince_rupert_coords = {'lat': 54.317, 'lon': -130.324} 		#54.3150N 130.3208W

	#opening a model file in order to get the nav_lon and nav_lat coordinates
	grc_filepaths_txt = 'filepaths/' + 'grc_filepaths_1h_grid_T_2D.csv'
	grc_filepaths = list(pd.read_csv(grc_filepaths_txt))
	ds = xr.open_dataset(grc_filepaths[0])

	#finds indices of the grid points nearest a given location  
	def shortest_distance(lat,lon):
		abslat = np.abs(ds.nav_lat.to_numpy()-lat)
		abslon = np.abs(ds.nav_lon.to_numpy()-lon)
		c = (abslon**2 + abslat**2)**0.5
		return np.where(c == np.min(c))

	#indices of tide gauges on model grid
	grc_lowe_y, grc_lowe_x = shortest_distance(lowe_inlet_coords['lat'],lowe_inlet_coords['lon'])
	grc_hartley_y, grc_hartley_x = shortest_distance(hartley_bay_coords['lat'],hartley_bay_coords['lon'])
	
	#corresponding coordinates on the model grid
	grc_lowe_lat = ds.nav_lat.sel(x=grc_lowe_x,y=grc_lowe_y).to_numpy()
	grc_lowe_lon = ds.nav_lon.sel(x=grc_lowe_x,y=grc_lowe_y).to_numpy()
	grc_hartley_lat = ds.nav_lat.sel(x=grc_hartley_x,y=grc_hartley_y).to_numpy()
	grc_hartley_lon = ds.nav_lon.sel(x=grc_hartley_x,y=grc_hartley_y).to_numpy()

	#printing results
	print('Lowe inlet tide gauge coords: ' + str(lowe_inlet_coords['lat']) + ', ' + str(lowe_inlet_coords['lon']))
	print('Lowe inlet xy ids on grc100 grid: ' + str(grc_lowe_x[0]) + ', ' + str(grc_lowe_y[0]))
	print('...which corresponds to these coords: ' + str(grc_lowe_lat[0][0]) + ', ' + str(grc_lowe_lon[0][0]))
	print('Hartley bay tide gauge coords: ' + str(hartley_bay_coords['lat']) + ', ' + str(hartley_bay_coords['lon']))
	print('Hartley bay xy ids on grc100 grid: ' + str(grc_hartley_x[0]) + ', ' + str(grc_hartley_y[0]))
	print('...which corresponds to these coords: ' + str(grc_hartley_lat[0][0]) + ', ' + str(grc_hartley_lon[0][0]))

	#Notei on the results:
	#On the grc100 grid, Lowe Inlet is an exact match(ish) and Hartley Bay is ~30m off, possibly less

def compare_models_and_obs():

	#opening model surface data
	grc_filepaths_txt = 'filepaths/' + 'grc_filepaths_1h_grid_T_2D.csv'
	grc_filepaths = list(pd.read_csv(grc_filepaths_txt))[1:3]
	grc_preprocessor = lambda ds: ds[['zos','zos_ib']] #specify veriables to retrieve  
	ds = xr.open_mfdataset(grc_filepaths,preprocess=grc_preprocessor)
	
	#extracting only the important points
	grc_lowe_ds = ds.sel(x=564, y=189)
	grc_hartley_ds = ds.sel(x=862, y=76) #very near edge==bad
	
	print(grc_lowe_ds)

if __name__ == "__main__":
	find_nearest_model_ids()
	compare_models_and_obs()
