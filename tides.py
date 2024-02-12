#Comparing tide gauges with model(s) (needs to be updated once kit500 files are untarred!)
#Many functions are quite hard coded and "one-off"; I think that makes it a bit more readable and simple
#Rowan Brown
#Feb 2024

import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date
import matplotlib.pyplot as plt
import pytide

def get_tide_gauge_obs():
	'''Returns at dataframe containing the dates/times and values (centred at 0) for the tide gauges. 
	Where data was missing for Lowe Inlet, it is filled in using Harmonic analysis.
	(Similar harmonic analysis might be done for Hartley Bay, if its few missing dates are important.)'''

	#path to the tide gauge csv directory
	tidal_gauges_obs_dir_path = '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/'

	#dictionary of csv names
	file_name = {
		'PrinceRupert': 'PrinceRupert_Jun012019-Nov202023.csv',
		'HartleyBay': 'HartleyBay_Jun012019-Nov202023.csv',
		'LoweInlet': 'LoweInlet_2014_forConstituents.csv'}

	#creating dataframe with the tide gauge data
	df_list = []
	for location in file_name.keys():
		obs_path = tidal_gauges_obs_dir_path + file_name[location]
		df = pd.read_csv(obs_path, sep=",", header=None,engine='python')
		df = df.drop(index=range(0,8), columns=[2])
		df.columns = [location+'_date',location+'_val']
		df[location+'_date'] = pd.to_datetime(df[location+'_date'],format = "%Y/%m/%d %H:%M")
		df[location+'_val'] = df[location+'_val'].astype(float)
		mean = df[location+'_val'].mean()
		df[location+'_val'] = df[location+'_val'] - mean
		df_list.append(df)
	df = pd.concat(df_list,axis=1)

	#extending the 2014 Lowe Inlet data forward to match the other data using the Pytide package
	wt = pytide.WaveTable()
	time2014 = df['LoweInlet_date'].dropna().to_numpy()
	time2020 = df['PrinceRupert_date'].dropna().to_numpy()
	h = df['LoweInlet_val'].dropna().to_numpy()
	f, vu = wt.compute_nodal_modulations(time2014)
	w = wt.harmonic_analysis(h, f, vu)
	hp = wt.tide_from_tide_series(time2020, w) #this is the "modelled" data; can change the time if you want!!
	df['LoweInlet_date'] = time2020 #adding the modelled data to the dataframe
	df['LoweInlet_val'] = hp

	print('Done; obs data has been read and harmonic analysis has been completed')
	return df

def find_nearest_model_ids(gauge_location): 
	'''Searches for model nav_lat and nav_lon coords that are nearest the tide gauges.
	Acceptable inputs: "Lowe_Inlet", "Hartley_Bay", or "Prince_Rupert".
	(Although note that Prince Rupert isn't on the grc100 grid.)
	Returns the x and y indices.'''

	#observation coords (i.e., where the tide gauges were)     							Coords from Google:
	if gauge_location == 'Hartley_Bay': coords={'lat': 53.424213, 'lon': -129.251877} 	#53.4239N 129.2535W 
	elif gauge_location == 'Lowe_Inlet': coords={'lat': 53.55, 'lon': -129.583333}		#53.5564N 129.5875W 
	elif gauge_location == 'Prince_Rupert': coords={'lat': 54.317, 'lon': -130.324}		#54.3150N 130.3208W 
	
	#opening a random model file in order to get the nav_lon and nav_lat coordinates
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
	grc_y, grc_x = shortest_distance(coords['lat'],coords['lon'])

	#corresponding coordinates on the model grid
	grc_lat = ds.nav_lat.sel(x=grc_x,y=grc_y).to_numpy()
	grc_lon = ds.nav_lon.sel(x=grc_x,y=grc_y).to_numpy()

	#printing results
	print(gauge_location + ' tide gauge coords: ' + str(coords['lat']) + ', ' + str(coords['lon']))
	print('Corresponding xy ids on grc100 grid: ' + str(grc_x[0]) + ', ' + str(grc_y[0]))
	print('Corresponding grc100 coords: ' + str(grc_lat[0][0]) + ', ' + str(grc_lon[0][0]))

	#Note on the results:
	#On the grc100 grid, Lowe Inlet is an exact match(ish) and Hartley Bay is ~30m off, possibly less

	print('Done; nearest grc100 coords to the specified tide gauge have been found')
	return grc_x, grc_y

def model_tidal_signal(model_xy):
	'''Outputs the sea surface variations, centred around 0.
	Accepts tuple containing x,y coordinates associated with a tide gauge.
	Returns...'''

	#opening model surface data
	grc_filepaths_txt = 'filepaths/' + 'grc_filepaths_1h_grid_T_2D.csv'
	grc_filepaths = list(pd.read_csv(grc_filepaths_txt))[1:3]
	grc_preprocessor = lambda ds: ds[['zos','zos_ib']] #specify veriables to retrieve  
	ds = xr.open_mfdataset(grc_filepaths,preprocess=grc_preprocessor)

	#extracting only the important points
	grc_x, grc_y = model_xy
	ds_at_gauge = ds.zos.sel(x=grc_x, y=grc_y)
	#grc_hartley_ds = ds.sel(x=862, y=76) #very near edge==bad

	#STOPPING FOR THE DAY: Feb 11
	print(ds_at_gauge.time_counter)
	return ds_at_gauge

def plotting():
	#plotting
    fig,ax1 = plt.subplots()

    c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

    #tide gauge data
    df.plot(x='PrinceRupert_date', y='PrinceRupert_val', ax=ax1, color=c1, label='Prince Rupert observed')
    df.plot(x='HartleyBay_date', y='HartleyBay_val', ax=ax1, color=c2, label='Hartley Bay observed')
    ##df.plot(x='LoweInlet_date', y='LoweInlet_val', ax=ax1, color=c1, label='Lowe Inlet observed')

    #modelled tides
    ax1.plot(time2020,hp,color=c3,label='Lowe Inlet harmonic model') #can change the time, so long as it is the same as when you made hp

    ax1.legend()
    ax1.set_title('Grenville Channel tides, mid-2020')
    ax1.set_ylabel('SSH')
    ax1.set_xlabel('Date')
    #ax1.set_xlim([date(2020, 1, 1), date(2020, 1, 7)])
    ax1.set_xlim([date(2020, 6, 1), date(2020, 6, 7)])

    fig.savefig('plots/tide_gauges_2020.png',dpi=300, bbox_inches="tight")
    fig.clf()

    print('done')

if __name__ == "__main__":
    obs_df = get_tide_gauge_obs()
	model_xy = find_nearest_model_ids('Lowe_Inlet')
    model_tidal_signal(model_xy)
	#compare_models_and_obs()
	#get_tide_gauge_obs()
