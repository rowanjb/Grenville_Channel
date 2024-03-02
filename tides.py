#Comparing tide gauges with model(s) (needs to be updated once kit500 files are untarred!)
#Many functions are quite hard coded and "one-off"; I think that makes it a bit more readable and simple
#Rowan Brown
#Feb 2024

print('importing...')
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytide
print('done importing')

def get_tide_gauge_obs():
	'''Returns at dataframe containing the dates/times and values (centred at 0) for the tide gauges. 
	Where data was missing for Lowe Inlet, it is filled in using Harmonic analysis.
	(Similar harmonic analysis might be done for Hartley Bay, if its few missing dates are important.)'''

	#path to the tide gauge csv directory
	tidal_gauges_obs_dir_path = '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/'

	#dictionary of csv names
	file_name = {
		'Prince_Rupert': 'PrinceRupert_Jun012019-Nov202023.csv',
		'Hartley_Bay': 'HartleyBay_Jun012019-Nov202023.csv',
		'Lowe_Inlet': 'LoweInlet_2014_forConstituents.csv'}

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
	time2014 = df['Lowe_Inlet_date'].dropna().to_numpy()
	time2020 = df['Prince_Rupert_date'].dropna().to_numpy()
	h = df['Lowe_Inlet_val'].dropna().to_numpy()
	f, vu = wt.compute_nodal_modulations(time2014)
	w = wt.harmonic_analysis(h, f, vu)
	hp = wt.tide_from_tide_series(time2020, w) #this is the "modelled" data; can change the time if you want!!
	df['Lowe_Inlet_date'] = time2020 #adding the modelled data to the dataframe
	df['Lowe_Inlet_val'] = hp

	print('Done; obs data has been read and harmonic analysis has been completed')
	return df

def find_nearest_model_ids(gauge_location,model): 
	'''Searches for model nav_lat and nav_lon coords that are nearest the tide gauges.
	Acceptable inputs: "Lowe_Inlet", "Hartley_Bay", or "Prince_Rupert".
	(Although note that Prince Rupert isn't on the grc100 grid.)
	Returns the x and y indices.'''

	#observation coords (i.e., where the tide gauges were)     							Coords from Google:
	if gauge_location == 'Hartley_Bay': coords={'lat': 53.424213, 'lon': -129.251877} 	#53.4239N 129.2535W 
	elif gauge_location == 'Lowe_Inlet': coords={'lat': 53.55, 'lon': -129.583333}		#53.5564N 129.5875W 
	elif gauge_location == 'Prince_Rupert': coords={'lat': 54.317, 'lon': -130.324}		#54.3150N 130.3208W 
	
	#opening a random model file in order to get the nav_lon and nav_lat coordinates
	filepaths_txt = 'filepaths/' + model+ '_filepaths_1h_grid_T_2D.csv'
	filepaths = list(pd.read_csv(filepaths_txt))
	ds = xr.open_dataset(filepaths[0]).isel(time_counter=10) #picking a random time to save on memory 
	
	#essentially zeroing coordinates where zos is nan; will stop the "nearest coordinate" from being in land
	nav_lat = xr.where(~np.isnan(ds.zos),ds.nav_lat,0)
	nav_lon = xr.where(~np.isnan(ds.zos),ds.nav_lon,0)

	#finds indices of the grid points nearest a given location  
	def shortest_distance(lat,lon):
		abslat = np.abs(nav_lat.to_numpy()-lat)
		abslon = np.abs(nav_lon.to_numpy()-lon)
		c = (abslon**2 + abslat**2)**0.5
		return np.where(c == np.min(c))

	#indices of tide gauges on model grid
	model_y, model_x = shortest_distance(coords['lat'],coords['lon'])
	
	#corresponding coordinates on the model grid
	model_lat = ds.nav_lat.sel(x=model_x,y=model_y).to_numpy()
	model_lon = ds.nav_lon.sel(x=model_x,y=model_y).to_numpy()

	#printing results
	print('-> ' + gauge_location + ' tide gauge coords: ' + str(coords['lat']) + ', ' + str(coords['lon']))
	print('-> Corresponding xy ids on '+model+' grid: ' + str(model_x[0]) + ', ' + str(model_y[0]))
	print('-> Corresponding zos on '+model+' grid: ' + str(ds.zos[model_y[0],model_x[0]].to_numpy())) #to ensure not zero
	print('-> Corresponding '+model+' coords: ' + str(model_lat[0][0]) + ', ' + str(model_lon[0][0]))

	print('Done; nearest grc100 coords to the specified tide gauge have been found')
	return model_x, model_y

def model_tidal_signal(model_xy,model,yearmonth):
	'''Outputs the model sea surface variations, NOT centred around 0, for one month.
	Accepts tuple containing model x,y coordinates nearest a tide gauge.
	Returns dataarray containing ssh above geoid at the specified x,y coords.'''

	#opening model surface data
	filepaths_txt = 'filepaths/' + model + '_filepaths_1h_grid_T_2D.csv'
	filepaths = list(pd.read_csv(filepaths_txt))#[100:102] for kit500 #[30:33] for grc100 first week of 2020
	filepaths = [filepath for filepath in filepaths if yearmonth in filepath] #just for prototyping 
	model_preprocessor = lambda ds: ds[['zos','zos_ib']] #specify veriables to retrieve  
	#zos is sea surface height above geoid, zos_ib is the inverse barometer ssh
	ds = xr.open_mfdataset(filepaths,preprocess=model_preprocessor)

	#extracting only the important points
	model_x, model_y = model_xy
	ds['zos+zos_ib'] = ds.zos + ds.zos_ib
	da_at_gauge = ds['zos+zos_ib'].sel(x=model_x, y=model_y) #only looking at zos for now

	#ending for the day on feb 11
	#grc_hartley_ds = ds.sel(x=862, y=76) #very near edge==bad
	#print(ds_at_gauge.time_counter)

	print('Done; ssh data from '+model+' accessed')
	return da_at_gauge

def error(obs_df,location,model_da,model):
	'''Returns the RMS error of the model ssh compared to the tide gauge obs.'''

	model_da = model_da.drop_vars(['time_instant','nav_lat','nav_lon']).isel(x=0,y=0) #dropping unnecessary coords and vars from model da
	obs_df = obs_df[[location+'_date',location+'_val']].set_index(location+'_date') #dropping unnecessary obs data and setting dates to the index
	#note: when converting a df to a ds, the index becomes the coords
	ds = obs_df.to_xarray().rename({location+'_date':'time_counter',location+'_val':'obs_ssh'}) #turning obs df to a ds
	ds = ds.where(ds['time_counter']==model_da['time_counter']) #keeping obs data only where it overlaps in time with the model da
	ds['model_ssh'] = model_da #combining obs and model data
	ds['error_sq'] = (ds.model_ssh - ds.obs_ssh)**2
	ds['rmse'] = (ds.error_sq.mean(dim='time_counter'))**0.5
	print(ds.rmse.to_numpy())

def tide_plotting(obs_df,location,yearmonth,model_da,model,**kwargs):
	'''Plotting the observations and the model(s).
	Works for the first "period" (ie a week; can be changed) of a specified year and month.
	Can optionally pass model2_da and model2 to plot two models.'''

	#in case we want to plot two models instead of one
	model2_da = kwargs.get('model2_da', None)
	model2 = kwargs.get('model2', None)

	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert tidal signal ',
		'Hartley_Bay': 'Hartley Bay tidal signal ',
		'Lowe_Inlet': 'Lowe Inlet tidal signal '}

	#prepping to plot
	fig,ax1 = plt.subplots()
	c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

	#slicing the data
	year = yearmonth[:4]
	month = yearmonth[-2:]
	start_date = year+'-'+month+'-01'
	end_date = year+'-'+month+'-07'
	obs_df = obs_df.set_index(location+'_date')[start_date:end_date]
	model_da = model_da.sel(time_counter=slice(start_date,end_date))
	if model2: model2_da = model2_da.sel(time_counter=slice(start_date,end_date))
	
	#tide gauge data
	model_da.plot(ax=ax1, color=c2, label=model) #does it need to be centerd around 0?
	if model2: model2_da.plot(ax=ax1, color=c3, linestyle='dashed', label=model2)
	obs_df.reset_index().plot(x=location+'_date', y=location+'_val', ax=ax1, color=c1, label='observations')

	#holdover lines from when I was plotting tide gauges vs harmonic model; could probably delete this
	##df.plot(x='HartleyBay_date', y='HartleyBay_val', ax=ax1, color=c2, label='Hartley Bay observed')
	##df.plot(x='LoweInlet_date', y='LoweInlet_val', ax=ax1, color=c1, label='Lowe Inlet observed')
	###modelled tides
	##ax1.plot(time2020,hp,color=c3,label='Lowe Inlet harmonic model') #can change the time, so long as it is the same as when you made hp

	#horizontal axis stuff
	ax1.set_xlabel('Date')
	#ax1.set_xlim([date(year, month, 1), date(year, month, 8)])
	#ax1.xaxis.set_major_locator(mdates.DayLocator(interval=1))
	#ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

	#standard plotting stuff
	ax1.legend()
	ax1.set_title(plot_titles[location])# + '\n' + start_date + ' - ' + end_date)
	ax1.set_ylabel('SSH ($m$)')

	#saving
	fig.savefig('figures/incl_zos_ib_' + location + '_tides_' + start_date + '.png',dpi=300, bbox_inches="tight")
	fig.clf()

if __name__ == "__main__":

	yearmonth='202006'
	obs_df = get_tide_gauge_obs()

	#== this section is for computing the model-obs error ==#

	location = 'Lowe_Inlet'

	kit500_xy = find_nearest_model_ids(location,'kit500')
	kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
	error(obs_df,location,kit500_da,'kit500')

	grc100_xy = find_nearest_model_ids(location,'grc100')
	grc100_da = model_tidal_signal(grc100_xy,'grc100',yearmonth)
	error(obs_df,location,grc100_da,'grc100')

	quit()


	#== and this section is for making plots of ssh ==#

	location = 'Lowe_Inlet'
	kit500_xy = find_nearest_model_ids(location,'kit500')
	grc100_xy = find_nearest_model_ids(location,'grc100')
	kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
	grc100_da = model_tidal_signal(grc100_xy,'grc100',yearmonth)
	tide_plotting(obs_df,location,yearmonth,kit500_da,'kit500',model2_da=grc100_da,model2='grc100')

	location = 'Hartley_Bay'
	kit500_xy = find_nearest_model_ids(location,'kit500')
	grc100_xy = find_nearest_model_ids(location,'grc100')
	kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
	grc100_da = model_tidal_signal(grc100_xy,'grc100',yearmonth)
	tide_plotting(obs_df,location,yearmonth,kit500_da,'kit500',model2_da=grc100_da,model2='grc100')

	location = 'Prince_Rupert'
	kit500_xy = find_nearest_model_ids('Prince_Rupert','kit500')
	kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
	tide_plotting(obs_df,location,yearmonth,kit500_da,'kit500')