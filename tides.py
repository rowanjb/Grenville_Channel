#Comparing tide gauges with model(s) (needs to be updated once kit500 files are untarred!)
#Many functions are quite hard coded and "one-off"; I think that makes it a bit more readable and simple
#Rowan Brown
#Feb 2024

print('importing packages...')
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytide
from scipy import signal
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

def tide_gauge_rebuilt_signal(obs_df,**kwargs):
	'''Returns observational data rebuilt with only a subset of constituents (usually the 8 from the model).
	All other constituents are removed.'''

	#in case you want to specify the number of constituents 
	num_const = kwargs.get('num_const', None)

	#ordered as it comes out of pytide.WaveTable()
	all_const = ['O1', 'P1', 'K1', '2N2', 'Mu2', 'N2', 'Nu2', 'M2', 'L2', 'T2', 'S2', 'K2', 'M4', 'S1', 'Q1', 'Mm', 'Mf', 'Mtm', 'Msqm', 'Eps2', 'Lambda2', 'Eta2', '2Q1', 'Sigma1', 'Rho1', 'M11', 'M12', 'Chi1', 'Pi1', 'Phi1', 'Theta1', 'J1', 'OO1', 'M3', 'M6', 'MN4', 'MS4', 'N4', 'R2', 'R4', 'S4', 'MNS2', 'M13', 'MK4', 'SN4', 'SK4', '2MN6', '2MS6', '2MK6', 'MSN6', '2SM6', 'MSK6', 'MP1', '2SM2', 'Psi1', '2MS2', 'MKS2', '2MN2', 'MSN2', 'MO3', '2MK3', 'MK3', 'S6', 'M8', 'MSf', 'Ssa', 'Sa']
	#subset_const = ['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2']#, 'Eps2','M11','M12','MNS2','M13','2MS2','Mu2','MO3','2MK3']#['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2']#, 'Eps2', 'M11', 'M12', 'MNS2', 'M13', '2MS2']
	#descending order of energy in lowe inlet:
	subset_const = ['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2','M11', 'M13','M12','Eps2', 'MNS2','S1', 'SK4', 'R4', 'MO3', '2MK3', '2MS2', 'Mu2', 'Psi1', 'Nu2', 'MKS2', 'Pi1', 'Sa', 'Mf', 'Chi1', '2N2', 'Ssa', 'J1', 'Msqm', 'MSf', '2MN2', 'L2', 'Mm', 'MP1', 'M3', 'Phi1', 'R2', 'MS4', 'Eta2', 'Mtm', 'OO1', 'Sigma1', 'Rho1', '2Q1', 'T2', 'S4', 'Theta1', 'MN4', 'M4', 'MSN2', 'Lambda2', '2SM2', 'MK4', 'M6', '2MN6', 'S6', 'N4', '2MS6', 'MK3', 'SN4', '2SM6', '2MK6', 'MSN6', 'M8', 'MSK6']
	subset_const_ids = [all_const.index(i) for i in subset_const]	

	for location in ['Prince_Rupert','Lowe_Inlet','Hartley_Bay']:
		time = obs_df[location+'_date'].dropna().to_numpy()
		h = obs_df[location+'_val'].dropna().to_numpy()
		wt = pytide.WaveTable()
		f, vu = wt.compute_nodal_modulations(time)
		w = wt.harmonic_analysis(h, f, vu)
		w_short = w[subset_const_ids]
		wt_short = pytide.WaveTable(subset_const)
		hp = wt_short.tide_from_tide_series(time, w_short)
		if num_const is not None:
			new_obs_df = pd.DataFrame({
				location+'_date_'+str(num_const)+'_subset_constituents': time,
    			location+'_val_'+str(num_const)+'_subset_constituents': hp})
		else:
			new_obs_df = pd.DataFrame({
				location+'_date_subset_constituents': time,
    			location+'_val_subset_constituents': hp})
		obs_df = pd.concat([obs_df, new_obs_df], axis=1)

	print('Done removing all but a subset of constituents from the obs.')
	return obs_df

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

def model_tidal_signal(model_xy,model,yearmonth=False):
	'''Outputs the model sea surface variations, NOT centred around 0, for one month.
	Accepts tuple containing model x,y coordinates nearest a tide gauge.
	Returns dataarray containing ssh above geoid at the specified x,y coords.'''

	#function used to find the closest datetime within a list of datetime objects
	def closest_date(dt_list,d):
		delta_dates = [abs(dt_obj-d) for dt_obj in dt_list]
		id = delta_dates.index(min(delta_dates))
		return dt_list[id]

	#opening model surface data
	filepaths_txt = 'filepaths/' + model + '_filepaths_1h_grid_T_2D.csv'
	filepaths = list(pd.read_csv(filepaths_txt))#[100:102] for kit500 #[30:33] for grc100 first week of 2020
	if yearmonth==False: #ie if we want mid-2019 to mid-2021
		dates = [date(int(fp[-39:-35]),int(fp[-35:-33]),int(fp[-33:-31])) for fp in filepaths]
		start_date = closest_date(dates,date(2019,5,15)) #using date comfortably before the exact start time that we want
		end_date = closest_date(dates,date(2021,6,7)) #using date comfortably after the exact end time that we want
		filepath_df = pd.DataFrame({'filepaths': filepaths}, index=dates).loc[start_date:end_date]
		filepaths = filepath_df['filepaths'].to_list()
	else: #ie if we just want a month
		filepaths = [filepath for filepath in filepaths if yearmonth in filepath] #just for prototyping 
	model_preprocessor = lambda ds: ds[['zos']]#,'zos_ib']] #specify veriables to retrieve, zos_ib not neessary rn 	#zos is sea surface height above geoid, zos_ib is the inverse barometer ssh
	ds = xr.open_mfdataset(filepaths,preprocess=model_preprocessor)

	#extracting only the important points
	model_x, model_y = model_xy
	if yearmonth==False:
		ds = ds.sel(time_counter=slice(date(2019,6,1),date(2021,6,1)))
	#ds['zos_and_zos_ib'] = ds.zos + ds.zos_ib #wanted when comparing signal to obs
	da_at_gauge = ds['zos'].sel(x=model_x, y=model_y) #Can choose to look at only zos or zos_and_zos_ib

	print('Done; ssh data from '+model+' accessed')
	return da_at_gauge

def error(obs_df,location,model_da,model):
	'''Returns the RMS error of a model ssh compared to tide gauge obs.
	Probably need to expand this eventually to cover obs to obs error.'''

	model_da = model_da.drop_vars(['time_instant','nav_lat','nav_lon']).isel(x=0,y=0) #dropping unnecessary coords and vars from model da
	obs_df = obs_df[[location+'_date',location+'_val']].set_index(location+'_date') #dropping unnecessary obs data and setting dates to the index
	#note: when converting a df to a ds, the index becomes the coords
	ds = obs_df.to_xarray().rename({location+'_date':'time_counter',location+'_val':'obs_ssh'}) #turning obs df to a ds
	ds = ds.where(~np.isnan(ds['obs_ssh']),drop=True) #in case there are any nans in the obs data
	ds = ds.where(ds['time_counter']==model_da['time_counter']) #keeping obs data only where it overlaps in time with the model da
	ds['model_ssh'] = model_da #combining obs and model data
	ds['difference_squared'] = (ds.model_ssh - ds.obs_ssh)**2
	ds['rmse'] = (ds.difference_squared.mean(dim='time_counter'))**0.5
	print(model + ' error: ' + str(ds.rmse.to_numpy()))

	return ds.rmse.to_numpy()

def tide_plotting(obs_df,location,yearmonth,model_da,model,**kwargs):
	'''Plotting the observations and the model(s).
	Works for the first "period" (ie a week; can be changed) of a specified year and month.
	Can optionally pass model2_da and model2 to plot two models.'''

	#in case we want to plot two models instead of one
	model2_da = kwargs.get('model2_da', None)
	model2 = kwargs.get('model2', None)

	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert tidal water level \n modelling vs observations',
		'Hartley_Bay': 'Hartley Bay tidal water level \n modelling vs observations',
		'Lowe_Inlet': 'Lowe Inlet tidal water level \n modelling vs observations'}

	#prepping to plot
	fig,ax1 = plt.subplots(figsize=(6,5))
	c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

	#slicing the data
	year = yearmonth[:4]
	month = yearmonth[-2:]
	start_date = year+'-'+month+'-04'
	end_date = year+'-'+month+'-07'
	if location+'_date_subset_constituents' in obs_df.columns: 
		new_obs_df = obs_df[[location+'_date_subset_constituents',location+'_val_subset_constituents']]
		new_obs_df = new_obs_df.set_index(location+'_date_subset_constituents')[start_date:end_date]
	obs_df = obs_df[[location+'_date',location+'_val']]
	obs_df = obs_df.set_index(location+'_date')[start_date:end_date]
	model_da = model_da.sel(time_counter=slice(start_date,end_date))
	if model2: model2_da = model2_da.sel(time_counter=slice(start_date,end_date))
	
	#tide gauge data
	model_da.plot(ax=ax1, color=c2, label=model) #does it need to be centerd around 0?
	if model2: model2_da.plot(ax=ax1, color=c3, label=model2)
	obs_df.reset_index().plot(x=location+'_date', y=location+'_val', ax=ax1, color=c1, label='observations - full signal')
	try:
		new_obs_df.reset_index().plot(
			x=location+'_date_subset_constituents', 
			y=location+'_val_subset_constituents', 
			ax=ax1, 
			color=c1, 
			linestyle='dashed', 
			label='observations - rebuilt with 8 \nmodel constituents')		
		print('Also plotting rebuilt obs signals')
	except:
		print('Not plotting any rebuilt obs signals')

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
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax1.set_title(plot_titles[location])# + '\n' + start_date + ' - ' + end_date)
	ax1.set_ylabel('SSH ($m$)')

	#saving
	fig.savefig('tidal_figures/' 'tides_' + location + '_' + start_date + '_with_rebuilt_obs.png',dpi=300, bbox_inches="tight") #incl_zos_ib_
	fig.clf()

def obs_plotting_with_rebuilt_signals(obs_df,location,yearmonth):
	'''Plotting the observations, full signal and rebuilt signals.
	Hard coded for time-saving; could improve later!'''

	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert tidal water level \n observations - full vs rebuilt signals',
		'Hartley_Bay': 'Hartley Bay tidal water level \n observations - full vs rebuilt signals',
		'Lowe_Inlet': 'Lowe Inlet tidal water level \n observations - full vs rebuilt signals'}

	#prepping to plot
	fig,ax1 = plt.subplots(figsize=(6,5))
	c1, c2, c3, c4, c5 = plt.cm.viridis([0, 0.2, 0.4, 0.6, 0.8])

	#slicing the data
	year = yearmonth[:4]
	month = yearmonth[-2:]
	start_date = year+'-'+month+'-04'
	end_date = year+'-'+month+'-07'

	obs_df = obs_df.set_index(location+'_date')[start_date:end_date].reset_index()
	obs_df_8 = obs_df[[location+'_date_8_subset_constituents',location+'_val_8_subset_constituents']]
	obs_df_12 = obs_df[[location+'_date_12_subset_constituents',location+'_val_12_subset_constituents']]
	obs_df_17 = obs_df[[location+'_date_17_subset_constituents',location+'_val_17_subset_constituents']]
	obs_df_26 = obs_df[[location+'_date_26_subset_constituents',location+'_val_26_subset_constituents']]
	obs_df = obs_df[[location+'_date',location+'_val']]

	obs_df.plot(x=location+'_date', y=location+'_val', ax=ax1, color=c1, label='full signal')
	obs_df_8.plot(x=location+'_date_8_subset_constituents', y=location+'_val_8_subset_constituents', ax=ax1, color=c2, label='8 constituents')
	obs_df_12.plot(x=location+'_date_12_subset_constituents', y=location+'_val_12_subset_constituents', ax=ax1, color=c3, label='12 constituents')
	obs_df_17.plot(x=location+'_date_17_subset_constituents', y=location+'_val_17_subset_constituents', ax=ax1, color=c4, label='17 constituents')
	obs_df_26.plot(x=location+'_date_26_subset_constituents', y=location+'_val_26_subset_constituents', ax=ax1, color=c5, label='26 constituents')

	ax1.set_xlabel('Date')
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax1.set_title(plot_titles[location])# + '\n' + start_date + ' - ' + end_date)
	ax1.set_ylabel('SSH ($m$)')

	fig.savefig('tidal_figures/tides_' + location + '_' + start_date + '_with_5_rebuilt_obs.png',dpi=300, bbox_inches="tight") #incl_zos_ib_
	fig.clf()

def model_harmonic_analysis_plots(model_da,model,yearmonth,location):
	'''Breaks down the tidal signal from within a model dataarray and (currently) 
	creates a plot of the consituents; the exact utility of harmonic analysis on the
	model data can be fleshed out later.'''

	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert \n ' + model + ' tidal constituents',
		'Hartley_Bay': 'Hartley Bay \n ' + model + ' tidal constituents',
		'Lowe_Inlet': 'Lowe Inlet \n ' + model + ' tidal constituents'}

	#prepping to plot
	fig,ax1 = plt.subplots()
	c = plt.cm.viridis(np.linspace(0, 1, 8)) #8 constituents

	#prepping the data for harmonic analysis
	model_da = model_da.drop_vars(['time_instant','nav_lat','nav_lon']).isel(x=0,y=0)
	time = model_da.time_counter.to_numpy()
	h = model_da.to_numpy()

	#plotting model data
	if model=='grc100' and location=='Prince_Rupert': 
		print("Lowe Inlet isn't within the grc100 grid")
		quit()
	else:
		model_da.plot(ax=ax1, color='black', linestyle='dashed', label=model) #does it need to be centerd around 0?
	
	#plotting constituents
	for i,constituent in enumerate(['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2']):
		wt = pytide.WaveTable([constituent]) #these are the forcing constituents
		f, vu = wt.compute_nodal_modulations(time)
		w = wt.harmonic_analysis(h, f, vu)
		hp = wt.tide_from_tide_series(time, w)
		ax1.plot(time,hp,c=c[i],label=constituent)
	
	#horizontal axis stuff
	ax1.set_xlabel('Date')
	year = int(yearmonth[:4])
	month = int(yearmonth[-2:])
	ax1.set_xlim([date(year, month, 1), date(year, month, 8)])

	#standard plotting stuff
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax1.set_title(plot_titles[location])# + '\n' + start_date + ' - ' + end_date)
	ax1.set_ylabel('SSH ($m$)')

	#saving
	fig.savefig('tidal_figures/' + model + '_constituents_' + location + '.png',dpi=300, bbox_inches="tight")
	fig.clf()

	print('Done creating plot of model tidal constituents')

def observations_harmonic_analysis_plots(obs_df,yearmonth,location):
	'''Breaks down the tidal signal from tide gauges and (currently) 
	creates a plot of the consituents.'''

	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert \n tide gauge constituents',
		'Hartley_Bay': 'Hartley Bay \n tide gauge constituents',
		'Lowe_Inlet': 'Lowe Inlet \n tide gauge constituents'}

	#prepping to plot
	fig,ax1 = plt.subplots()
	#c = plt.cm.viridis(np.linspace(0, 1, 14)) #8 constituents

	#prepping the data for harmonic analysis
	time = obs_df[location+'_date'].to_numpy()
	h = obs_df[location+'_val'].to_numpy()
	time = time[~np.isnan(h)]
	h = h[~np.isnan(h)]

	#initial plot of the base data
	obs_df.plot(x=location+'_date', y=location+'_val', ax=ax1, color='black', label='observations - full signal')

	#plotting constituents
	const_list = []
	wt = pytide.WaveTable()#[constituent]) #these are the forcing constituents
	constituents = wt.constituents()
	f, vu = wt.compute_nodal_modulations(time)
	max_constituent_hp = []
	for i,constituent in enumerate(constituents):
		w = wt.harmonic_analysis(h, f, vu)
		new_w = np.array([w[i]]) 
		wt = pytide.WaveTable([constituent])
		hp = wt.tide_from_tide_series(time, new_w)
		max_constituent_hp.append(hp.max())
		if hp.max() > 1: 
			ax1.plot(time,hp,label=constituent) #c=c[i]
			const_list.append(hp)
		print(constituent + ' done')
	
	#== ugly code for presentation stats =================================#######################################################
	max_constituent_hp_df = pd.DataFrame({'max_hp': max_constituent_hp},index=list(constituents))
	max_constituent_hp_df.sort_values(by=['max_hp']).to_csv('CSVs/constituent_hp_'+location+'.csv')
	return

	#plotting the reconstructed obs
	const_df = pd.DataFrame(const_list).transpose()
	const_df = const_df.assign(sum=const_df.sum(axis=1))
	const_df = const_df.assign(t=time)
	const_df.plot(x='t', y='sum', ax=ax1, linestyle='dashed', label='observations - reconstructed')

	#horizontal axis stuff
	ax1.set_xlabel('Date')
	year = int(yearmonth[:4])
	month = int(yearmonth[-2:])
	ax1.set_xlim([date(year, month, 4), date(year, month, 8)])

	#standard plotting stuff
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	ax1.set_title(plot_titles[location])# + '\n' + start_date + ' - ' + end_date)
	ax1.set_ylabel('SSH ($m$)')

	#saving
	fig.savefig('tidal_figures/' + 'observational_constituents_' + location + '.png',dpi=300, bbox_inches="tight")
	fig.clf()

	print('Done creating plot of model tidal constituents')

def plot_periodogram(obs_df,kit500_da,location,**kwargs):
	'''Plots a power spectrum comparing obs and the models.
	A work in progress!'''

	#there's no grc100 data from Prince Rupert, so it isn't always passed in
	grc100_da = kwargs.get('grc100_da', None)

	#processing data
	if grc100_da is not None:
		grc100_da = grc100_da.isel(x=0,y=0).to_numpy()
	kit500_da = kit500_da.isel(x=0,y=0).to_numpy()
	if location+'_date_subset_constituents' in obs_df.columns: 
		subset_obs_df = obs_df[location+'_val_subset_constituents'].to_numpy()
		subset_obs_df = subset_obs_df[~np.isnan(subset_obs_df)]
	else: 
		subset_obs_df = None #basically a true/false value for future if statements
	obs_df = obs_df[location+'_val'].to_numpy()
	obs_df = obs_df[~np.isnan(obs_df)]
	
	#init the plot
	fig,ax1 = plt.subplots(figsize=(8,4)) 

	#getting PSD of the tidal signals
	obs_f, obs_pxx = signal.periodogram(obs_df)
	if grc100_da is not None: grc100_f, grc100_pxx = signal.periodogram(grc100_da)
	kit500_f, kit500_pxx = signal.periodogram(kit500_da)	
	if subset_obs_df is not None: 
		subset_obs_f, subset_obs_pxx = signal.periodogram(subset_obs_df)

	'''
	#removing all tidal constituent frequencies to see what happens
	#would need to rearrange things if you want to uncomment this
	for i,n in enumerate(all_freqs):
		freq = n*3600/(2*3.14159265359) #getting the frequency in hr^-1 (which is the PSD unit)
		min_freqs = list(np.absolute(obs_f-freq)) #list of the differences between the PSD freqs and the const freq
		id = min_freqs.index(min(min_freqs)) #id of the corresponding PSD freq to the constituent freq
		obs_pxx[id] = 0
	'''
	
	#plotting the PSDd
	c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])
	l1, = ax1.semilogy(1/obs_f[1:],obs_pxx[1:],c=c1,label='Tide gauge') #plot the data 
	if grc100_da is not None: l2, = ax1.semilogy(1/grc100_f[1:],grc100_pxx[1:],c=c2,label='grc100') #plot the data 
	l3, = ax1.semilogy(1/kit500_f[1:],kit500_pxx[1:],c=c3,label='kit500') #plot the data 
	if subset_obs_df is not None: l4, = ax1.semilogy(1/subset_obs_f[1:],subset_obs_pxx[1:],c='black',alpha=0.6,label='Tide gauge - rebuilt signal') #plot the data 
	ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

	#== hard coding for the presentation: getting the frequencies of the constituents and annotating a subset of them ===#
	'''
	wt=pytide.WaveTable()
	all_constituents = wt.constituents()
	all_freqs = wt.freq()
	subset_constituents = ['M2', 'K1', 'N2', 'S2', 'O1', 'P1', 'Q1', 'K2', 'Eps2','M11','M12','MNS2','M13','2MS2','Mu2','MO3','2MK3']
	subset_freqs = []
	for i,c in enumerate(subset_constituents): #getting the frequency/period of the subset of constituents that we're interested in
		id = all_constituents.index(c)
		subset_freqs.append(all_freqs[id]*3600/(2*3.14159265359)) #freq in hr^-1
	subset_pwr = []
	for i,f in enumerate(subset_freqs): #getting the power density of the constituents
		closest_freqs = [abs(obs_freq-f) for obs_freq in obs_f]
		id = closest_freqs.index(min(closest_freqs))
		subset_pwr.append(obs_pxx[id])
	c = plt.cm.viridis(np.linspace(0, 1, 8))
	for i,n in enumerate(subset_constituents): #plotting vertical lines at the 8 model constituents
		if i < 9: ax1.plot(1/subset_freqs[i], subset_pwr[i], 'o', color='black')
		else: ax1.plot(1/subset_freqs[i], subset_pwr[i], 'o', color='orange')
	'''

	#== hard(ish) coding this for presentation plots ==############################################################################
	'''
	def annotate_constituents(i,h,v,colour):
		ax1.annotate(subset_constituents[i], xy=(1/subset_freqs[i], subset_pwr[i]),
				xytext=(1/subset_freqs[i]+h, subset_pwr[i]+v),
				arrowprops=dict(color=colour, shrink=0.05, width=1, headwidth=5, headlength=6), 
				ha='right', 
				va='center')
	annotate_constituents(0,3,0,'black') #M2
	annotate_constituents(1,-2,0,'black') #K1
	annotate_constituents(2,3,5000,'black') #N2
	annotate_constituents(3,-2,0,'black') #S2
	annotate_constituents(4,3,0,'black') #O1
	annotate_constituents(5,-2,0,'black') #P1
	annotate_constituents(6,3,0,'black') #Q1
	annotate_constituents(7,-2,0,'black') #K2
	annotate_constituents(8,3,-650,'orange') #Eps2 
	annotate_constituents(9,3,0,'orange') #M11
	annotate_constituents(10,-2,0,'orange') #M12
	annotate_constituents(11,3.5,1000,'orange') #MNS2
	annotate_constituents(12,-2,0,'orange') #M13
	annotate_constituents(13,3,-150,'orange') #2MS2
	annotate_constituents(14,3,-235,'orange') #Mu2
	annotate_constituents(15,-2,1000,'orange') #MO3
	annotate_constituents(16,3,-300,'orange') #2MK3
	'''
	'''
	wt=pytide.WaveTable()
	all_constituents = wt.constituents()
	all_freqs = wt.freq()
	all_freqs = [f*3600/(2*3.14159265359) for f in all_freqs]
	constituents_pwr = []
	for i,f in enumerate(all_freqs): #getting the power density of the constituents
		closest_freqs = [abs(obs_freq-f) for obs_freq in obs_f]
		id = closest_freqs.index(min(closest_freqs))
		constituents_pwr.append(obs_pxx[id])
	constituents_df = pd.DataFrame({'PD': constituents_pwr},index=list(all_constituents))
	constituents_df.sort_values(by=['PD']).to_csv('CSVs/constituent_PD_'+location+'.csv')
	'''
	#=============================================================================================================#
	
	#dictionary of plot titles
	plot_titles = {
		'Prince_Rupert': 'Prince Rupert tidal power spectra\nmodels vs observations',
		'Hartley_Bay': 'Hartley Bay tidal power spectra\nmodels vs observations',
		'Lowe_Inlet': 'Lowe Inlet tidal power spectra\nmodels vs observations'}

	#standard plotting params
	plt.title(plot_titles[location])
	ax1.set_ylabel('PSD')
	ax1.set_xlabel('Period ($hr$)')# ($hr^{-1}$)')
	ax1.set_xlim(2,30) #set xlims for aesthetics (30 hrs)
	ax1.set_ylim(0.000001,100000)
	fig.savefig('tidal_figures/ps_'+location+'_incl_rebuilt_signal.png',dpi=300, bbox_inches="tight")
	fig.clf()

if __name__ == "__main__":

	#== this section is for looking at the power spectra ==#
	'''
	locations=['Lowe_Inlet']#,'Prince_Rupert','Hartley_Bay']
	obs_df = get_tide_gauge_obs()
	obs_df = tide_gauge_rebuilt_signal(obs_df)
	yearmonth='202006' #should eventually extend this to the full series?
	for location in locations:
		kit500_xy = find_nearest_model_ids(location,'kit500')
		kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
		if location != 'Prince_Rupert':
			grc100_xy = find_nearest_model_ids(location,'grc100')
			grc100_da = model_tidal_signal(grc100_xy,'grc100',yearmonth)
			plot_periodogram(obs_df,kit500_da,location,grc100_da=grc100_da)
		else:
			plot_periodogram(obs_df,kit500_da,location)
	'''

	#== this section is for harmonic analysis and plotting constituents on the obs data ==#
	'''
	yearmonth='202006'
	locations = ['Lowe_Inlet','Prince_Rupert','Hartley_Bay']
	obs_df = get_tide_gauge_obs()
	for location in locations:
		observations_harmonic_analysis_plots(obs_df,yearmonth,location)
	'''

	#== this section is for harmonic analysis on the model data ==# 
	'''
	yearmonth='202006'
	location = 'Lowe_Inlet'

	grc100_xy = find_nearest_model_ids(location,'grc100')
	grc100_da = model_tidal_signal(grc100_xy,'grc100',yearmonth)
	model_harmonic_analysis_plots(grc100_da,'grc100',yearmonth,location)
	kit500_xy = find_nearest_model_ids(location,'kit500')
	kit500_da = model_tidal_signal(kit500_xy,'kit500',yearmonth)
	model_harmonic_analysis_plots(kit500_da,'kit500',yearmonth,location)
	quit()
	'''

	#== this section is for computing the model-obs error ==#
	'''
	#yearmonth='202006'
	obs_df = get_tide_gauge_obs()
	for location in ['Hartley_Bay','Lowe_Inlet','Prince_Rupert']:
		kit500_xy = find_nearest_model_ids(location,'kit500')
		kit500_da = model_tidal_signal(kit500_xy,'kit500')#,yearmonth,fulltimeseries=True)
		kit500_err = error(obs_df,location,kit500_da,'kit500')
		f = open('error_files/'+location+'_error.txt','w')
		if location != 'Prince_Rupert': #no grc100 data at prince rupert!
			grc100_xy = find_nearest_model_ids(location,'grc100')
			grc100_da = model_tidal_signal(grc100_xy,'grc100')#,yearmonth,fulltimeseries=True)
			grc100_err = error(obs_df,location,grc100_da,'grc100')
			f.write(location+' error:\n'+'kit500: '+str(kit500_err)+'\ngrc100: '+str(grc100_err))
		else:
			f.write(location+' error:\n'+'kit500: '+str(kit500_err))
		f.close()
	'''

	#== and this section is for making plots of ssh ==#
	'''
	yearmonth='202101'
	obs_df = get_tide_gauge_obs()
	
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
	'''

	#== this section is pretty hard coded (FIX THIS); plots rebuilt tide water signals ==#

	yearmonth='202101'
	location = 'Lowe_Inlet'
	obs_df = get_tide_gauge_obs()
	obs_df = tide_gauge_rebuilt_signal(obs_df,num_const=8) #can optionally comment this out, if you don't want the rebuilt obs
	obs_df = tide_gauge_rebuilt_signal(obs_df,num_const=12) #can optionally comment this out, if you don't want the rebuilt obs
	obs_df = tide_gauge_rebuilt_signal(obs_df,num_const=17) #can optionally comment this out, if you don't want the rebuilt obs
	obs_df = tide_gauge_rebuilt_signal(obs_df,num_const=26) #can optionally comment this out, if you don't want the rebuilt obs
	obs_plotting_with_rebuilt_signals(obs_df,location,yearmonth)
	
