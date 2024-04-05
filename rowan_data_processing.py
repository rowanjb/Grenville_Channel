# For calculating model vs error (gauges, 2x ADCPs, 2x tidal packages at Lowe Inlet)
# and times series of model ssh at different locations, and currents at the ADCP 

print('importing packages...')
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime,timedelta,date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pytide
from utide import solve, reconstruct
from scipy import signal
import os
print('done importing')

def open_tide_gauge(location): 
    '''Opens the tide gauge data.
    location can be 'Lowe_Inlet', 'Prince_Rupert', or 'Hartley_Bay'.
    Returns dataframe of the tide gauge data.'''
    obs_file_paths = {
        'Prince_Rupert': '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/PrinceRupert_Jun012019-Nov202023.csv', 
        'Hartley_Bay': '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/HartleyBay_Jun012019-Nov202023.csv', 
        'Lowe_Inlet': '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/LoweInlet_2014_forConstituents.csv'}
    obs_path = obs_file_paths[location]
    df = pd.read_csv(obs_path, sep=",", header=None,engine='python') 
    df = df.drop(index=range(0,8), columns=[2]).reset_index().rename(columns={0: 'date', 1: 'h'}).drop(columns='index') # Dropping metadata
    df['h'] = df['h'].astype('float') # Convert ssh from str to float
    df['date'] = pd.to_datetime(df['date']) # And convert str date to datetime
    print('Observational data has been opened')
    return df

def open_ADCP_ssh(which_adcp):
    '''Opens the ADCP data.
    which_adcp permits choice of the 'upper' or 'lower' unit.
    Returns dataframe of the ssh observations.'''
    obs_paths = {
        'upper': '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/grc1_20190809_20200801_0027m.adcp.L1.nc',
        'lower': '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/grc1_20190808_20200801_0089m.adcp.L1.nc'}
    ds = xr.open_dataset(obs_paths[which_adcp]) # opening the dataset
    h = ds.PPSAADCP.dropna(dim='time') # note: turns the dataset into a dataarray; nans are also dropped 
    h = h.resample(time="H").mean() # putting data onto hourly grid
    df = h.to_pandas().reset_index().rename(columns={'time': 'date', 0: 'h'})
    print('ADCP ssh data has been opened')
    return df

def open_ADCP_current(depth='Full'):
    '''Opens the ADCP data.
    which_adcp permits choice of the 'upper' or 'lower' unit.
    Returns dataframe of the current velocity observations.'''
    print('ADCP current velocity data has been opened')
    if (depth=='Full') | (depth=='Full_column'):
        obs_path = '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/grc1_20190808_20200801_0089m.adcp.L1.nc'
        print('Velocities are coming from the lower ADCP')
    elif float(depth) > 25:
        obs_path = '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/grc1_20190808_20200801_0089m.adcp.L1.nc'
        print('Velocities are coming from the lower ADCP')
    elif float(depth) < 25:
        obs_path = '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/grc1_20190809_20200801_0027m.adcp.L1.nc'
        print('Velocities are coming from the upper ADCP')
    ds = xr.open_dataset(obs_path) #opening the dataset
    ew = ds['LCEWAP01'].resample(time="H").mean()#.dropna(dim='time') # Note using LCEWAP01_QC doesn't work
    ns = ds['LCNSAP01'].resample(time="H").mean()#.dropna(dim='time') # also note dropping nans looses you a lot of data
    #err = ds['LERRAP01'].resample(time="H").mean()#.dropna(dim='time')
    distances_to_surface = ds['DISTTRAN']
    
    ds = ew.to_dataset().rename({'LCEWAP01':'ew'})
    ds['ns'] = ns
    ds['distance'] = distances_to_surface
    
    if depth=='Full':
        ds = ds.mean(dim='distance',skipna=True).dropna(dim='time')
        print('ADCP velocities calculated and averaged in the water column')
    if depth=='Full_column':
        ds = ds.dropna(dim='time')
        print('ADCP velocities through the water column have been output')
    else:
        ds = ds.sel(distance=float(depth),method='nearest').dropna(dim='time').drop_vars('distance')
        print('ADCP velocities calculated indexed nearest '+str(depth)+' m')
    
    print('ADCP ssh data has been opened')
    return ds

def get_nearest_model_id(location,model):
    '''Returns the nearest model ID to a chosen location.
    location can be 'Lowe_Inlet', 'Prince_Rupert', Hartley_Bay', or 'ADCP'.
    model can be 'grc100' or 'kit500'. 
    Returns the model x, y coordinates closest to the specified observation.'''
    obs_coords = {                                              #Coords from Google:
        'Hartley_Bay': {'lat': 53.424213, 'lon': -129.251877}, 	#53.4239N 129.2535W 
        'Lowe_Inlet': {'lat': 53.55, 'lon': -129.583333},		#53.5564N 129.5875W 
        'Prince_Rupert': {'lat': 54.317, 'lon': -130.324},		#54.3150N 130.3208W 
        'ADCP': {'lat': 53.545945, 'lon': -129.617236666667},
        'Channel_Inlet': {'lat': 53.909153, 'lon': -130.114036},
        'Channel_Outlet': {'lat': 53.363713, 'lon': -129.322383}}
    model_fp_txt = 'filepaths/' + model+ '_filepaths_1h_grid_T_2D.csv'
    fp = list(pd.read_csv(model_fp_txt))[10] # Path to a random nc model file
    ds = xr.open_dataset(fp).isel(time_counter=5) # Picking a random time to save on memory 
    nav_lat = xr.where(~np.isnan(ds.zos),ds.nav_lat,0) # Essentially zeroing coordinates where zos is nan;
    nav_lon = xr.where(~np.isnan(ds.zos),ds.nav_lon,0) # will stop the "nearest coordinate" from being in land
    def shortest_distance(lat,lon): # Finds indices of the grid point nearest a given location  
        abslat = np.abs(nav_lat.to_numpy()-lat)
        abslon = np.abs(nav_lon.to_numpy()-lon)
        c = (abslon**2 + abslat**2)**0.5
        return np.where(c == np.min(c))
    model_y, model_x = shortest_distance(obs_coords[location]['lat'],obs_coords[location]['lon']) # Indices of tide gauges on model grid
    model_lat = ds.nav_lat.sel(x=model_x,y=model_y).to_numpy() # Corresponding latitude on the model grid, and
    model_lon = ds.nav_lon.sel(x=model_x,y=model_y).to_numpy() # corresponding longitude on the model grid, and
    print('-> ' + location + ' observation coords: ' + str(obs_coords[location]['lat']) + ', ' + str(obs_coords[location]['lon']))
    print('-> Corresponding xy ids on ' + model + ' grid: ' + str(model_x[0]) + ', ' + str(model_y[0]))
    print('-> Corresponding zos on ' + model + ' grid: ' + str(ds.zos[model_y[0],model_x[0]].to_numpy())) #to ensure not zero
    print('-> Corresponding ' + model + ' coords: ' + str(model_lat[0][0]) + ', ' + str(model_lon[0][0]))
    print('Done; nearest '+model+' coords to the specified location have been found')
    return model_x, model_y

def model_ssh_signal(model_xy,model):
    '''Obtains the ssh signal within a model between 2019/6/1 and 2023/1/1 (ie the same dates as the tide gauges).
    model_xy is the x, y coordinate tuple from 'get_nearest_model_id'.
    model is either 'grc100' or 'kit500'.
    Returns a dataarray containing the timeseries ssh data.'''
    start_date = datetime(2019,6,1) # Rough starting date of the Hartley Bay and Prince Rupert data
    end_date = datetime(2024,1,1) #APRIL 3 NOTE: DATE UPDATED TO MATCH WEATHER STATION
    model_fp_txt = 'filepaths/' + model + '_filepaths_1h_grid_T_2D.csv'
    fps = list(pd.read_csv(model_fp_txt))
    dates = [datetime(int(fp[-39:-35]),int(fp[-35:-33]),int(fp[-33:-31])) for fp in fps]
    fp_df = pd.DataFrame({'filepaths': fps}, index=dates).loc[start_date:end_date]
    fps = fp_df['filepaths'].to_list()
    model_preprocessor = lambda ds: ds[['zos','zos_ib']]
    ds = xr.open_mfdataset(fps,preprocess=model_preprocessor)
    ds['ssh'] = ds.zos + ds.zos_ib # Can also comment out the second term
    da = ds['ssh'].sel(x=model_xy[0], y=model_xy[1]) 
    print('Done; ssh data from '+model+' accessed')
    return da.isel(x=0,y=0)

def model_current(model_xy,model,depth='Full'): 
    '''Obtains the u and v velocties within a model between 2019/08/08 and 2020/08/01 (ie the same dates as the ADCP).
    model_xy is the x, y coordinate tuple from 'get_nearest_model_id.
    depth is 'Full' by default; returns averaged velocities in depth.
    depth can be specified, e.g., '8.1' for comparison to the ADCP at 8.1 m.
    model is either 'grc100' or 'kit500'...
    HOWEVER, current quits if model!='grc100' because u and v in kit500 are not e-w and n-s.
    Functions by interpolating the velocities and then picking out the correct x and y (explains why it's slow).
    Returns a column dataset with u and v vels at the specified x and y for every hour.'''
    if model!='grc100':
        print('Only acceptable model currently is grc100')
        quit()
    start_date = datetime(2019,8,8)
    end_date = datetime(2020,8,1)

    model_mesh_fp = '/mnt/storage3/tahya/DFO/'+model+'_config/mesh_mask.nc'
    mesh = xr.open_dataset(model_mesh_fp).rename({'z':'d'})
    
    model_fp_txt_gridU = 'filepaths/'+model+'_filepaths_1h_grid_U.csv'
    model_fp_list_gridU = list(pd.read_csv(model_fp_txt_gridU))
    dates_gridU = [datetime(int(fp[-36:-32]),int(fp[-32:-30]),int(fp[-30:-28])) for fp in model_fp_list_gridU]
    fp_df_gridU = pd.DataFrame({'filepaths': model_fp_list_gridU}, index=dates_gridU).loc[start_date:end_date]
    model_fp_list_gridU = fp_df_gridU['filepaths'].to_list()
    model_preprocessor = lambda ds: ds[['uo']]
    dsu = xr.open_mfdataset(model_fp_list_gridU,preprocess=model_preprocessor).rename({'depthu':'d'})

    model_fp_txt_gridV = 'filepaths/'+model+'_filepaths_1h_grid_V.csv'
    model_fp_list_gridV = list(pd.read_csv(model_fp_txt_gridV))
    dates_gridV = [datetime(int(fp[-36:-32]),int(fp[-32:-30]),int(fp[-30:-28])) for fp in model_fp_list_gridV]
    fp_df_gridV = pd.DataFrame({'filepaths': model_fp_list_gridV}, index=dates_gridV).loc[start_date:end_date]
    model_fp_list_gridV = fp_df_gridV['filepaths'].to_list()
    model_preprocessor = lambda ds: ds[['vo']]
    dsv = xr.open_mfdataset(model_fp_list_gridV,preprocess=model_preprocessor).rename({'depthv':'d'})

    dsu['old_x'] = dsu.x
    dsu['old_y'] = dsu.y
    dsu = dsu.where((dsu.x>(model_xy[0][0]-2))&(dsu.x<(model_xy[0][0]+2))&(dsu.y>(model_xy[1][0]-2))&(dsu.y<(model_xy[1][0]+2)),drop=True)
    dsv = dsv.where((dsv.x>(model_xy[0][0]-2))&(dsv.x<(model_xy[0][0]+2))&(dsv.y>(model_xy[1][0]-2))&(dsv.y<(model_xy[1][0]+2)),drop=True)
    #mesh = mesh.where((mesh.x>(model_xy[0][0]-2))&(mesh.x<(model_xy[0][0]+2))&(mesh.y>(model_xy[1][0]-2))&(mesh.y<(model_xy[1][0]+2)),drop=True)
    old_x = dsu.old_x
    old_y = dsu.old_y
    dsu = dsu.interp(x = dsu.x - 0.5)
    dsv = dsv.interp(y = dsv.y - 0.5)

    ds = dsu['uo'].to_dataset()
    ds['vo'] = dsv['vo']
    #ds['nav_lat'] = mesh['nav_lat']
    #ds['nav_lon'] = mesh['nav_lon']
    
    ds = ds.where((old_x==model_xy[0][0])&(old_y==model_xy[1][0]),drop=True)
    ds = ds.assign_coords(nav_lat=mesh.nav_lat.sel(x=model_xy[0][0], y=model_xy[1][0]))
    ds = ds.assign_coords(nav_lon=mesh.nav_lon.sel(x=model_xy[0][0], y=model_xy[1][0]))
    
    if depth=='Full':
        ds = ds.where(mesh.tmask.sel(x=model_xy[0], y=model_xy[1]).isel(t=0)==1,drop=True) # Masking below seafloor
        weights = ds.d
        weights.name = "weights"
        uo_weighted = ds.uo.weighted(weights)
        vo_weighted = ds.vo.weighted(weights)
        ds['uo'] = uo_weighted.mean('d')
        ds['vo'] = vo_weighted.mean('d')
        ds = ds.isel({'x':0,'y':0}).drop_vars('d')
        print('Model x and y velocities calculated and averaged in the water column')
    elif depth=='Full_column':
        ds = ds.where(mesh.tmask.sel(x=model_xy[0], y=model_xy[1]).isel(t=0)==1,drop=True) # Masking below seafloor
        ds = ds.isel({'x':0,'y':0})
        print('Model x and y velocities calculated')
    else:
        ds = ds.sel(d=float(depth),method='nearest').isel({'x':0,'y':0})
        print('Model x and y velocities calculated indexed nearest '+str(depth)+' m')
    return ds

def date_range(date1, date2): 
    '''Creates np array of hourly datetime objects between two dates.'''
    delta = timedelta(hours=1)
    dates = [date1]
    while dates[-1] < date2:
        dates.append(dates[-1] + delta)
    dates = np.array(dates, dtype='datetime64')
    return dates

def extend_Lowe_Inlet_pytide(obs_df):
    '''Extends the SSH signal at Lowe Inlet to match the dates 
    from the gauges at Hartley Bay and Prince Rupert.
    This uses the pytide package.
    obs_df is the raw data from DFO, as formatted by 'open_tide_gauge'.
    Returns the dataframe with but with updated dates.'''
    start_date = datetime(2019,6,1) # Rough starting date of the Hartley Bay and Prince Rupert data
    end_date = datetime(2023,1,1)   # and the rough end date
    date_nparr = date_range(start_date,end_date) # Getting np array of hourly datetime objs between the start and end dates
    wt = pytide.WaveTable()
    f, vu = wt.compute_nodal_modulations(obs_df['date'].to_numpy())
    w = wt.harmonic_analysis(obs_df['h'].to_numpy(), f, vu)
    hp = wt.tide_from_tide_series(date_nparr, w)
    obs_df_new = pd.DataFrame({'date': date_nparr, 'h': hp})
    print('Lowe Inlet tidal observations have been extended forward using pytide')
    return obs_df_new

def extend_Lowe_Inlet_utide(obs_df):
    '''Extends the SSH signal at Lowe Inlet to match the dates 
    from the gauges at Hartley Bay and Prince Rupert.
    This uses the utide package.
    obs_df is the raw data from DFO, as formatted by 'open_tide_gauge'.
    Returns the dataframe with but with updated dates.'''
    start_date = datetime(2019,6,1) # Rough starting date of the Hartley Bay and Prince Rupert data
    end_date = datetime(2023,1,1)   # and the rough end date
    date_nparr = date_range(start_date,end_date) # Getting list of hourly datetime objs between the start and end dates
    coef = solve(obs_df['date'].to_numpy(),obs_df['h'].to_numpy(),lat=53.55,method="ols",conf_int="MC",verbose=False)
    tide = reconstruct(date_nparr, coef, verbose=False)
    obs_df_new = pd.DataFrame({'date': date_nparr, 'h': tide['h']})
    print('Lowe Inlet tidal observations have been extended forward using utide')
    return obs_df_new

def extend_SSH_obs_2023(obs_df,location):
    '''Extends an SSH signal to match the period from the Tom Island weather station.
    This uses the utide package.
    obs_df is the raw data from DFO.
    Returns the dataframe with but with updated dates.'''
    start_date = datetime(2023,1,1) # Rough starting date of the weather station
    end_date = datetime(2023,12,31) # and the rough end date
    date_nparr = date_range(start_date,end_date) # Getting list of hourly datetime objs between the start and end dates
    obs_coords = {                                              #Coords from Google:
        'Hartley_Bay': {'lat': 53.424213, 'lon': -129.251877}, 	#53.4239N 129.2535W 
        'Lowe_Inlet': {'lat': 53.55, 'lon': -129.583333},		#53.5564N 129.5875W 
        'ADCP': {'lat': 53.545945, 'lon': -129.617236666667}}
    obs_df['h'] = obs_df['h'] - obs_df['h'].mean()
    coef = solve(obs_df['date'].to_numpy(),obs_df['h'].to_numpy(),lat=obs_coords[location]['lat'],method="ols",conf_int="MC",verbose=False)
    tide = reconstruct(date_nparr, coef, verbose=False)
    obs_df_new = pd.DataFrame({'date': date_nparr, 'h': tide['h']})
    print(location + ' SSH observations have been extended forward using utide')
    return obs_df_new

def ssh_error(obs_fp,model_fp):
    '''Calculates the timeseries of error between an observation timeseries
    and a model timeseries.
    obs_fp is the .csv from, for ex., 'open_tide_gauge'.
    model_fp is the .nc from 'model_ssh_signal'.
    Returns the timeseries as a datarray, which can then be saved or used 
    to calculate the long-term average error.
    '''
    obs_df = pd.read_csv(obs_fp).set_index('date')  # Open the obs and set the dates to the index
    obs_df['h'] = obs_df['h'] - obs_df['h'].mean()  # Subtract the mean
    obs_df.index = pd.to_datetime(obs_df.index)
    model_da = xr.open_dataarray(model_fp)          # Open the model data
    model_df = model_da.to_pandas()                 # Convert to a dataframe
    model_df = model_df - model_df.mean()           # Subtract the mean
    idx = obs_df.index.intersection(model_df.index) # Find the common indices 
    ssh = obs_df.loc[idx].rename(columns={'h':'obs_h'}) # Create a final dataframe based on the obs h
    ssh['model_h'] = model_df.loc[idx]              # Add col for the mdel h
    ssh['difference'] = ssh['obs_h'] - ssh['model_h']
    ssh['abs_err'] = np.abs(ssh['difference']) # Add col for the abs(error)
    return ssh    

def vel_error(obs_fp,model_fp):
    '''Calculates the timeseries of error between an ADCP observation timeseries
    and a model timeseries
    obs_fp is the .csv from, for ex., 'open_tide_gauge'.
    model_fp is the .nc from 'model_ssh_signal'.
    Returns the timeseries as a datarray, which can then be saved or used 
    to calculate the long-term average error.
    '''
    obs_ds = xr.open_dataset(obs_fp)#pd.read_csv(obs_fp).set_index('date')  # Open the obs and set the dates to the index
    obs_df = obs_ds.to_pandas().rename(columns={'ew':'obs_ew','ns':'obs_ns'})
    model_ds = xr.open_dataset(model_fp)
    model_df = model_ds.to_pandas().rename(columns={'uo':'model_ew','vo':'model_ns'})
    idx = obs_df.index.intersection(model_df.index)
    vels = pd.concat([obs_df.loc[idx],model_df['model_ew'].loc[idx],model_df['model_ns'].loc[idx]],axis=1)
    vels['diff_ew'] = vels['obs_ew'] - vels['model_ew']
    vels['diff_ns'] = vels['obs_ns'] - vels['model_ns']
    vels['err_speed'] = (vels['diff_ns']**2 + vels['diff_ew']**2)**0.5
    return vels

def weather_station():
    '''Looks at the weather station data.
    Returns two df, 'wind' and 'pressure'.
    Each df contains the associated data (speed and dir, and pressure, respectively)
    along with bins of said data, in order to correlate it to other data.'''
    
    #WS_ms_S_WVT - mean windspeed in meter/second
    #WindDir_D1_WVT - mean wind direction in degrees
    #WindDir_SD1_WVT - standard deviation of wind direction
    #TiltNS_deg_Max
    #TiltWE_deg_Max
    #BP_mbar_Avg
    print('Opening the weather station data')
    fp = '/mnt/storage3/tahya/DFO/Observations/Weather_Station/TomIsland_hourlyData_Feb012023-Nov202023.csv'
    obs_df = pd.read_csv(fp)
    obs_df.columns = obs_df.iloc[0]
    
    obs_df = obs_df.drop(index=[0,1,2]).reset_index(drop=True)
    obs_df['TIMESTAMP'] = pd.to_datetime(obs_df['TIMESTAMP'])
    obs_df = obs_df.set_index('TIMESTAMP')
    obs_df = obs_df.astype(float)

    pressure_bins = pd.cut(obs_df['BP_mbar_Avg'],4,labels=['1','2','3','4'])
    pressure = pd.DataFrame(data={'pressure':obs_df['BP_mbar_Avg'],'pressure_bins':pressure_bins})
    
    wind_speed_bins = pd.cut(obs_df['WS_ms_S_WVT'],4,labels=['1','2','3','4'])
    wind_dir_bins = pd.cut(obs_df['WindDir_D1_WVT'],4,labels=['NE','SE','SW','NW'])
    wind = pd.DataFrame(data={'speed':obs_df['WS_ms_S_WVT'],'deg_CW_from_north':obs_df['WindDir_D1_WVT'],'speed_bins':wind_speed_bins,'dir_bins':wind_dir_bins})
    print('Outputting the processed weather station data')
    return wind, pressure

def save_tide_data(which_data):
    '''Saves the observation and/or model ssh data (or velocity data).
    which_data specifies the location and whether you want obs, model, etc.
    Observations are saved as CSVs.
    Model data are saved as netcdfs.'''

    print("We're lookin' at the following SSH signal: " + which_data)
    if which_data == 'Hartley_Bay_tide_gauge':
        obs_df = open_tide_gauge('Hartley_Bay')
        obs_df.to_csv('processed_data/SSH_Hartley_Bay_gauge.csv',index=False)
    elif which_data == 'Prince_Rupert_tide_gauge':
        obs_df = open_tide_gauge('Prince_Rupert')
        obs_df.to_csv('processed_data/SSH_Prince_Rupert_gauge.csv',index=False)
    elif which_data == 'Lowe_Inlet_tide_gauge':
        obs_df = open_tide_gauge('Lowe_Inlet')
        obs_df.to_csv('processed_data/SSH_Lowe_Inlet_gauge.csv',index=False)
    elif which_data == 'Lowe_Inlet_tide_gauge_pytide':
        obs_df = open_tide_gauge('Lowe_Inlet')
        obs_df = extend_Lowe_Inlet_pytide(obs_df)
        obs_df.to_csv('processed_data/SSH_Lowe_Inlet_pytide.csv',index=False)
    elif which_data == 'Lowe_Inlet_tide_gauge_utide':
        obs_df = open_tide_gauge('Lowe_Inlet')
        obs_df = extend_Lowe_Inlet_utide(obs_df)
        obs_df.to_csv('processed_data/SSH_Lowe_Inlet_utide.csv',index=False)
    elif which_data == 'ADCP_upper_SSH':
        obs_df = open_ADCP_ssh('upper')
        obs_df.to_csv('processed_data/SSH_ADCP_upper.csv',index=False)
    elif which_data == 'ADCP_lower_SSH':
        obs_df = open_ADCP_ssh('lower')
        obs_df.to_csv('processed_data/SSH_ADCP_lower.csv',index=False)
    elif which_data == 'Prince_Rupert_kit500_SSH':
        model_xy = get_nearest_model_id('Prince_Rupert','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_Prince_Rupert.nc')
    elif which_data == 'Hartley_Bay_kit500_SSH':
        model_xy = get_nearest_model_id('Hartley_Bay','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_Hartley_Bay.nc')
    elif which_data == 'Lowe_Inlet_kit500_SSH':
        model_xy = get_nearest_model_id('Lowe_Inlet','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_Lowe_Inlet.nc')
    elif which_data == 'ADCP_kit500_SSH':
        model_xy = get_nearest_model_id('ADCP','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_ADCP.nc')
    elif which_data == 'Hartley_Bay_grc100_SSH':
        model_xy = get_nearest_model_id('Hartley_Bay','grc100')
        model_da = model_ssh_signal(model_xy,'grc100')
        model_da.to_netcdf('processed_data/SSH_grc100_Hartley_Bay.nc')
    elif which_data == 'Lowe_Inlet_grc100_SSH':
        model_xy = get_nearest_model_id('Lowe_Inlet','grc100')
        model_da = model_ssh_signal(model_xy,'grc100')
        model_da.to_netcdf('processed_data/SSH_grc100_Lowe_Inlet.nc')
    elif which_data == 'ADCP_grc100_SSH':
        model_xy = get_nearest_model_id('ADCP','grc100')
        model_da = model_ssh_signal(model_xy,'grc100')
        model_da.to_netcdf('processed_data/SSH_grc100_ADCP.nc')
    elif which_data == 'ADCP_grc100_velocity':
        print("Note: we're looking at the average vel in the water column (can change manually; ensure you also change the err save fp)")
        model_xy = get_nearest_model_id('ADCP','grc100')
        model_ds = model_current(model_xy,'grc100',depth='Full_column')
        model_ds.to_netcdf('processed_data/velocities_grc100_ADCP_full_column.nc')
    elif which_data == 'ADCP_velocity':
        print("Note: we're looking at the average vel in the water column (can change manually; ensure you also change the err save fp)")
        obs_ds = open_ADCP_current(depth='Full_column') 
        obs_ds.to_netcdf('processed_data/velocities_ADCP_full_column.nc')
    elif which_data == 'Channel_Inlet_grc100_SSH':
        model_xy = get_nearest_model_id('Channel_Inlet','grc100')
        model_da = model_ssh_signal(model_xy,'grc100')
        model_da.to_netcdf('processed_data/SSH_grc100_channel_inlet.nc')
    elif which_data == 'Channel_Outlet_grc100_SSH':
        model_xy = get_nearest_model_id('Channel_Outlet','grc100')
        model_da = model_ssh_signal(model_xy,'grc100')
        model_da.to_netcdf('processed_data/SSH_grc100_channel_outlet.nc')
    elif which_data == 'Channel_Inlet_kit500_SSH':
        model_xy = get_nearest_model_id('Channel_Inlet','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_channel_inlet.nc')
    elif which_data == 'Channel_Outlet_kit500_SSH':
        model_xy = get_nearest_model_id('Channel_Outlet','kit500')
        model_da = model_ssh_signal(model_xy,'kit500')
        model_da.to_netcdf('processed_data/SSH_kit500_channel_outlet.nc') 
    elif which_data == 'Lowe_Inlet_extension_2023':
        obs_df = open_tide_gauge('Lowe_Inlet')
        obs_df = extend_SSH_obs_2023(obs_df,'Lowe_Inlet')
        obs_df.to_csv('processed_data/SSH_Lowe_Inlet_2023.csv',index=False)
    elif which_data == 'Hartley_Bay_extension_2023':
        obs_df = open_tide_gauge('Hartley_Bay')
        obs_df = extend_SSH_obs_2023(obs_df,'Hartley_Bay')
        obs_df.to_csv('processed_data/SSH_Hartley_Bay_2023.csv',index=False)
    elif which_data == 'ADCP_extension_2023':
        obs_df = open_ADCP_ssh('lower')
        obs_df = extend_SSH_obs_2023(obs_df,'ADCP')
        obs_df.to_csv('processed_data/SSH_ADCP_lower_2023.csv',index=False)
    else:
        'Bad choice of data!'
        quit()
    print(which_data+' data have been saved')

def save_err_data(which_data):
    '''Saves the error data between obs and models.
    which_data specifies the data series, location, and type (i.e., ssh vs vel)
    Error is saved as a csv, probably.'''

    print("We're looking at the following error: " + which_data)
    if which_data=='Hartley_Bay_grc100_SSH':
        obs_fp = 'processed_data/SSH_Hartley_Bay_gauge.csv'
        model_fp = 'processed_data/SSH_grc100_Hartley_Bay.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Hartley_Bay_grc100.csv')
    if which_data=='Lowe_Inlet_pytide_grc100_SSH':
        obs_fp = 'processed_data/SSH_Lowe_Inlet_pytide.csv'
        model_fp = 'processed_data/SSH_grc100_Lowe_Inlet.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Lowe_Inlet_pytide_grc100.csv')
    if which_data=='Lowe_Inlet_utide_grc100_SSH':
        obs_fp = 'processed_data/SSH_Lowe_Inlet_utide.csv'
        model_fp = 'processed_data/SSH_grc100_Lowe_Inlet.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Lowe_Inlet_utide_grc100.csv')
    if which_data=='Hartley_Bay_kit500_SSH':
        obs_fp = 'processed_data/SSH_Hartley_Bay_gauge.csv'
        model_fp = 'processed_data/SSH_kit500_Hartley_Bay.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Hartley_Bay_kit500.csv')
    if which_data=='Prince_Rupert_kit500_SSH':
        obs_fp = 'processed_data/SSH_Prince_Rupert_gauge.csv'
        model_fp = 'processed_data/SSH_kit500_Prince_Rupert.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Prince_Rupert_kit500.csv')
    if which_data=='Lowe_Inlet_pytide_kit500_SSH':
        obs_fp = 'processed_data/SSH_Lowe_Inlet_pytide.csv'
        model_fp = 'processed_data/SSH_kit500_Lowe_Inlet.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Lowe_Inlet_pytide_kit500.csv')
    if which_data=='Lowe_Inlet_utide_kit500_SSH':
        obs_fp = 'processed_data/SSH_Lowe_Inlet_utide.csv'
        model_fp = 'processed_data/SSH_kit500_Lowe_Inlet.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Lowe_Inlet_utide_kit500.csv')
    if which_data=='ADCP_kit500_SSH':
        obs_fp = 'processed_data/SSH_ADCP_lower.csv'
        model_fp = 'processed_data/SSH_kit500_ADCP.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_ADCP_lower_kit500.csv')
    if which_data=='ADCP_grc100_SSH':
        obs_fp = 'processed_data/SSH_ADCP_lower.csv'
        model_fp = 'processed_data/SSH_grc100_ADCP.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_ADCP_lower_grc100.csv')
    if which_data=='ADCP_full':
        obs_fp = 'processed_data/velocities_ADCP.nc'
        model_fp = 'processed_data/velocities_grc100_ADCP.nc'
        err_df = vel_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/velocities_err_grc100.csv')
    if which_data=='ADCP_8.4':
        obs_fp = 'processed_data/velocities_ADCP_8.4m.nc'
        model_fp = 'processed_data/velocities_grc100_ADCP_8.4m.nc'
        err_df = vel_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/velocities_err_grc100_8.4m.csv')
    if which_data=='ADCP_grc100_SSH_2023':
        obs_fp = 'processed_data/SSH_ADCP_lower_2023.csv'
        model_fp = 'processed_data/SSH_grc100_ADCP.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_ADCP_grc100_2023.csv')
    if which_data=='Hartley_Bay_grc100_SSH_2023':
        obs_fp = 'processed_data/SSH_Hartley_Bay_2023.csv'
        model_fp = 'processed_data/SSH_grc100_Hartley_Bay.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Hartley_Bay_grc100_2023.csv')
    if which_data=='Lowe_Inlet_grc100_SSH_2023':
        obs_fp = 'processed_data/SSH_Lowe_Inlet_2023.csv'
        model_fp = 'processed_data/SSH_grc100_Lowe_Inlet.nc'
        err_df = ssh_error(obs_fp,model_fp)
        err_df.to_csv('processed_data/SSH_err_Lowe_Inlet_grc100_2023.csv')
    else:
        'Bad choice of data!'
    print(which_data+' error data have been saved')

if __name__ == '__main__':

    #== SSH data ==#
    #save_tide_data('Hartley_Bay_tide_gauge')
    #save_tide_data('Prince_Rupert_tide_gauge')
    #save_tide_data('Lowe_Inlet_tide_gauge')
    #save_tide_data('Lowe_Inlet_tide_gauge_pytide')
    #save_tide_data('Lowe_Inlet_tide_gauge_utide')
    #save_tide_data('ADCP_upper_SSH')
    #save_tide_data('ADCP_lower_SSH')
    #save_tide_data('Prince_Rupert_kit500_SSH')
    #save_tide_data('Hartley_Bay_kit500_SSH')
    #save_tide_data('Lowe_Inlet_kit500_SSH')
    #save_tide_data('ADCP_kit500_SSH')
    #save_tide_data('Hartley_Bay_grc100_SSH')
    #save_tide_data('Lowe_Inlet_grc100_SSH')
    #save_tide_data('ADCP_grc100_SSH')
    #save_tide_data('ADCP_grc100_velocity')
    #save_tide_data('ADCP_velocity')
    #save_tide_data('Channel_Inlet_grc100_SSH')
    #save_tide_data('Channel_Outlet_grc100_SSH')
    #save_tide_data('Channel_Inlet_kit500_SSH')
    #save_tide_data('Channel_Outlet_kit500_SSH')
    #save_tide_data('Lowe_Inlet_extension_2023')
    #save_tide_data('Hartley_Bay_extension_2023')
    #save_tide_data('ADCP_extension_2023')

    #== Error data ==#
    #save_err_data('Hartley_Bay_grc100_SSH')
    #save_err_data('Lowe_Inlet_pytide_grc100_SSH')
    #save_err_data('Lowe_Inlet_utide_grc100_SSH')
    #save_err_data('Hartley_Bay_kit500_SSH')
    #save_err_data('Prince_Rupert_kit500_SSH')
    #save_err_data('Lowe_Inlet_pytide_kit500_SSH')
    #save_err_data('Lowe_Inlet_utide_kit500_SSH')
    #save_err_data('ADCP_kit500_SSH')
    #save_err_data('ADCP_grc100_SSH')
    #save_err_data('ADCP_full')
    #save_err_data('ADCP_8.4')
    #save_err_data('Lowe_Inlet_grc100_SSH_2023')
    #save_err_data('Hartley_Bay_grc100_SSH_2023')
    #save_err_data('ADCP_grc100_SSH_2023')

    #== Weather station ==#
    #wind, pressure = weather_station()
    #wind.to_csv('processed_data/weather_station_wind.csv')
    #pressure.to_csv('processed_data/weather_station_pressure.csv')

    #ds = xr.open_dataset('processed_data/velocities_grc100_ADCP_full_column.nc')
    #print(ds)