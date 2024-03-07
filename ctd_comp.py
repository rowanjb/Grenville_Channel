import os
import glob
import difflib
import numpy as np
import xarray as xr
import datetime as dt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

def closest_grid(nav_lon, nav_lat, lon, lat):
    abslat = np.abs(nav_lat-lat)
    abslon = np.abs(nav_lon-lon)

    c = (abslon**2 + abslat**2)**0.5
    return np.where(c == np.min(c))

grc_paths = ['/mnt/storage3/tahya/DFO/grc100_model_results/', '/mnt/storage5/tahya/DFO/grc100_model_results/']
kit_paths = ['/mnt/storage5/tahya/DFO/kit500_model_results/', '/mnt/storage6/tahya/DFO/kit500_model_results/']

#first read in the ctd data
ctd_path = '/mnt/storage3/tahya/DFO/Observations/CTD_Casts/'

ctd_files = glob.glob(ctd_path+'20*.nc')

for c in ctd_files:

    df = xr.open_dataset(c)

    lat = df['latitude'].values
    lon = df['longitude'].values
    print(lat)
    print(lon)

    y = df['time'].dt.year.values
    m = df['time'].dt.month.values
    d = df['time'].dt.day.values

    exact_loc = str(y)+str(m).zfill(2)+str(d).zfill(2)+'00_000'
    print(exact_loc)

    t = dt.date(y,m,d)

    #lets try to find model output from around the same date
    gp_files = []
    kp_files = []
    for gp in grc_paths:
        gp_files = gp_files+[name for name in os.listdir(gp)
            if os.path.isdir(os.path.join(gp, name))]    
    
    g_dates = [dt.datetime.strptime(date[0:8], "%Y%m%d").date() for date in gp_files] 
    gmatch = min(g_dates, key=lambda x: abs(x - t))
    print(gmatch)
    i = g_dates.index(gmatch)

    if gmatch.year == t.year:

        try:
            gf = xr.open_dataset(grc_paths[0]+gp_files[i]+'/NEMO_RPN_1d_grid_T.nc')
        except:
            gf = xr.open_dataset(grc_paths[1]+gp_files[i]+'/NEMO_RPN_1d_grid_T.nc')

        print(gf)
         
        #get the right profile
        nav_lat = xr.where(gf.nav_lat != 0, gf.nav_lat, np.nan)
        nav_lon = xr.where(gf.nav_lon != 0, gf.nav_lon, np.nan)
        y, x = closest_grid(nav_lon, nav_lat, lon, lat)
        grc_profile = gf['so'].sel(x=x,y=y)
        
        #make a plot of the location where the ctd cast was taken
        gf['thetao'][0,0].plot()
        plt.scatter(x, y, color='b')
        plt.savefig('figs/grc_ctd_loc_'+t.strftime('%d%m%Y')+'_'+str(lon)+'_'+str(lat)+'.png')
        plt.clf()

        #now plot the profile
        grc_profile[0].plot(y='deptht', label='grc100')

    else: continue

    for kp in kit_paths:
        kp_files = kp_files+[name for name in os.listdir(kp)
            if os.path.isdir(os.path.join(kp, name))]
    
    k_dates = [dt.datetime.strptime(date[0:8], "%Y%m%d").date() for date in kp_files]
    kmatch = min(k_dates, key=lambda x: abs(x - t))
    print(kmatch)
    i = k_dates.index(kmatch)

    try:
        kf = xr.open_dataset(kit_paths[0]+kp_files[i]+'/NEMO_RPN_1d_grid_T.nc')
    except:
        kf = xr.open_dataset(kit_paths[1]+kp_files[i]+'/NEMO_RPN_1d_grid_T.nc')

    print(kf)

    #get the right profile
    nav_lat = xr.where(kf.nav_lat != 0, kf.nav_lat, np.nan)
    nav_lon = xr.where(kf.nav_lon != 0, kf.nav_lon, np.nan)
    y, x = closest_grid(nav_lon, nav_lat, lon, lat)
    kit_profile = kf['so'].sel(x=x,y=y)
    kit_profile[0].plot(y='deptht', label='kit500')
    
    #and now add the ctd cast profile
    df['sea_water_practical_salinity'].plot(y='z', label='ctd')
    plt.gca().invert_yaxis()
    plt.title("Salinity profile at lon: "+str(lon)+", lat: "+str(lat)+" on "+t.strftime('%d/%m/%Y'))
    plt.legend()
    #plt.savefig('figs/sal_profile_'+t.strftime('%d%m%Y')+'_'+str(lon)+'_'+str(lat)+'.png')
    #plt.clf()
    plt.show()
    exit()

