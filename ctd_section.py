"""
ctd_section.py
author: Tahya Weiss-Gibbons, weissgib@ualberta.ca

"""

import glob
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

#gets the list of grid points for a straight line section between two points
#uses the bresenham line algorithm
def section_calculation(x1, x2, y1, y2):
    ii = []
    jj = []

    dx = x2-x1
    dy = y2-y1
    yi = 1

    if dy < 0:
        yi = -1
        dy = -dy

    D = (2*dy)-dx
    y = y1

    for x in range(x1,x2):
        ii.append(x)
        jj.append(y)
        if D > 0:
            y = y+yi
            D = D+(2*(dy-dx))
            ii.append(x)
            jj.append(y)
        else:
            D = D+2*dy

    return ii,jj


#find the closest model grid point to a lat lon point
def closest_grid(nav_lon, nav_lat, lon, lat):
    abslat = np.abs(nav_lat-lat)
    abslon = np.abs(nav_lon-lon)

    c = (abslon**2 + abslat**2)**0.5
    return np.where(c == np.min(c))


def model_section():

    #get the model mesh information
    mesh_path = '/mnt/storage3/tahya/DFO/grc100_config/mesh_mask.nc'

    mesh = xr.open_dataset(mesh_path)

    nav_lon = mesh['nav_lon'].values
    nav_lat = mesh['nav_lat'].values

    mesh.close()

    #going along the ctd transect taken Aug 2020
    #the channel is so narrow, need to do this in sections
    #to make the line actucally follow the channell

    lon1 = -130.0845
    lat1 = 53.889

    lon2 = -129.91617
    lat2 = 53.7735

    lon3 = -129.81883
    lat3 = 53.720833

    lon4 = -129.75684
    lat4 = 53.657665

    lon5 = -129.7075
    lat5 = 53.61017

    lon6 = -129.6475
    lat6 = 53.564335

    lon7 = -129.62233
    lat7 = 53.549667

    lon8 = -129.56984
    lat8 = 53.518665

    lon9 = -129.33783
    lat9 = 53.375668

    y1, x1 = closest_grid(nav_lon, nav_lat, lon1, lat1)
    y2, x2 = closest_grid(nav_lon, nav_lat, lon2, lat2)
    y3, x3 = closest_grid(nav_lon, nav_lat, lon3, lat3)
    y4, x4 = closest_grid(nav_lon, nav_lat, lon4, lat4)
    y5, x5 = closest_grid(nav_lon, nav_lat, lon5, lat5)
    y6, x6 = closest_grid(nav_lon, nav_lat, lon6, lat6)
    y7, x7 = closest_grid(nav_lon, nav_lat, lon7, lat7)
    y8, x8 = closest_grid(nav_lon, nav_lat, lon8, lat8)
    y9, x9 = closest_grid(nav_lon, nav_lat, lon9, lat9)

    ii_1, jj_1 = section_calculation(x1[0], x2[0], y1[0], y2[0])
    ii_2, jj_2 = section_calculation(x2[0], x3[0], y2[0], y3[0])
    ii_3, jj_3 = section_calculation(x3[0], x4[0], y3[0], y4[0])
    ii_4, jj_4 = section_calculation(x4[0], x5[0], y4[0], y5[0])
    ii_5, jj_5 = section_calculation(x5[0], x6[0], y5[0], y6[0])
    ii_6, jj_6 = section_calculation(x6[0], x7[0], y6[0], y7[0])
    ii_7, jj_7 = section_calculation(x7[0], x8[0], y7[0], y8[0])
    ii_8, jj_8 = section_calculation(x8[0], x9[0], y8[0], y9[0])

    ii = ii_1+ii_2+ii_3+ii_4+ii_5+ii_6+ii_7+ii_8
    jj = jj_1+jj_2+jj_3+jj_4+jj_5+jj_6+jj_7+jj_8

    """
    #quick check of what the line section looks like...
    ln = []
    lt = []
    for k in range(len(ii)):
        ln.append(nav_lon[jj[k], ii[k]])
        lt.append(nav_lat[jj[k], ii[k]])

    
    projection = ccrs.AlbersEqualArea(central_longitude=-129.4, central_latitude=53.15,standard_parallels=(54.4,51.9))
    fig = plt.figure(figsize=(10,9))
    ax = plt.subplot(1, 1, 1, projection=projection)

    ax.set_extent([-131.1, -127.7, 54.4, 51.9], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='10m')
    ax.plot(ln, lt, linewidth=3.0, transform=ccrs.PlateCarree())
    plt.show()
    exit()
    """

    #now get the list of model files to read in
    #have saved this information previously in filepaths/

    grc_filepaths_txt = 'filepaths/'+'grc100_filepaths_1d_grid_T.csv'
    grc_filepaths = list(pd.read_csv(grc_filepaths_txt))

    #go through each file in the list
    #pull the section, and make a snapshot of the salinity and temp contours

    for gr in grc_filepaths:

        grc = xr.open_dataset(gr)

        temp_sec = []
        sal_sec = []

        depth = grc['deptht'].values

        #get the line section
        for n in range(0,(len(ii)-1)):
            i = ii[n]
            j = jj[n]

            temp = grc['thetao'].sel(x=i, y=j).values
            sal = grc['so'].sel(x=i, y=j).values

            temp_sec.append(temp)
            sal_sec.append(sal)

        temp_sec = np.stack(temp_sec, axis=0)
        sal_sec = np.stack(sal_sec, axis=0)
        temp_sec = temp_sec.transpose(2,0,1)
        sal_sec = sal_sec.transpose(2,0,1)

        times = grc['time_counter'].values
        #now plot for each time step
        for k in range(len(times)):
            t = times[k]
            print(t)

            #temp
            plt.contourf(ii[:-1], depth, temp_sec[:,:,k], vmin=0, vmax=18)

            ax = plt.gca()
            ax.set_ylim([0, 350])
            ax.invert_yaxis()
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_xticks([])
 
            plt.colorbar()
            plt.title(t)
            plt.savefig('figs/sections/temp_'+str(t)+'.png')
            plt.clf()

            #salinity
            plt.contourf(ii[:-1], depth, sal_sec[:,:,k])

            ax = plt.gca()
            ax.set_ylim([0, 350])
            ax.invert_yaxis()
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_xticks([])
            
            plt.colorbar()
            plt.title(t)
            plt.savefig('figs/sections/sal_'+str(t)+'.png')
            plt.clf()

        grc.close()


if __name__ == "__main__":
    model_section()
