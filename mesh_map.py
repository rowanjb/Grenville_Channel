#simple throwaway script making maps of data 

import xarray as xr
import LSmap 
import numpy as np 
import math
import os
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors

def mesh_map(mesh):

	meshPath = '/mnt/storage3/tahya/DFO/' + mesh + '_config/mesh_mask.nc'
	DS = xr.open_dataset(meshPath)
	tmask = DS.tmask.isel(t=0,z=0) #(taking slice of tmask at z=0)

	westLon = -131.1
	eastLon = -127.7
	northLat = 51.9
	southLat = 54.4

	##shapefile of land with 1:50,000,000 scale
	#land_50m = feature.NaturalEarthFeature('physical', 'land', '10m',edgecolor='black', facecolor='gray')

	#defining the projection, note that standard parallels are the parallels of correct scale
	projection = ccrs.AlbersEqualArea(central_longitude=-129.4, central_latitude=53.15,standard_parallels=(southLat,northLat))

	#create figure (using the specified projection)
	ax = plt.subplot(1, 1, 1, projection=projection)

	#define map dimensions (using Plate Carree coordinate system)
	ax.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())

	#add coast lines 
	coast = feature.GSHHSFeature(scale="f") #high res coast line
	ax.add_feature(coast)

	#ticks
	gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, dms=False) #dms is minuts vs frac
	gl.top_labels=False #suppress top labels
	gl.right_labels=False #suppress right labels
	gl.rotate_labels=False
	gl.xlabel_style = {'size': 9}
	gl.ylabel_style = {'size': 9}

	#colour map
	cm = 'viridis'

	#plotting data
	p1 = ax.pcolormesh(DS.nav_lon, DS.nav_lat, tmask, transform=ccrs.PlateCarree(), vmin=0, vmax=1, cmap=cm)
	#ax_cb = plt.axes([0.83, 0.25, 0.022, 0.5])
	#cb = plt.colorbar(p1,cax=ax_cb, orientation='vertical')#, format='%.0e')
	#cb.formatter.set_powerlimits((0, 0))
	#cb.ax.set_ylabel(CBlabel)

	#title
	ax.set_title(mesh)# + ' ' + date)#,fontdict={'fontsize': 12})

	#save and close figure
	plt.savefig(mesh + '_map.png',dpi=300, bbox_inches="tight")
	plt.clf()

if __name__ == "__main__":
	for mesh in ['kit500','grc100']:
		mesh_map(mesh)
	quit()
