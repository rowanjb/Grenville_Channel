#Script for creating plot(s) of the two meshes (kit500 and grc100)
#Rowan Brown, 5 Jan 2024

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

def mesh_map():

	#== setting up the map ==#

	#extents of the maps
	westLon = -131.1
	eastLon = -127.7
	northLat = 51.9
	southLat = 54.4

	#defining the projection, note that standard parallels are the parallels of correct scale
	projection = ccrs.AlbersEqualArea(central_longitude=-129.4, central_latitude=53.15,standard_parallels=(southLat,northLat))

	#create figure (using the specified projection)
	fig, axs = plt.subplots(1, 2, subplot_kw={'projection': projection})

	#colour map
	cm = 'viridis'

	#meshes to plot
	meshes = ['grc100', 'kit500']

	for i,(mesh,ax) in enumerate(zip(meshes,axs)):

		#define map dimensions (using Plate Carree coordinate system)
		ax.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())

		#add coast lines 
		coast = feature.GSHHSFeature(scale="f") #high res coast line
		ax.add_feature(coast)

		#ticks
		gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5, dms=False) #dms is minuts vs frac
		gl.top_labels=False #suppress top labels
		gl.right_labels=False #suppress right labels
		if i==1: gl.left_labels=False #suppress left labels on right plot
		gl.rotate_labels=False
		gl.xlabel_style = {'size': 9}
		gl.ylabel_style = {'size': 9}

		#opening the mesh 
		meshPath = '/mnt/storage3/tahya/DFO/' + mesh + '_config/mesh_mask.nc'
		DS = xr.open_dataset(meshPath)
		tmask = DS.tmask.isel(t=0,z=0) #(taking slice of tmask at z=0)

		#plotting meshes
		p = ax.pcolormesh(DS.nav_lon, DS.nav_lat, tmask, transform=ccrs.PlateCarree(), vmin=0, vmax=1, cmap=cm)

		#title
		ax.set_title(mesh)

	#overall title
	fig.suptitle('Comparing mesh sizes around Grenville Channel',y=0.88)

	#save and close figure
	plt.savefig('maps/meshes' + '_map.png',dpi=300, bbox_inches="tight")
	plt.clf()

	print('done')

if __name__ == "__main__":
	mesh_map()
