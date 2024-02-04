#Script for creating plot(s) of the two meshes (kit500 and grc100)
#Rowan Brown, 5 Jan 2024

import xarray as xr
import numpy as np 
import math
import os
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors

#useful colours for plotting
c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

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

	#meshes to plot below
	meshes = ['grc100', 'kit500']

	#loops to create subplots (i.e., two maps in the same figure)
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

		transform = ccrs.PlateCarree()._as_mpl_transform(ax)
		
		ax.plot(-129.2535, 53.4329, 'o', color='black', transform=ccrs.PlateCarree())
		ax.plot(-129.2535, 53.4329, '.', color='white', transform=ccrs.PlateCarree())
		
		ax.plot(-129.5799, 53.5553, 'o', color='black', transform=ccrs.PlateCarree())
		ax.plot(-129.5799, 53.5553, '.', color='white', transform=ccrs.PlateCarree())

		ax.plot(-130.3208, 54.3150, 'o', color='black', transform=ccrs.PlateCarree())
		ax.plot(-130.3208, 54.3150, '.', color='white', transform=ccrs.PlateCarree())

		ax.annotate('Hartley\nBay', xy=(-129.2535, 53.4329),xytext=(-128.75, 53),arrowprops=dict(facecolor='white', shrink=0.05, width=2.5, headwidth=5, headlength=6), xycoords=transform, ha='left', va='center',bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),**{'backgroundcolor':'white','fontsize':'small'})#transform=ccrs.PlateCarree())#, zorder=12)
		ax.annotate('Lowe Inlet', xy=(-129.5799, 53.5553),xytext=(-128.75, 53.75),arrowprops=dict(facecolor='white', shrink=0.05, width=2.5, headwidth=5, headlength=6), xycoords=transform, ha='left', va='center',bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),**{'backgroundcolor':'white','fontsize':'small'})#transform=ccrs.PlateCarree())#, zorder=12)
		ax.annotate('Prince Rupert', xy=(-130.3208, 54.3150),xytext=(-129.3, 54.25),arrowprops=dict(facecolor='white', shrink=0.05, width=2.5, headwidth=5, headlength=6), xycoords=transform, ha='left', va='center',bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),**{'backgroundcolor':'white','fontsize':'small'})#transform=ccrs.PlateCarree())#, zorder=12)

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
