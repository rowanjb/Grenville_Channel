#Creates list(s) of model output .nc files (currently only works for grc100, since kit500 isn't fully untarred)
#Note: Might need to be modified to identify and remove "bad" files, i.e., any that might give "all nan slice" errors
#Rowan Brown
#11 Feb 2024

#import xarray as xr
import os
import csv 

def filepaths(): 

	#directories of nemo output "sub"directories
	grc_storage3 = '/mnt/storage3/tahya/DFO/grc100_model_results/'
	grc_storage5 = '/mnt/storage5/tahya/DFO/grc100_model_results/'

	#get list of paths to the output "sub"directories
	get_subdirs = lambda output_dir : sorted([output_dir + subdir for subdir in os.listdir(output_dir) if subdir.endswith('000')])
	grc_subdirs = get_subdirs(grc_storage3) + get_subdirs(grc_storage5)
	
	#get list of paths to the files within all the output "sub"directories
	def get_subdir_files(subdir_paths):
		filepaths = []
		for subdir_path in subdir_paths:
			filepaths = filepaths + [subdir_path + '/' + f for f in os.listdir(subdir_path)]
		return sorted(filepaths)
	grc_filepaths = get_subdir_files(grc_subdirs)

	#file suffixes
	suffixes = ['1d_grid_T.nc','1h_grid_T.nc','1h_grid_V.nc','1ts_geo_U.nc','1d_grid_U.nc','1h_grid_U_2D.nc','1ts_geo_T_2D.nc','1ts_geo_V_2D.nc','1d_grid_V.nc','1h_grid_U.nc','1ts_geo_T.nc','1ts_geo_V.nc','1h_grid_T_2D.nc','1h_grid_V_2D.nc','1ts_geo_U_2D.nc','1ts_geo_W.nc']

	#saving the filepath lists
	for suffix in suffixes:
		filepaths_to_save = [filepath for filepath in grc_filepaths if filepath.endswith(suffix)]
		with open('filepaths/grc_filepaths_' + suffix[:-3] + '.csv', 'w', newline='') as output:
			write = csv.writer(output, quoting=csv.QUOTE_ALL)
			write.writerow(filepaths_to_save)

if __name__ == '__main__':
	filepaths()
