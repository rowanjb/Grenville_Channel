#Creates list(s) of model output .nc files (currently only works for grc100, since kit500 isn't fully untarred)
#Note: Can be modified to identify and remove "bad" files, i.e., any that might give "all nan slice" errors
#Rowan Brown
#11 Feb 2024

import xarray as xr
import os

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
		with open('filepaths/grc_filepaths_' + suffix[:-3] + '.txt', 'w') as output:
			for filepath in filepaths_to_save:
				output.write(str(filepath) + '\n')
	quit()
	#output_dir) if f.endswith('000')])
	#grc_subdirs = get_subdirs(grc_storage3) + get_subdirs(grc_storage5)

	#get list of all 
	

	#creating lists of filepaths to the individual output files
	"""
	grc_1d_grid_T = 
	grc_1d_grid_U
	grc_1d_grid_V
	grc_1h_grid_T
    grc_1h_grid_U
    grc_1h_grid_V
	grc_1h_grid_T_2D
    grc_1h_grid_U_2D
    grc_1h_grid_V_2D
	grc_geo_T
	grc_geo_U
	grc_geo_V
	grc_geo_T_2D
    grc_geo_U_2D
    grc_geo_V_2D
	grc_geo_W
	"""
	#testing if files are read-able
	bad_files = [] #initializing list of bad filepaths
	for filepath in grc_subdirs_fps:
		try:
			DS = xr.open_dataset(filepath)
		except:
			bad_files.append(filepath[:-8]) #saving any bad filepaths
			print('gridT: ' + filepath)

	###testing if gridU files are read-able
	##for filepath in filepaths_gridU:
	##    try:
	##        DS = xr.open_dataset(filepath)
	##    except:
	##        bad_files.append(filepath[:-8]) #saving any bad filepaths
	##        print('gridU: ' + filepath)
	##
	###testing if gridV files are read-able
	##for filepath in filepaths_gridV:
	##    try:
	##        DS = xr.open_dataset(filepath)
	##    except:
	##        bad_files.append(filepath[:-8]) #saving any bad filepaths
	##        print('gridV: ' + filepath)
	##
	###testing if icemod files are read-able
	##for filepath in filepaths_icemod:
	##    try:
	##        DS = xr.open_dataset(filepath)
	##    except:
	##        bad_files.append(filepath[:-9]) #saving any bad filepaths
	##        print('icemod: ' + filepath) 

	#removing duplicates from the list
	bad_files = list( dict.fromkeys(bad_files) )

	#removing bad filepaths
	for bad_file in bad_files:
		print(bad_file + ' is a bad file')
	filepaths_gridT.remove(bad_file + 'gridT.nc')
	##filepaths_gridU.remove(bad_file + 'gridU.nc')
	##filepaths_gridV.remove(bad_file + 'gridV.nc')
	##filepaths_gridB.remove(bad_file + 'gridB.nc')
	##filepaths_gridW.remove(bad_file + 'gridW.nc')
	##filepaths_icebergs.remove(bad_file + 'icebergs.nc')
	##filepaths_icemod.remove(bad_file + 'icemod.nc')

	#creating directory if doesn't already exist
	dir = run + '_filepaths/'
	if not os.path.exists(dir):
		os.makedirs(dir)

	#saving the filepaths as txt files
	with open(dir + run + '_gridT_filepaths_Jan2024.txt', 'w') as output:
		for i in filepaths_gridT:
			output.write(str(i) + '\n')
	##with open(dir + run + '_gridU_filepaths.txt', 'w') as output:
	##    for i in filepaths_gridU:
	##        output.write(str(i) + '\n')
	##with open(dir + run + '_gridV_filepaths.txt', 'w') as output:
	##    for i in filepaths_gridV:
	##        output.write(str(i) + '\n')
	##with open(dir + run + '_gridB_filepaths.txt', 'w') as output:
	##    for i in filepaths_gridB:
	##        output.write(str(i) + '\n')
	##with open(dir + run + '_gridW_filepaths.txt', 'w') as output:
	##    for i in filepaths_gridW:
	##        output.write(str(i) + '\n')
	##with open(dir + run + '_icebergs_filepaths.txt', 'w') as output:
	##    for i in filepaths_icebergs:
	##        output.write(str(i) + '\n')
	##with open(dir + run + '_icemod_filepaths.txt', 'w') as output:
	##    for i in filepaths_icemod:
	##        output.write(str(i) + '\n')

if __name__ == '__main__':
	filepaths()
