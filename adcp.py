#For analysing Grenville Channel ADCP data
#Requires the packages/environment from https://github.com/IOS-OSD-DPG/pycurrents_ADCP_processing
#Rowan Brown
#March 2024

print('importing packages...')
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
#from pycurrents_ADCP_processing import plot_westcoast_nc_LX
from scipy import signal
import os
print('done importing')

def limit_data(ncdata: xr.Dataset, ew_data, ns_data, time_range=None, bin_range=None):
    """
    Limits data to be plotted to only "good" data, either automatically or with user-input
    time and bin ranges
    :param ncdata: xarray dataset-type object containing ADCP data from a netCDF file
    :param ew_data: east-west velocity data
    :param ns_data: north-south velocity data
    :param time_range: optional; a tuple of the form (a, b) where a is the index of the first
                       good ensemble and b is the index of the last good ensemble in the dataset
    :param bin_range: optional; a tuple of the form (a, b) where a is the index of the minimum
                      good bin and b is the index of the maximum good bin in the dataset
    :return: time_lim, cleaned time data; bin_depths_lim; cleaned bin depth data; ns_lim,
             cleaned north-south velocity data; and ew_lim; cleaned east-west velocity data
    """
    if ncdata.orientation == 'up':
        bin_depths = ncdata.instrument_depth.data - ncdata.distance.data
    else:
        bin_depths = ncdata.instrument_depth.data + ncdata.distance.data
    # print(bin_depths)

    # REVISION Jan 2024: bad leading and trailing ensembles are deleted from dataset, so don't need this step
    # data.time should be limited to the data.time with no NA values; bins must be limited
    if time_range is None:
        # if 'L1' in ncdata.filename.data.tolist() or 'L2' in ncdata.filename.data.tolist():
        #     leading_ens_cut, trailing_ens_cut = parse_processing_history(
        #         ncdata.attrs['processing_history']
        #     )
        #     time_first_last = (leading_ens_cut, len(ew_data[0]) - trailing_ens_cut)
        time_first_last = (0, len(ew_data[0]))
    else:
        time_first_last = time_range

    # Remove bins where surface backscatter occurs
    time_lim = ncdata.time.data[time_first_last[0]:time_first_last[1]]

    if bin_range is None:
        bin_first_last = (np.where(bin_depths >= 0)[0][0], np.where(bin_depths >= 0)[0][-1])
        bin_depths_lim = bin_depths[bin_depths >= 0]

        ew_lim = ew_data[bin_depths >= 0, time_first_last[0]:time_first_last[1]]  # Limit velocity data
        ns_lim = ns_data[bin_depths >= 0, time_first_last[0]:time_first_last[1]]
    else:
        bin_first_last = (bin_range[0], bin_range[1])
        bin_depths_lim = bin_depths[bin_range[0]: bin_range[1]]

        ew_lim = ew_data[bin_range[0]: bin_range[1], time_first_last[0]:time_first_last[1]]
        ns_lim = ns_data[bin_range[0]: bin_range[1], time_first_last[0]:time_first_last[1]]

    return time_lim, bin_depths_lim, ns_lim, ew_lim, time_first_last, bin_first_last

def get_adcp_obs():
	'''For ADCP analyses'''

	adcp_dir = '/mnt/storage3/tahya/DFO/Observations/GRC1_Mooring_Data/ADCP/'
	adcp_files = [adcp_dir + file for file in os.listdir(adcp_dir) if file.endswith('.nc')] #there are two files...
	
	ncfile = adcp_files[1]
	dest_dir = 'tidal_figures'

	ncdata = xr.open_dataset(ncfile)

	time_lim, bin_depths_lim, ns_lim, ew_lim, time_range_idx, bin_range_idx = limit_data(
        ncdata, ncdata.LCEWAP01.data, ncdata.LCNSAP01.data)#, time_range, bin_range)

	resampled=None
	if resampled is None:
		resampled_str = ''
		resampled_4fname = ''

	colourmap_lim=None
	if colourmap_lim is None:
		vminvmax = [5,5] #get_vminvmax(ns_lim, ew_lim)

	instrument_depth = float(ncdata.instrument_depth)

	fig = plt.figure(figsize=(13.75, 10))
	ax = fig.add_subplot(2, 1, 1)

	f1 = ax.pcolormesh(time_lim, bin_depths_lim, ns_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0], vmax=vminvmax[1], shading='auto')
	cbar = fig.colorbar(f1, shrink=0.8)
	cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)
	ax.set_ylabel('Depth [m]', fontsize=14)

	filter_type = 'raw'
	if 'h' in filter_type:  # xxh-average; e.g. '30h', '35h'
		filter_type_title = '{} average'.format(filter_type)
	elif filter_type == 'Godin':
		filter_type_title = 'Godin filtered'
	elif filter_type == 'raw':
		filter_type_title = filter_type
	else:
		ValueError('Not a recognized data type; choose one of \'raw\', \'30h\' or \'Godin\'')

	magnetic = ''
	ax.set_title(
		'ADCP ({}North, {}) {}-{} {}m{}'.format(
			magnetic, filter_type_title, ncdata.attrs['station'], ncdata.attrs['deployment_number'],
			instrument_depth, resampled_str
		), fontsize=14
	)

	ax.invert_yaxis()

	ax2 = fig.add_subplot(2, 1, 2)

	f2 = ax2.pcolormesh(time_lim, bin_depths_lim, ew_lim[:, :], cmap='RdBu_r', vmin=vminvmax[0],
						vmax=vminvmax[1], shading='auto')
	cbar = fig.colorbar(f2, shrink=0.8)
	cbar.set_label('Velocity [m s$^{-1}$]', fontsize=14)

	ax2.set_ylabel('Depth [m]', fontsize=14)

	ax2.set_title(
		'ADCP ({}East, {}) {}-{} {}m{}'.format(
			magnetic, filter_type_title, ncdata.attrs['station'], ncdata.attrs['deployment_number'],
			instrument_depth, resampled_str
		),
		fontsize=14
	)

	ax2.invert_yaxis()

	# Create L1_Python_plots or L2_Python_plots subfolder if not made already
	plot_dir = 'tidal_figures' #get_plot_dir(nc.filename, dest_dir)

	if not os.path.exists(plot_dir):
		os.makedirs(plot_dir)

	vel_type = 'NE'

	# Have to round instrument depth twice due to behaviour of the float
	plot_name = 'ADCP_test.png'# plot_dir + '{}-{}_{}_{}m_{}_{}{}.png'.format(
	#	ncdata.attrs['station'], ncdata.attrs['deployment_number'], ncdata.instrument_serial_number.data,
	#	round_to_int(instrument_depth), vel_type, filter_type, resampled_4fname
	#)
	fig.savefig(plot_name)
	plt.close()

	return os.path.abspath(plot_name)



if __name__ == "__main__":

	#== ADCPs! ==#
	get_adcp_obs()
