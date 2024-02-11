# initial looking at observations, tide gauges specifically

import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt
import pytide

file_name = {
	'PrinceRupert': 'PrinceRupert_Jun012019-Nov202023.csv',
	'HartleyBay': 'HartleyBay_Jun012019-Nov202023.csv',
	'LoweInlet': 'LoweInlet_2014_forConstituents.csv'}

c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

def tide_gauges():

	#path to the tide gauge CSVs
	tidal_gauges_obs_dir_path = '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/'

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

	#extending the 2014 Lowe Inlet data forward to match the other data
	wt = pytide.WaveTable()
	time2014 = df['LoweInlet_date'].dropna().to_numpy()
	time2020 = df['HartleyBay_date'].dropna().to_numpy()
	h = df['LoweInlet_val'].dropna().to_numpy()
	f, vu = wt.compute_nodal_modulations(time2014)
	w = wt.harmonic_analysis(h, f, vu)
	hp = wt.tide_from_tide_series(time2020, w) #can change the time if you want!!

	#IGNORE THIS SECTION
	#wtM2 = pytide.WaveTable(["M2"])
	#fM2, vuM2 = wtM2.compute_nodal_modulations(time)	
	#wM2 = wtM2.harmonic_analysis(h, fM2, vuM2)
	#hpM2 = wtM2.tide_from_tide_series(df['LoweInlet_date'].dropna().to_numpy(), wM2)

	#plotting
	fig,ax1 = plt.subplots()
	
	#tide gauge data
	df.plot(x='PrinceRupert_date', y='PrinceRupert_val', ax=ax1, color=c1, label='Prince Rupert observed')
	df.plot(x='HartleyBay_date', y='HartleyBay_val', ax=ax1, color=c2, label='Hartley Bay observed')
	##df.plot(x='LoweInlet_date', y='LoweInlet_val', ax=ax1, color=c1, label='Lowe Inlet observed')

	#modelled tides
	ax1.plot(time2020,hp,color=c3,label='Lowe Inlet harmonic model') #can change the time, so long as it is the same as when you made hp

	ax1.legend()
	ax1.set_title('Grenville Channel tides, mid-2020')
	ax1.set_ylabel('SSH')
	ax1.set_xlabel('Date')
	#ax1.set_xlim([date(2020, 1, 1), date(2020, 1, 7)])
	ax1.set_xlim([date(2020, 6, 1), date(2020, 6, 7)])

	fig.savefig('plots/tide_gauges_2020.png',dpi=300, bbox_inches="tight")
	fig.clf()

	print('done')

if __name__ == "__main__":
	tide_gauges()
