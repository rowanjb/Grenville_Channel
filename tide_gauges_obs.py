# initial looking at observations, tide gauges specifically

import pandas as pd
from datetime import datetime, date
import matplotlib.pyplot as plt

file_name = {
        'PrinceRupert': 'PrinceRupert_Jun012019-Nov202023.csv',
        'HartleyBay': 'HartleyBay_Jun012019-Nov202023.csv',
        'LoweInlet': 'LoweInlet_2014_forConstituents.csv'
    }

c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

def tide_gauges():

    tidal_gauges_obs_dir_path = '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/'

    df_list = []
    for location in file_name.keys():
        obs_path = tidal_gauges_obs_dir_path + file_name[location]
        df = pd.read_csv(obs_path, sep=",", header=None,engine='python')
        df = df.drop(index=range(0,8), columns=[2])
        df.columns = [location+'_date',location+'_val']
        df[location+'_date'] = pd.to_datetime(df[location+'_date'],format = "%Y/%m/%d %H:%M")
        df[location+'_val'] = df[location+'_val'].astype(float)
        df_list.append(df)

    df = pd.concat(df_list,axis=1)
    fig,ax1 = plt.subplots()
    df.plot(x='PrinceRupert_date', y='PrinceRupert_val', ax=ax1, color=c1)
    df.plot(x='HartleyBay_date', y='HartleyBay_val', ax=ax1, color=c2)
    df.plot(x='LoweInlet_date', y='LoweInlet_val', ax=ax1, color=c3)

    ax1.set_title('Tide Gauges')
    ax1.set_ylabel('SSH')
    ax1.set_xlabel('Date')
    ax1.set_xlim([date(2020, 1, 1), date(2020, 1, 7)])

    fig.savefig('plots/tide_gauges.png',dpi=300, bbox_inches="tight")
    fig.clf()

    print('done')

if __name__ == "__main__":
    tide_gauges()