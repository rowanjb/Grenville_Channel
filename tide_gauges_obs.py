# initial looking at observations, tide gauges specifically

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

file_name = {
        'PrinceRupert': 'PrinceRupert_Jun012019-Nov202023.csv',
        'HartleyBay': 'HartleyBay_Jun012019-Nov202023.csv',
        'LoweInlet': 'LoweInlet_2014_forConstituents.csv'
    }

fig_title = {
        'PrinceRupert': 'Tide gauge in Prince Rupert',
        'HartleyBay': 'Tide gauge in Hartley Bay',
        'LoweInlet': 'Tide gauge in Lowe Inlet'
    }

c1, c2, c3 = plt.cm.viridis([0, 0.5, 0.8])

def tide_gauges(location):

    tidal_gauges_obs_dir_path = '/mnt/storage3/tahya/DFO/Observations/Tide_Gauges/'
    obs_path = tidal_gauges_obs_dir_path + file_name[location]

    df = pd.read_csv(obs_path, sep=",", header=None,engine='python')
    df = df.drop(index=range(0,8), columns=[2])
    df.columns = ['original_date','val']
    df['datetime_date'] = pd.to_datetime(df['original_date'],format = "%Y/%m/%d %H:%M")
    df['val'] = df['val'].astype(float)


    plt.figure()
    df.plot(x='datetime_date', y='val', title=fig_title[location], legend=None, color=c)

    plt.savefig('plots/' 'test.png',dpi=300, bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    tide_gauges('PrinceRupert')