# Auth: cj
# del percentile calculation part -- 0915
# add domain param

import netCDF4 as nc4
import wrf
import pandas as pd
import numpy as np
import os
import sys
import arrow
from multiprocessing import Pool

city = "Shantou"
# chongqing Obs data start from 2015
start_year = '2019' # min is 2011, another option for ChongQing is 2015
domain = 'd02'

# set Chongqing grid in wrf lonlat
we_range = np.arange(0, 222)
sn_range = np.arange(0, 162)

def get_now():
    return arrow.now().format('YYYYMMDD HH:mm:ss')


def do_load_data(cluster_type, idx, fldir):
    print('Type', cluster_type, 'loading data: [', idx+1, '/', len_type, ']')
    # print('loading file({})'.format(fldir))
    ncfile = nc4.Dataset(fldir)
    temp_kelvins = wrf.getvar(ncfile, 'T2')
    if idx == 0:
        print('temp_kelvins shape:', temp_kelvins.shape)
    temp_kelvins = temp_kelvins.isel(south_north=sn_range, west_east=we_range).values.flatten()
    humidity = wrf.getvar(ncfile, 'rh2')
    if idx == 0:
        print('humidity shape:', humidity.shape)
    humidity = humidity.isel(south_north=sn_range, west_east=we_range).values.flatten()
    slp = wrf.getvar(ncfile, 'slp')
    if idx == 0:
        print('slp shape:', slp.shape)
    slp = slp.isel(south_north=sn_range, west_east=we_range).values.flatten()
    
    wind = wrf.getvar(ncfile, 'wspd_wdir10')
    windspd = wind[0]
    if idx == 0:
        print('windspd shape:', windspd.shape)
    windspd = windspd.isel(south_north=sn_range, west_east=we_range).values.flatten()

    winddir = wind[1]
    if idx == 0:
        print('winddir shape', winddir.shape)
    winddir = winddir.isel(south_north=sn_range, west_east=we_range).values.flatten()

    return [temp_kelvins, humidity, slp, windspd, winddir]

def adj_time(time_str):
    return time_str[:-8]+'04'+time_str[-6:]

print(get_now(), 'start to calculate history weather plot...')
target_fldrdir = sys.path[0] + '/input/training'
flnm_list = os.listdir(target_fldrdir)
# fldir_list = [target_fldrdir +'\\'+ flnm for flnm in flnm_list]

df_cluster = pd.read_csv(sys.path[0]+'/db/train_cluster.csv', names=['x', 'type']) 
# type is y;
# pass 2/3 col names, pd will take 1st col as index
# pass 1/3 col names, pd will take the 1st and 2nd col as multi-index
df_cluster.drop(columns=['x'], inplace=True)
df_cluster = df_cluster.loc[start_year:]
df_cluster['type'] = df_cluster['type']+1
cluster_types = df_cluster.type.unique()
cluster_types.sort()

idx = df_cluster.index
idx = pd.Series(idx).apply(lambda x: adj_time(x))
df_cluster.index = idx
print(df_cluster)
print(df_cluster.index)

# df_pctl = pd.DataFrame()
# df_pctg_types = pd.DataFrame(columns=cluster_types)

df_freq_alltp = pd.DataFrame()
df_tmp_mean_alltp = pd.DataFrame()
df_humi_mean_alltp = pd.DataFrame()
df_slp_mean_alltp = pd.DataFrame()
df_wspd_mean_alltp = pd.DataFrame()
df_wdir_alltp = pd.DataFrame()

for cluster_type in cluster_types:
    print(get_now(), 'calculating type {}...'.format(cluster_type))
    
    flnm_list = df_cluster[df_cluster.type == cluster_type].index
    freq_type = len(flnm_list)/len(df_cluster)
    print('type {}  frequency:'.format(cluster_type), freq_type)
    se_freq_type = pd.Series(freq_type, ['Type {}'.format(cluster_type)])
    df_freq_alltp = pd.concat([df_freq_alltp, se_freq_type])
    # df_pctg_types[cluster_type] = round(pctg_type, 4)
    # print(flnm_list)
    
    fldir_list = [target_fldrdir + '/' + f'wrfout_{domain}_' + flnm for flnm in flnm_list]
    print(fldir_list)

    # print(fldir_list)
    len_type = len(fldir_list)

    temp_kelvins = []
    humidity = []
    slp = []
    windspd = []
    winddir = []
    
    p = Pool(28)
    for idx, fldir in enumerate(fldir_list):
        print('loading file({}).'.format(fldir))
        sync = p.apply_async(do_load_data, args=(cluster_type, idx, fldir,))
        load_result = sync.get()
        temp_kelvins_temp = load_result[0]
        humidity_temp = load_result[1]
        slp_temp = load_result[2]
        windspd_temp = load_result[3]
        winddir_temp = load_result[4]
        temp_kelvins = np.append(temp_kelvins, temp_kelvins_temp)
        humidity = np.append(humidity, humidity_temp)
        slp = np.append(slp, slp_temp)
        windspd = np.append(windspd, windspd_temp)
        winddir = np.append(winddir, winddir_temp)
    p.close()
    p.join()

    temp_celsius_type = temp_kelvins - 272.15 # k unit to c unit
    temp_mean = np.mean(temp_celsius_type)
    humi_mean = np.mean(humidity)
    slp_mean = np.mean(slp)
    windspd_mean = np.mean(windspd)

    df_temp_mean = pd.DataFrame([temp_mean], index=['Type {}'.format(cluster_type)], columns=['Value'])
    df_tmp_mean_alltp = pd.concat([df_tmp_mean_alltp, df_temp_mean])
    df_humi_mean = pd.DataFrame([humi_mean], index=['Type {}'.format(cluster_type)], columns=['Value'])
    df_humi_mean_alltp = pd.concat([df_humi_mean_alltp, df_humi_mean])
    df_slp_mean = pd.DataFrame([slp_mean], index=['Type {}'.format(cluster_type)], columns=['Value'])
    df_slp_mean_alltp = pd.concat([df_slp_mean_alltp, df_slp_mean])
    df_wspd_mean = pd.DataFrame([windspd_mean], index=['Type {}'.format(cluster_type)], columns=['Value'])
    df_wspd_mean_alltp = pd.concat([df_wspd_mean_alltp, df_wspd_mean])

    # process wind dir
    wdir_edge = np.arange(-22.5, 338, 45)
    wdir_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    print(winddir)
    wdir_type_adj = pd.Series(winddir).apply(lambda x: x-360 if x>337.5 else x)
    print(wdir_type_adj)
    wdir_cut = pd.cut(wdir_type_adj, wdir_edge, labels=wdir_labels)
    wdir_counts = wdir_cut.value_counts()
    print(wdir_counts)
    wdir_counts.index.name = 'W_dir_tp{}'.format(cluster_type)
    df_wdir = wdir_counts.to_frame(name='Count_tp{}'.format(cluster_type))
    df_wdir.reset_index(inplace=True)
    if df_wdir_alltp.empty:
        df_wdir_alltp = df_wdir.copy()
    else:
        df_wdir_alltp = df_wdir_alltp.join(df_wdir)
    print(df_wdir_alltp)

df_freq_alltp.columns = ['freq']
df_freq_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/type_frequency_{}.csv'.format(city), encoding='gbk')
df_tmp_mean_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/temp_mean_{}.csv'.format(city), encoding='gbk')
df_humi_mean_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/humi_mean_{}.csv'.format(city), encoding='gbk')
df_slp_mean_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/slp_mean_{}.csv'.format(city), encoding='gbk')
df_wdir_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/wdir_mean_{}.csv'.format(city), encoding='gbk')
df_wspd_mean_alltp.to_csv(sys.path[0]+'/db/hist_avg_param/wspd_mean_{}.csv'.format(city), encoding='gbk')

print('All file saved to {}'.format(sys.path[0]+'/db/hist_avg_param'))
