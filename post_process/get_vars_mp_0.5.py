# Auth: cj
# del percentile calculation part -- 0922
# del interpolate_na() in do_load_data() -- 0924
# def ver 0.5 20230130

import netCDF4 as nc4
import wrf
import pandas as pd
import numpy as np
import os
import sys
import arrow
from multiprocessing import Pool
import pickle

start_year = '2019' 
sub_fldr_dir = 'vinterp_d02'
#domain = 'd01'
domain = 'd02'
wrfout_dir = '/home/dataop/data/nmodel/wrf_fc/'

#set grid in wrf lonlat
we_range = np.arange(0, 222) #d02
#we_range = np.arange(0, 282) #d01
sn_range = np.arange(0, 162) #d02
#sn_range = np.arange(0, 183) #d01
lvls = [10, 500, 700, 850]


def get_ymd(date):
    """
    :param date: arrow date
    """
    y = date.format('YYYY')
    ym = date.format('YYYYMM')
    ymd = date.format('YYYYMMDD')
    return y, ym, ymd


def get_now():
    return arrow.now().format('YYYYMMDD HH:mm:ss')


def do_load_data(cluster_type, idx, fldir):
    print('Type', cluster_type, 'loading data: [', idx+1, '/', len_type, ']')
    # print('loading file({})'.format(fldir))
    ncfile = nc4.Dataset(fldir)

    uwind = wrf.getvar(ncfile, 'ua')
    vwind = wrf.getvar(ncfile, 'va')
    z = wrf.getvar(ncfile, 'z')

    slp = wrf.getvar(ncfile, 'slp')
    u_interp = wrf.vinterp(ncfile, uwind, vert_coord="pressure", interp_levels=lvls, field_type='z')
    v_interp = wrf.vinterp(ncfile, vwind, vert_coord="pressure", interp_levels=lvls, field_type='z')
    h_interp = wrf.vinterp(ncfile, z, vert_coord="pressure", interp_levels=lvls, field_type='z', extrapolate=True)
    
    slp = slp.isel(south_north=sn_range, west_east=we_range).values
    uv10 = wrf.getvar( ncfile, 'uvmet10' )
    u10 = uv10[0].isel( south_north=sn_range, west_east=we_range ).values
    u500 = u_interp.isel(interp_level=1, south_north=sn_range, west_east=we_range).values
    u700 = u_interp.isel(interp_level=2, south_north=sn_range, west_east=we_range).values
    u850 = u_interp.isel(interp_level=3, south_north=sn_range, west_east=we_range).values
    v10 = uv10[1].isel( south_north=sn_range, west_east=we_range ).values
    v500 = v_interp.isel(interp_level=1, south_north=sn_range, west_east=we_range).values
    v700 = v_interp.isel(interp_level=2, south_north=sn_range, west_east=we_range).values
    v850 = v_interp.isel(interp_level=3, south_north=sn_range, west_east=we_range).values
    h500 = h_interp.isel(interp_level=1, south_north=sn_range, west_east=we_range).values
    h700 = h_interp.isel(interp_level=2, south_north=sn_range, west_east=we_range).values
    h850 = h_interp.isel(interp_level=3, south_north=sn_range, west_east=we_range).values
    
    return {'slp': slp, 'u10':u10, 'v10':v10, 
            'u500':u500, 'v500':v500, 'h500':h500,
            'u700':u700, 'v700':v700, 'h700':h700,
            'u850':u850, 'v850':v850, 'h850':h850,
            }


print(get_now(), 'start to calculate mean var of types...')
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


for cluster_type in cluster_types:
    print(get_now(), 'calculating type {}...'.format(cluster_type))
    
    flnm_list = df_cluster[df_cluster.type == cluster_type].index
    freq_type = len(flnm_list)/len(df_cluster)
    print('type {}  frequency:'.format(cluster_type), freq_type)
    # se_freq_type = pd.Series(freq_type, ['Type {}'.format(cluster_type)])
    # df_freq_alltp = pd.concat([df_freq_alltp, se_freq_type])
    # df_pctg_types[cluster_type] = round(pctg_type, 4)
    # fldir_list = [target_fldrdir+'/'+'wrfout_'+domain+'_'+flnm.replace('12:','04:') for flnm in flnm_list]

    fldir_list = []
    for flnm in flnm_list:
        flnm = flnm.replace('12:','04:')
        _date = arrow.get(flnm[:10]).shift(days=-1)
        y, ym, ymd = get_ymd(_date)
        ymd12 = ymd + '12'
        flnm = 'wrfout_' + domain + '_' + flnm
        fldir = os.path.join(wrfout_dir, y, ym, ymd12, flnm)
        fldir_list.append(fldir)
    
    len_type = len(fldir_list)
    print('dates of this type:', len_type)
    
    # init, get first date vars
    load_result = do_load_data(cluster_type, 1, fldir_list[0])
    slp = load_result['slp']
    u10 = load_result['u10']
    v10 = load_result['v10']
    u500 = load_result['u500']
    v500 = load_result['v500']
    h500 = load_result['h500']
    u700 = load_result['u700']
    v700 = load_result['v700']
    h700 = load_result['h700']
    u850 = load_result['u850']
    v850 = load_result['v850']
    h850 = load_result['h850']

    p = Pool(os.cpu_count())
    for idx, fldir in enumerate(fldir_list[1:]):
        print('loading file({}).'.format(fldir))
        sync = p.apply_async(do_load_data, args=(cluster_type, idx, fldir,))
        load_result = sync.get()
        slp_temp = load_result['slp']
        u10_temp = load_result['u10']
        v10_temp = load_result['v10']
        u500_temp = load_result['u500']
        v500_temp = load_result['v500']
        h500_temp = load_result['h500']
        u700_temp = load_result['u700']
        v700_temp = load_result['v700']
        h700_temp = load_result['h700']
        u850_temp = load_result['u850']
        v850_temp = load_result['v850']
        h850_temp = load_result['h850']
        slp = np.mean([slp, slp_temp], axis=0)
        u10 = np.mean([u10, u10_temp], axis=0)
        v10 = np.mean([v10, v10_temp], axis=0)
        u500 = np.mean([u500 , u500_temp], axis=0)
        v500 = np.mean([v500 , v500_temp], axis=0)
        h500 = np.mean([h500 , h500_temp], axis=0)
        u700 = np.mean([u700 , u700_temp], axis=0)
        v700 = np.mean([v700 , v700_temp], axis=0)
        h700 = np.mean([h700 , h700_temp], axis=0)
        u850 = np.mean([u850 , u850_temp], axis=0)
        v850 = np.mean([v850 , v850_temp], axis=0)
        h850 = np.mean([h850 , h850_temp], axis=0)
    p.close()
    p.join()

    with open(sys.path[0]+'/db/{}/slp_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(slp, file)
    with open(sys.path[0]+'/db/{}/u10_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(u10, file)
    with open(sys.path[0]+'/db/{}/v10_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(v10, file)
    with open(sys.path[0]+'/db/{}/u500_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(u500, file)
    with open(sys.path[0]+'/db/{}/v500_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(v500, file)
    with open(sys.path[0]+'/db/{}/h500_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(h500, file)
    with open(sys.path[0]+'/db/{}/u700_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(u700, file)
    with open(sys.path[0]+'/db/{}/v700_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(v700, file)
    with open(sys.path[0]+'/db/{}/h700_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(h700, file)
    with open(sys.path[0]+'/db/{}/u850_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(u850, file)
    with open(sys.path[0]+'/db/{}/v850_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(v850, file)
    with open(sys.path[0]+'/db/{}/h850_type{}.data'.format(sub_fldr_dir, cluster_type), 'wb') as file:
        pickle.dump(h850, file)

print('All file saved to {}'.format(sys.path[0]+'/db/{}'.format(sub_fldr_dir)))
