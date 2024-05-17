# auth CJ 2021/12/31
# match wrf with som type -- winter
# using only slp and 10m uv

import arrow
import numpy as np
import netCDF4 as nc4
import wrf
import xarray as xr
import pickle
import pandas as pd
import os
import yaml


def _euclidien_distance(x, y):
    if len(x) != len(y):
        print('Error: Sizes of datasets does not match, please check the code.')
        exit()
        return -1
    else:
        total_sum = 0 
        for i in range(0, len(x)):
            var_x = np.linalg.norm(x[i])
            var_y = np.linalg.norm(y[i])
            sum = np.nansum(np.square(var_x - var_y))
#            print('eu sum:', sum) 
            total_sum += sum
        d = np.sqrt(total_sum)
        return d


#todo: auto find wrf file at afternoon in T+1

# e.g. 'spring', 'summer', 'autumn', 'winter'
season = 'winter'
# todo: auto set season
we_range = np.arange(0, 222)
sn_range = np.arange(0, 162)
wrf_root = '/home/dataop/data/nmodel/wrf_fc'
fmt_dir = '/YYYY/YYYYMM/YYYYMMDD12'
domain = 'd02'

config_flnm = 'config_shantou.yml'
current_dir = os.path.dirname( os.path.abspath(__file__) )
config_dir = os.path.join(current_dir, config_flnm)
config = yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader)
day_shift = config['baseday_shift']
#baseday = arrow.now().shift(days=day_shift)
baseday = arrow.get('2022-12-21')
wrfday_str = baseday.shift(days=-2).format(fmt_dir)
wrf_dir_yesterday = wrf_root + wrfday_str
fmt_wrf = 'YYYY-MM-DD'

som_cluster_dir = '/home/arcspv/disk65/Prism/Prism_{}8tp/Prism-master/db'.format(season) # som result
diff_gpm_dir = som_cluster_dir + '/vinterp_diff_gpm' # som interp result
xdata = xr.open_dataset(r'{}/som_cluster.nc'.format(som_cluster_dir))
df = pd.DataFrame()

for j in range(1,5):
    date_str = baseday.shift(days=j).format(fmt_wrf)
    print(date_str)
    wrf_fl_dir = '{}/wrfout_{}_{}_04:00:00'.format(wrf_dir_yesterday, domain, date_str) 
    if not os.path.exists(wrf_fl_dir):
        wrf_dir_yesterday_tmp = wrf_root + baseday.shift(days=-1).format(fmt_dir)
        wrf_fl_dir = '{}/wrfout_{}_{}_04:00:00'.format(wrf_dir_yesterday_tmp, domain, date_str)
    wrf_fcst = nc4.Dataset(wrf_fl_dir)
    uv_fcst = wrf.getvar(wrf_fcst, 'uvmet10').isel(south_north=sn_range, west_east=we_range)
    u_fcst = uv_fcst[0].values
    v_fcst = uv_fcst[1].values
    slp_fcst = wrf.getvar(wrf_fcst, 'slp').isel(south_north=sn_range, west_east=we_range).values
    z_fcst = wrf.getvar(wrf_fcst, 'z')
    h500_fcst = wrf.vinterp(wrf_fcst, z_fcst, vert_coord='pressure', interp_levels=[500], field_type='z').values
    # dataset_fcst = [u_fcst, v_fcst, slp_fcst, h500_fcst]
    dataset_fcst = [u_fcst, v_fcst, slp_fcst]
    eu_ds = []
    for i in range(0,8):
        type_data = xdata.isel(n_nodey=i)
        u10_som = np.array(type_data['som_cluster'].isel(nvar=0))[0]
        v10_som = np.array(type_data['som_cluster'].isel(nvar=1))[0]
        h500_som = np.array(type_data['som_cluster'].isel(nvar=2))[0]
        with open(r'{}/slp_type{}.data'.format(diff_gpm_dir, i+1), 'rb') as file:
            slp_som = pickle.load(file)
        # dataset_som = [u10_som, v10_som, slp_som, h500_som]
        dataset_som = [u10_som, v10_som, slp_som]
        d = _euclidien_distance(dataset_som, dataset_fcst)
        eu_ds.append(d)
        # print(j, i+1, d)
    min_eu_d = np.min(eu_ds)
    tp = eu_ds.index(min_eu_d)
    # print(date_str, tp+1, min_eu_d)
    df_date = pd.DataFrame(columns=['dates', 'type'])
    df_date.loc[0] = date_str, tp+1
    # print(df_date)
    df = pd.concat([df, df_date])
print(df)

if season == 'spring':
    type_prefix = 'S1_N06_C0'
if season == 'summer':
    type_prefix = 'S2_N08_C0'
if season == 'autumn':
    type_prefix = 'S3_N06_C0'
if season == 'winter':
    type_prefix = 'S4_N08_C0'

df_shantou = df.copy()
df_shantou.type = df_shantou.type.apply(lambda x: '{}{}'.format(type_prefix, x))

fmt_dir = '/YYYY/YYYYMM/YYYYMMDD'
shantou_root_dir = '/home/arcspv/public_html/weather_pattern_shantou/SOM_OUTPUT'
shantou_dir = shantou_root_dir + baseday.format(fmt_dir)
if not os.path.exists(shantou_dir):
    os.makedirs(shantou_dir)
df_shantou.to_csv('{}/SOM_OUTPUT_{}.csv'.format(shantou_dir, baseday.format('YYYYMMDD')), index=False, header=False)
print('shantou SOM_OUTPUT file saved at:', '{}/SOM_OUTPUT_{}.csv'.format(shantou_dir, baseday.format('YYYYMMDD')))

