from os import path 

fmt_time = 'YYYYMMDD' 

dir_runlog = '/work/home/pathop/testfield/prismp-main/workdir/run.log'
dirname_log = '/work/home/pathop/testfield/prismp-main/log'
dirname_nc = '/work/home/pathop/prism_cj/data/link_prune_d01/20230226'
dir_output = '/work/home/pathop/testfield/prismp-main/output'
dir_evaluation = path.join( dir_output, 'evaluation.json')
dir_archive = path.join( dir_output, 'som.archive')
dir_cluster_csv = path.join( dir_output, 'cluster.csv')
dir_cluster_nc = path.join( dir_output, 'cluster.nc')

cpu_num = 32

var_list = ['U10', 'V10', 'h500']
# modify read_wrf.get_var_xr() functione to get more variables
# find more var names from https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.getvar.htm

grid_interval = 1

# start/end grid for S-N/W-E direction
sn_start = 0
sn_end = 162
we_start = 0
we_end = 222
domain = 'd02'

# SOM varisbles
n_nodex = 1
n_nodey = 8
sigma = 0.8
learning_rate = 0.5
iterations = 3000
nb_func = 'gaussian'
preprocess_method = 'temporal_norm'
