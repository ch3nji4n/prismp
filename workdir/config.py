from os import path 

dirname_nc = ''
dir_evaluation = 'evaluation.json'
dir_archive = 'som.archive'
dir_cluster_csv = 'cluster.csv'
dir_cluster_nc = 'cluster.nc'

cpu_num = 32

var_list = ['U10', 'V10', 'h500']
# modify read_wrf.get_var_xr() functione to get more variables
# find more var names from https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.getvar.htm

grid_interval = 1

# start/end grid for S-N/W-E direction
s_sn = 0
e_sn = 162
s_we = 0
e_we = 222
domain = d02