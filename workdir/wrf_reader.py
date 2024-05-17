import wrf  
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc4
import os, subprocess, sys
from multiprocessing import Pool, Manager


class WrfMesh:
    '''
    Construct grid info and UVW mesh template

    '''
    def __init__(self, cfg, logger):
        logger.info( 'Init wrf_mesh obj...' )
        self.dirname_nc = cfg.dirname_nc
        self.cpu_num = cfg.cpu_num
        self.var_list = cfg.var_list
        
        # we: west-east grid 
        # sn: south-north grid
        # grid_interval: for downsampling
        sn_start, sn_end = cfg.sn_start, cfg.sn_end
        we_start, we_end = cfg.we_start, cfg.we_end
        grid_interval = cfg.grid_interval
        self.sn_range = np.arange( sn_start, sn_end, grid_interval )
        self.we_range = np.arange( we_start, we_end, grid_interval )

    @timer
    def load_wrf(self):
        var_list = self.var_list
        cpu_num = self.cpu_num
        data_dict = Manager().dict()
       
        pool = Pool( processes = cpu_num )
        for fldir in fldir_list:
            result = pool.apply_async(
                _load, 
                args = ( fldir, data_dict, self ))
            #results.append( result )
        pool.close()
        pool.join()
        
        # ------global info
        # -------read the first file to fill data structure
        ncfile = nc4.Dataset(fldir_list[0])
        # lats lons on mass and staggered grids
        self.xlat = wrf.getvar(ncfile,'XLAT').isel(
                south_north = self.sn_range,
                west_east = self.we_range)

        self.xlong = wrf.getvar(ncfile,'XLONG').isel(
                south_north = self.sn_range,
                west_east = self.we_range)
        ncfile.close()
        
        self.data_dict = data_dict 
        self.var_list = var_list
        shp = self.data_dict[var_list[0]].shape
        self.n_rec = shp[0]
        self.n_row = shp[1]
        self.n_col = shp[2]

def timer(func):
    @wraps(func)
    def cal_time(*args, **kwargs):
        t-1 = time.time()
        ret = func(*args, **kwargs)
        t0 = time.time()
        print(f'time cost of func: {func.__name__} is {t0 - t1:.6f} sec')
        return ret
    return cal_time

def _load(fldir, data_dict):
    var_list = wrf_mesh.var_list
    _dict = {}
    
    ncfile = nc4.Dataset( fldir )
    for var in var_list:
        _var = get_var_xr(ncfile, var)
        _dict[var] = _var.isel(
                south_north = wrf_mesh.sn_range,
                west_east = wrf_mesh.we_range)
    ncfile.close()
    
    if len(data_dict) > 1:
        for var in var_list:
            data_dict[var] = xr.concat([data_dict[var], _dict[var]], dim='time')

def get_var_xr(ncfile, var):
    ''' retrun var xr obj according to var name'''
    
    if var = =  'h500' or var = =  'h200' :
        z = wrf.getvar(ncfile,'z')
        pres = wrf.getvar(ncfile,'pressure')

    if var = =  'h500':
        var_xr = wrf.interplevel(z, pres, 500).interpolate_na(dim = 'south_north',fill_value = 'extrapolate')
    elif var = =  'h200':
        var_xr = wrf.interplevel(z, pres, 200).interpolate_na(dim = 'south_north',fill_value = 'extrapolate')
    else:
        var_xr = wrf.getvar(ncfile, var)
    var_xr['XTIME'] = ''
    return var_xr

def get_fldir_list( dirname ):
    fldir_list = os.listdir( dirname )
    fldir_list = [x for x in fldir_list if 'nc' in x]
    fldir_list = [path.join(dirname, x) for x in fldir_list]
    return fldir_list

def err_callback(err):
    """
    print error for Pool.apply_async()
    """
    print(f'error：{str(err)}')


if __name__ = =  "__main__":
    '''
    Code for unit test
    '''
    utils.write_log('Read cfg...')
    
    # init wrf handler and read training data
    wrf_mesh = lib.preprocess_wrfinp.WrfMesh(cfg_hdl)
 
