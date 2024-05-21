'''
prismp:
PRIsm SiMPlified
author CJ
base on PRISM v0.99
'''

import prism
import arrow
import wrf_reader
import config as cfg
import shutil as sh
import os, sys, logging
from os import path

def get_logger():
    dir_log = cfg.dir_runlog
    dirname_log = cfg.dirname_log
    fmt_time = cfg.fmt_time
    if path.exists( dir_log ):
        timestamp = arrow.now().format( fmt_time )
        flnm_log_bak = 'run.log_' + timestamp
        dir_log_bak = path.join( dirname_log, flnm_log_bak )
        sh.move( dir_log, dir_log_bak )

    logger = logging.getLogger('runlog')
    fmt = '%(asctime)s - %(levelname)s: %(message)s'
    datefmt = '%Y/%m/%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    logger.setLevel(logging.DEBUG)
    stdout = logging.StreamHandler()
    stdout.setFormatter(formatter)
    logout = logging.FileHandler(dir_log)
    logout.setFormatter(formatter)
    logger.addHandler(stdout)
    logger.addHandler(logout)
    return logger

def check_input():
    # chekc if nc files exist
    # exist program if no nc files
    dirname_nc = cfg.dirname_nc
    if not path.exists( dirname_nc ):
        msg = 'Dirname:' + dirname_nc + ' not exists.'
        logger.error( msg )
        logger.info( 'Program exit.')
        exit()
    fl_list = os.listdir( dirname_nc )
    fl_list = [x for x in fl_list if 'wrfout' in x]
    fl_num = len( fl_list )
    if fl_num == 0:
        msg = 'No wrfout file in ' + dirname_nc 
        logger.error(msg)
        logger.info( 'Program exit.')
        exit()
 
def main():
    check_input()

    wrf_mesh = wrf_reader.WrfMesh( logger )
    wrf_mesh.load_wrf()

    model = prism.Prism( wrf_mesh, logger )
    model.train( logger )
    model.evaluate( logger )
    model.save( logger )

    logger.info('PRISM ACCOMPLISHED')

if __name__=='__main__':
    logger = get_logger()
    main()
