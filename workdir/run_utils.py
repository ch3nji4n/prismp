import arrow
import os, sys
import numpy as np
import pandas as pd
import config as cfg

DEG2RAD = np.pi / 180.0


def link_path(cfg, domain):
    """ link path wrfout to input dir """
    dateseries = pd.date_range(start = cfg['TRAINING']['training_start'], end = cfg['TRAINING']['training_end'])
    len_date = len(dateseries)
    for idx, datestamp in enumerate(dateseries):
        
        if idx % 100 == 0:
            write_log('''wrfout_%s relink done by %d/%d, 
            please be patient, may take several minutes for long-term data...''' % (domain, idx, len_date))

        yyyy = datestamp.strftime('%Y')
        mm = datestamp.strftime('%m')
        dd = datestamp.strftime('%d')
        
        datestamp_p1 = datestamp + datetime.timedelta(days = 1)
        yyyy_p1 = datestamp_p1.strftime('%Y')
        mm_p1 = datestamp_p1.strftime('%m')
        dd_p1 = datestamp_p1.strftime('%d')

        src_wrfpath = cfg['OTHER']['src_wrf'] + yyyy
        src_wrfpath = src_wrfpath + '/' + yyyy + mm + '/' + yyyy + mm + dd + '12'
        
        try:
            os.system('ln -sf ' + src_wrfpath + f'/wrfout_{domain}_' + yyyy + '-' + mm + '-' + dd + '* ' + CWD + '/input/training/')
            os.system('ln -sf ' + src_wrfpath + f'/wrfout_{domain}_' + yyyy_p1 + '-' + mm_p1 + '-' + dd_p1 + '_0* ' + CWD + '/input/training/')
            os.system('ln -sf ' + src_wrfpath + f'/wrfout_{domain}_' + yyyy_p1 + '-' + mm_p1 + '-' + dd_p1 + '_1[0-1]* ' + CWD + '/input/training/')
        except:
            write_log('wrfout_{domain}_' + yyyy + '-' + mm + '-' + dd + '_12:00:00 not found',30)

def link_realtime(cfg, domain):
    """ link realtime wrfout to input dir """
    today = datetime.datetime.now()
    offset_day = int(cfg['OTHER']['relink_realtime_offsetday'])
    today = today-datetime.timedelta(days = offset_day)
    while 1:
        yyyy = today.strftime('%Y')
        mm = today.strftime('%m')
        dd = today.strftime('%d')
        src_wrfpath = cfg['OTHER']['src_wrf'] + yyyy
        src_wrfpath = src_wrfpath + '/' + yyyy + mm + '/' + yyyy + mm + dd + '12'
        # test if file exist
        if os.path.exists(src_wrfpath):
            os.system('rm -f  ' + CWD + '/input/inference/*')
            os.system('ln -sf ' + src_wrfpath + f'/wrfout_{domain}* ' + CWD + '/input/inference/')
            write_log(src_wrfpath + ' has been successfully linked!')
            break
        else:
            write_log(src_wrfpath + ' not found, try 1 day before',30)
            today = today-datetime.timedelta(days = 1)

def get_std_dim0(data):
    """ standardize the series on the 0 dim """
    data_mean=data.mean(axis=0)
    data_std=data.std(axis=0)
    data=(data-data_mean)/data_std
    return data, data_mean, data_std

