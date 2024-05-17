"""
Core Component: Prism Classifier 
    Classes: 
    -----------
        prism: core class, weather classifier 

    Functions:
    -----------
"""
import copy
import json
import arrow
import pickle
import sys, os
import minisom
import numpy as np
import xarray as xr
import pandas as pd
import config as cfg
import sklearn.metrics as skm
from utils import utils

class Prism:

    '''
    Prism clusterer, use wrf mesh variables 
    
    Attributes
    -----------

    Methods
    -----------
    train(), train the model by historical WRF data
    evaluate(), evaluate the model performance by several metrics

    '''
    
    def __init__(self, wrf_mesh, cfg, call_from='trainning'):
        """ construct prism classifier """
        self.nrec = wrf_mesh.n_rec
        nrow = self.nrow = wrf_mesh.n_row
        ncol = self.ncol = wrf_mesh.n_col

        self.nfea = nrow*ncol 
        varlist = self.varlist = wrf_mesh.varlist 
        self.nvar = len(varlist)
        self.dateseries = wrf_mesh.dateseries

        self.xlat, self.xlong = wrf_mesh.xlat, wrf_hdl.xlong
        
        # self.data(recl, nvar, nrow*ncol)
        self.data = np.empty([self.nrec, self.nvar,nrow*ncol])

        if call_from == 'trainning':
            for idx, var in enumerate(varlist):
                raw_data = wrf_mesh.data_dic[var].values.reshape((self.nrec,-1))
                self.data[:,idx,:] = raw_data
                
            self.preprocess = cfg['TRAINING']['preprocess_method']
            self.n_nodex = int(cfg['TRAINING']['n_nodex'])
            self.n_nodey = int(cfg['TRAINING']['n_nodey'])
            self.sigma = float(cfg['TRAINING']['sigma'])
            self.lrate = float(cfg['TRAINING']['learning_rate'])
            self.iterations = int(cfg['TRAINING']['iterations'])
            self.nb_func = cfg['TRAINING']['nb_func']

            if self.preprocess  ==  'temporal_norm':
                self.data, self.mean, self.std = utils.get_std_dim0(self.data)
 
        self.data = self.data.reshape((self.nrec,-1))

    def train(self, train_data=None, verbose=True):
        """ train the prism classifier """
        if verbose:
            utils.write_log(print_prefix+'trainning...')
        
        if train_data is None:
            train_data = self.data
        
        # init som
        som = minisom.MiniSom(
                self.n_nodex, self.n_nodey, self.nvar*self.nfea, 
                neighborhood_function = self.nb_func,
                sigma = self.sigma, 
                learning_rate = self.lrate) 
        
        # train som
        som.train(train_data, self.iterations, verbose = verbose) 

        self.q_err = som.quantization_error(train_data)

        self.winners = [som.winner(x) for x in train_data]
        self.som = som

    def evaluate(self,cfg, train_data=None, verbose=True):
        """ evaluate the clustering result """
        if verbose: 
            utils.write_log(print_prefix+'prism evaluates...')
        
        if train_data is None:
            train_data  =  self.data
        
        edic = {'quatization_error':self.q_err}
        
        label = [str(winner[0])+str(winner[1]) for winner in self.winners]
        s_score = skm.silhouette_score(train_data, label, metric = 'euclidean')
        
        edic.update({'silhouette_score':s_score})
        
        if verbose:
            utils.write_log(print_prefix+'prism evaluation dict: %s' % str(edic))

        edic.update({'cfg_para':cfg._sections})
        
        self.edic = edic

    def save(self):
        """ archive the prism classifier in database """

        utils.write_log(print_prefix+'prism archives...')
        
        # archive evaluation dict
        with open(cfg.dir_evaluation, 'w') as f:
            json.dump(self.edic,f)

        # archive model
        with open(cfg.dir_archive, 'wb') as outfile:
            pickle.dump(self.som, outfile)

        # archive classification result in csv
        with open(cfg.dir_cluster_csv, 'w') as f:
            for datestamp, winner in zip(self.dateseries, self.winners):
                f.write(datestamp.strftime('%Y-%m-%d_12:00:00,')+str(winner[0])+','+str(winner[1])+'\n')

        # archive classification result in netcdf
        centroid = self.som.get_weights()
        centroid = centroid.reshape(self.n_nodex, self.n_nodey, self.nvar, self.nrow, self.ncol)
        
        ds_out = self.org_output_nc(centroid)
        out_fn = cfg.dir_cluster_nc
        ds_out.to_netcdf(out_fn)
        
        utils.write_log(print_prefix+'prism construction is completed!')

    def org_output_nc(self, centroid):
        """ organize output file """
        ds_vars = {   
            'som_cluster':(['n_nodex','n_nodey','nvar', 'nrow','ncol'], centroid),
            'var_vector':(['ntimes','ngrids'], self.data),
            'xlat':(['nrow', 'ncol'], self.xlat),
            'xlong':(['nrow', 'ncol'], self.xlong)}
        
        ds_coords = {'nvar':self.varlist}
        ds_attrs = {
            'preprocess_method':self.preprocess,
            'neighbourhood_function':self.nb_func}
        
        if self.preprocess  ==  'temporal_norm':
            self.mean = self.mean.reshape(self.nvar, self.nrow, self.ncol)
            self.std = self.std.reshape(self.nvar, self.nrow, self.ncol)
            
            # reverse temporal_norm
            for ii in range(0, self.n_nodex):
                for jj in range(0, self.n_nodey):
                    centroid[ii,jj,:,:,:] = centroid[ii,jj,:,:,:]*self.std+self.mean
            ds_vars.update({
                'mean':(['nvar', 'nrow', 'ncol'], self.mean),
                'std':(['nvar','nrow', 'ncol'], self.std)}) 
        
        ds_out =  xr.Dataset(
            data_vars = ds_vars, coords = ds_coords,
            attrs = ds_attrs) 

        return ds_out

    def load(self):
        """ load the archived prism classifier in database """
        with open(cfg.dir_archive, 'rb') as infile:
            self.som  =  pickle.load(infile)

if __name__  ==  "__main__":
    pass
