#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:31:31 2023
@author: federico
"""
import pandas as pd
import numpy as np
import os
from common.utils import preprocess_data_ml
from assets.cbfv.composition import generate_features
import pickle
from assets.fit import predict_crabnet, predict_rf
from common import chem
from pymatgen.core import Composition
from common.chem import _element_composition_L
from omegaconf import DictConfig
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def screen_materials_list(cfg: DictConfig):
    
    materials_list = pd.read_excel(cfg.action.screen_path)

    if cfg.model.name == 'crabnet':
        screen_result = screen_crabnet(materials_list, cfg=cfg)
    elif cfg.model.name == 'rf':
        screen_result = screen_rf(materials_list, cfg=cfg)

    screen_result = screen_result.reset_index(drop=True)
    screen_result['Phi_M']     = screen_result['eg_pred (eV)'] * screen_result['sigma_pred log10 (S/cm)']

    #propagating the uncertainties
    screen_result['Phi_M_std'] = np.sqrt((screen_result['eg_uncert (eV)'] * screen_result['sigma_pred log10 (S/cm)'])**2 + \
                                        (screen_result['eg_pred (eV)'] * screen_result['sigma_uncert log10 (S/cm)'])**2)
    
    screen_result['Phi_M_std_adj'] = screen_result['Phi_M'] - screen_result['Phi_M_std']
    #Dropping duplicates in case of targeting tcms
    if cfg.action.material_type == 'tcms':
        screen_result = screen_result.drop_duplicates(subset='formula', keep='first')

    #Ranking materials based on Phi_M
    screen_result = screen_result.sort_values(by='Phi_M_std_adj', ascending=False).reset_index(drop=True)

    '''Storing the results in a csv file'''
    if cfg.action.all_models:
        model_name ='all'
    else:
        model_name = cfg.model.name
    screen_result.to_csv(f'./screen_results/{model_name}_{cfg.action.material_type}.csv', index=False)

def screen_crabnet(materials_list, cfg: DictConfig):
    results_crab = pd.DataFrame() #empty dataframe to store the results.

    mat_list_crab= materials_list.copy()
    entries      = mat_list_crab.pop('source')
    sources      = mat_list_crab.pop('source')

    mat_list_crab['target'] = 0 #dummy target.

    print(f'\n--- Screening materials list using CrabNet.. ---\n')
    mat_list_crab = preprocess_data_ml(mat_list_crab,
                                        elem_prop='mat2vec',
                                        screening=True,
                                        shuffle  = False)    

    # extracting correct entries with respect to preprocessed materials list.
    entries_crab          = entries.loc[mat_list_crab.index]
    sources_crab          = sources.loc[mat_list_crab.index]

    results_crab['source']= sources_crab
    results_crab['Entry'] = entries_crab

    cfg.data.name = 'conductivity'
    pred_sigma, uncert_sigma = predict_crabnet(mat_list_crab, cfg=cfg)

    mat_list_crab.drop('count', axis=1,inplace=True)

    cfg.data.name = 'bandgap'
    pred_bandgap, uncert_bandgap = predict_crabnet(mat_list_crab, cfg=cfg)

    results_crab['formula']       = mat_list_crab['formula'].tolist()
    results_crab['eg_pred (eV)']  = np.round_(pred_bandgap,3)
    results_crab['eg_uncert (eV)']= np.round_(uncert_bandgap,3)

    results_crab['sigma_pred log10 (S/cm)']   = np.round_(pred_sigma, 2)
    results_crab['sigma_uncert log10 (S/cm)'] = np.round_(uncert_sigma,2)
    results_crab['model']   = 'crabnet'
    return results_crab.reset_index(drop=True)


def screen_rf(materials_list: pd.DataFrame, cfg:DictConfig):
    results_rf = pd.DataFrame() #empty dataframe to store the results.
    mat_list_rf = materials_list.copy()

    entries = mat_list_rf.pop('Entry')
    sources = mat_list_rf.pop('source')

    mat_list_rf['target'] = 0 #dummy target.

    print(f'\n--- Screening materials list using RF.. ---\n')
    mat_list_rf = preprocess_data_ml(mat_list_rf,
                                    elem_prop=cfg.model.elem_prop,
                                    screening=True,
                                    shuffle=False)

    # taking correct entries with respect to preprocessed materials list.
    entries_rf            = entries.loc[mat_list_rf.index]
    sources_rf            = sources.loc[mat_list_rf.index]

    results_rf['source']  = sources_rf
    results_rf['Entry']   = entries_rf

    cfg.data.name = 'conductivity'
    pred_sigma, uncert_sigma = predict_rf(mat_list_rf, cfg=cfg)

    cfg.data.name = 'bandgap'
    pred_bandgap, uncert_bandgap = predict_rf(mat_list_rf, cfg=cfg)

    results_rf['formula']       = mat_list_rf['formula']
    results_rf['eg_pred (eV)']  = np.round_(pred_bandgap,2)
    results_rf['eg_uncert (eV)']= np.round_(uncert_bandgap,2)

    results_rf['sigma_pred log10 (S/cm)']   = np.round_(pred_sigma,2)
    results_rf['sigma_uncert log10 (S/cm)'] = np.round_(uncert_sigma,2)
    results_rf['model']   = 'rf'

    return results_rf