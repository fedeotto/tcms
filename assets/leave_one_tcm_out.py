import pandas as pd
import numpy as np
import os
import sys
from assets import fit
from common import chem
from omegaconf import DictConfig
from assets.cbfv.composition import generate_features
from assets.random_forest.RandomForest import RandomForest
from assets.evaluate import eval_crabnet
from sklearn.preprocessing import StandardScaler
import torch

def get_tcms_split(df     : pd.DataFrame, #sigma or gap
                   family : str = 'ZnO',
                   dopant : str = 'Al'):
     
    if dopant == 'Al-Sn' or dopant=='Sn-Al':
        double_dopant = True
    else:
        double_dopant = False

    pristine_elems = chem._element_composition_L(family)[0]

    tcms_formulae  = []
    for formula in df['formula']:
        elems, _ = chem._element_composition_L(formula)
        if family == 'ZnO':
            if all(element in elems for element in pristine_elems) and ((len(elems) == 3 \
             and dopant in elems) or (len(elems)==4 and 'Al' in elems and 'Sn' in elems and double_dopant)):
                    tcms_formulae.append(formula)
        elif family == 'SnO2':
            if formula == 'In1.96Sn0.04O2.85':
                continue
            if all(element in elems for element in pristine_elems) \
                and len(elems)==3 and dopant in elems and 'O2' in formula:
                tcms_formulae.append(formula)
        
        elif family == 'In2O3':
            if formula == 'In1.96Sn0.04O2.85':
                tcms_formulae.append(formula)
            elif all(element in elems for element in pristine_elems) \
                and len(elems)==3 and dopant in elems and 'O3' in formula:
                tcms_formulae.append(formula)
            
    tcms    = df[df['formula'].isin(tcms_formulae)]
    no_tcms = df[~df['formula'].isin(tcms_formulae)]

    materials_to_drop = [] #to be eventually used
    if family == 'In2O3' and dopant == 'Sn':
        #Dropping all In:SnO2 materials
        pristine_elems    = ['Sn','O']
        dopant            = 'In'
        for formula in no_tcms['formula']:
            elems, _ = chem._element_composition_L(formula)
            if all(element in elems for element in pristine_elems) \
                and len(elems)==3 and dopant in elems and 'O2' in formula:
                materials_to_drop.append(formula)
    
    elif family == 'SnO2' and dopant == 'In':        
        #Dropping all Sn:In2O3 materials
        pristine_elems    = ['In','O']
        dopant            = 'Sn'
        for formula in no_tcms['formula']:
            if formula == 'In1.96Sn0.04O2.85':
                materials_to_drop.append(formula)
            else:
                elems, _ = chem._element_composition_L(formula)
                if all(element in elems for element in pristine_elems) \
                    and len(elems)==3 and dopant in elems and 'O3' in formula:
                    materials_to_drop.append(formula)

    no_tcms = no_tcms[~no_tcms['formula'].isin(materials_to_drop)]
    return tcms, no_tcms


def leave_one_tcm_out(family,
                      dopant,
                      cfg: DictConfig):
    """
    Simulates the discovery of TCMs by holding out a doped class of TCMs at a time.
    Possible families and dopants:
    - ZnO : Al, Ga
    - SnO2: Zn, Ta, Sb, In, Ga, Ti, Mn, W, Nb, Al-Sn
    - In2O3: Sn
    - CdO: To be added
    """

    torch.manual_seed(1234)
    np.random.seed(1234)
    pristine_elems = chem._element_composition_L(family)[0]

    double_dopant = False

    if dopant == 'Al-Sn' or dopant=='Sn-Al':
        double_dopant = True

    sigma = pd.read_excel(cfg.action.sigma_path)
    gap   = pd.read_excel(cfg.action.gap_path)
    
    sigma_tcms, sigma_no_tcms = get_tcms_split(sigma, 
                                               family, 
                                               dopant)

    gap_tcms, gap_no_tcms = get_tcms_split(gap, 
                                            family, 
                                            dopant)
    
    sigma_no_tcms = sigma_no_tcms.reset_index(drop=True)
    gap_no_tcms   = gap_no_tcms.reset_index(drop=True)
    
    mutual_tcms = set(sigma_tcms['formula']).intersection(set(gap_tcms['formula']))
    sigma_tcms  = sigma_tcms[sigma_tcms['formula'].isin(mutual_tcms)]
    gap_tcms    = gap_tcms[gap_tcms['formula'].isin(mutual_tcms)]

    #Preprocessing test datasets
    sigma_tcms =sigma_tcms.drop('Entry',axis=1)
    sigma_tcms.rename(columns={'Sigma (S/cm)' : 'target'}, inplace=True)
    sigma_tcms['target'] = np.log10(sigma_tcms['target'])

    gap_tcms = gap_tcms.drop('Entry',axis=1)
    gap_tcms.rename(columns={'Eg (eV)' : 'target'}, inplace=True)


    table_tcms_sigma = pd.DataFrame({'formula':sigma_tcms['formula'],'sigma_pred log10 (S/cm)':None,'sigma_uncert log10 (S/cm)':None})
    table_tcms_sigma.reset_index(drop=True, inplace=True)
    table_tcms_gap   = pd.DataFrame({'formula':gap_tcms['formula'],'eg_pred (eV)':None,'eg_uncert (eV)':None})
    table_tcms_gap.reset_index(drop=True, inplace=True)

    if cfg.model.name == 'crabnet':
        cfg.data.name = 'conductivity'
        fit.fit_crabnet(cfg      = cfg,
                        df       = sigma_no_tcms,
                        save_path= 'tmp')
        
        preds_sigma, unc_sigma = fit.predict_crabnet(sigma_tcms,
                                                    cfg=cfg,
                                                    predict_path='tmp')
        cfg.data.name = 'bandgap'
        fit.fit_crabnet(cfg      = cfg,
                        df       = gap_no_tcms,
                        save_path= 'tmp')
        
        preds_gap, unc_gap = fit.predict_crabnet(gap_tcms,
                                                cfg=cfg,
                                                predict_path='tmp')
    elif cfg.model.name == 'rf':
        cfg.data.name = 'conductivity'
        fit.fit_rf(cfg       = cfg,
                    df       = sigma_no_tcms,
                    save_path= 'tmp')
        
        preds_sigma, unc_sigma = fit.predict_rf(sigma_tcms,
                                                cfg=cfg,
                                                predict_path='tmp')
        cfg.data.name = 'bandgap'
        fit.fit_rf(cfg       = cfg,
                    df       = gap_no_tcms,
                    save_path= 'tmp')
        
        preds_gap, unc_gap = fit.predict_rf(gap_tcms,
                                            cfg=cfg,
                                            predict_path='tmp')
    
    table_tcms_sigma['sigma_actual log10 (S/cm)'] = sigma_tcms['target'].values
    table_tcms_sigma['sigma_pred log10 (S/cm)']   = preds_sigma
    table_tcms_sigma['sigma_uncert log10 (S/cm)'] = unc_sigma

    table_tcms_gap['eg_actual (eV)']             = gap_tcms['target'].values
    table_tcms_gap['eg_pred (eV)']               = preds_gap
    table_tcms_gap['eg_uncert (eV)']             = unc_gap

    table_tcms_sigma.to_excel(f'eval_results/LOTCMO/{family}_{dopant}_{cfg.model.name}_conductivity.xlsx',index=False)
    table_tcms_gap.to_excel(f'eval_results/LOTCMO/{family}_{dopant}_{cfg.model.name}_bandgap.xlsx',index=False)