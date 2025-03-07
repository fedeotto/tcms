# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 20:53:06 2022

@author: fedeotto
"""

import pandas as pd
import numpy as np
import re
import collections
import tqdm
import pkg_resources
import joblib

all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
               'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
               'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
               'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
               'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
               'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
               'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
               'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
               'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
               'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
               'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
               'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

def generate_features(df, 
                      elem_prop='magpie', 
                      drop_duplicates=True,
                      extend_features=True,
                      only_avg=False,
                      sum_feat=False,
                      features_path=None,
                      ):
    
    if drop_duplicates:
        
        if df['formula'].value_counts()[0] > 1:
            df.drop_duplicates('formula', inplace=True)
            print('Duplicate formula(e) removed using default pandas function')


    cbfv_path = pkg_resources.resource_stream(__name__, f'element_properties/{elem_prop}.csv')

    #table with properties
    elem_props = pd.read_csv(cbfv_path)

    elem_props.index = elem_props['element'].values
    elem_props.drop(['element'], inplace=True, axis=1)

    elem_symbols = elem_props.index.tolist()
    #elem index just the index of element properties [0,96] for magpie
    elem_index = np.arange(0, elem_props.shape[0], 1)
    elem_missing = list(set(all_symbols) - set(elem_symbols))
    
                
    elem_props_columns = elem_props.columns.values

    column_names = np.concatenate(['avg_' + elem_props_columns,
                                   'dev_' + elem_props_columns,
                                   'range_' + elem_props_columns,
                                   'max_' + elem_props_columns,
                                   'min_' + elem_props_columns,
                                   'mode_' + elem_props_columns])
        
    if sum_feat:
        column_names = np.concatenate(['sum_' + elem_props_columns,
                                       column_names])

    # make empty list where we will store the property value
    targets = []
    # store formula
    formulae = []
    # add the values to the list using a for loop

    elem_mat = elem_props.values

    formula_mat = []
    count_mat = []
    frac_mat = []
    target_mat = []

    if extend_features: # if we want to include Tempereature..
        features = df.columns.values.tolist()
        features.remove('target')
        extra_features = df[features]

    for index in tqdm.tqdm(df.index.values, desc='Processing Input Data'):
        formula, target = df.loc[index, 'formula'], df.loc[index, 'target']
        if 'x' in formula:
            continue
        #l1 = ['Na', 'Cl'], l2 = ['[1.0, 1.0]']
        
        try:
            l1, l2 = _element_composition_L(formula)
        except:
            print(f'Problem at index {index}')
            
        formula_mat.append(l1)
        count_mat.append(l2)
        _, l3 = _fractional_composition_L(formula)
        frac_mat.append(l3)
        target_mat.append(target)
        formulae.append(formula)

    print('\tfeaturizing compositions...'.title())

    matrices = [formula_mat, count_mat, frac_mat, elem_mat, target_mat]
    elem_info = [elem_symbols, elem_index, elem_missing]
    feats, targets, formulae, skipped = _assign_features(matrices,
                                                         elem_info,
                                                         formulae,
                                                         sum_feat=sum_feat,
                                                         )

    print('\tcreating pandas objects...'.title())

    # split feature vectors and targets as X and y
    X = pd.DataFrame(feats, columns=column_names, index=formulae)
    y = pd.Series(targets, index=formulae, name='target')
    formulae = pd.Series(formulae, index=formulae, name='formula')
    if extend_features:
        extended = pd.DataFrame(extra_features, columns=features)
        extended = extended.set_index('formula', drop=True)
        X = pd.concat([X, extended], axis=1)

    # reset dataframe indices
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    formulae.reset_index(drop=True, inplace=True)

    # drop elements that aren't included in the elmenetal properties list.
    # These will be returned as feature rows completely full of NaN values.
    X.dropna(inplace=True, how='all')
    y = y.iloc[X.index]
    formulae = formulae.iloc[X.index]

    # get the column names
    cols = X.columns.values
    # find the median value of each column
    median_values = X[cols].median()
    # fill the missing values in each column with the column's median value
    X[cols] = X[cols].fillna(median_values)
    
    if only_avg:
        avg_columns = [col for col in X.columns if 'avg_' in col]
        X=X[avg_columns]
    
    if features_path is not None:
        
        if isinstance(features_path, str):
            features = joblib.load(features_path)
            X = X[features]
        else:
            X = X[features_path]
            
    
    if elem_prop == 'mat2vec' or elem_prop=='onehot':
        
        X = X.loc[:,X.columns.str.startswith('avg')]
        

    
    return X, y, formulae, skipped

    
def _assign_features(matrices, 
                     elem_info, 
                     formulae, 
                     sum_feat=False):
    
    #we just take matrices retrieved with intial part of generate_features..
    formula_mat, count_mat, frac_mat, elem_mat, target_mat = matrices
    elem_symbols, elem_index, elem_missing = elem_info

    if sum_feat:
        sum_feats = []
    avg_feats = []
    range_feats = []
    # var_feats = []
    dev_feats = []
    max_feats = []
    min_feats = []
    mode_feats = []
    targets = []
    formulas = []
    skipped_formula = []

    #h just used for the progress bar here..
    for h in tqdm.tqdm(range(len(formulae)), desc='Assigning Features...'):
        #['Na','Cl']
        elem_list = formula_mat[h]
        #1
        target = target_mat[h]
        #NaCl
        formula = formulae[h]
        
        #comp_mat has just each row corresponding to features of each element
        #in formula.
        comp_mat = np.zeros(shape=(len(elem_list), elem_mat.shape[-1]))
        skipped = False

        for i, elem in enumerate(elem_list):
            if elem in elem_missing:
                skipped = True
            else:
                row = elem_index[elem_symbols.index(elem)]
                comp_mat[i, :] = elem_mat[row]

        if skipped:
            skipped_formula.append(formula)

        range_feats.append(np.ptp(comp_mat, axis=0))
        # var_feats.append(comp_mat.var(axis=0))
        max_feats.append(comp_mat.max(axis=0))
        min_feats.append(comp_mat.min(axis=0))

        comp_frac_mat = comp_mat.T * frac_mat[h]
        comp_frac_mat = comp_frac_mat.T
        avg_feats.append(comp_frac_mat.sum(axis=0))

        dev = np.abs(comp_mat - comp_frac_mat.sum(axis=0))
        dev = dev.T * frac_mat[h]
        dev = dev.T.sum(axis=0)
        dev_feats.append(dev)

        prominant = np.isclose(frac_mat[h], max(frac_mat[h]))
        mode = comp_mat[prominant].min(axis=0)
        mode_feats.append(mode)

        comp_sum_mat = comp_mat.T * count_mat[h]
        comp_sum_mat = comp_sum_mat.T
        if sum_feat:
            sum_feats.append(comp_sum_mat.sum(axis=0))

        targets.append(target)
        formulas.append(formula)

    if len(skipped_formula) > 0:
        print('\nNOTE: Your data contains formula with exotic elements.',
              'These were skipped.')
    if sum_feat:
        conc_list = [sum_feats, avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)
    
    else:
        conc_list = [avg_feats, dev_feats,
                     range_feats, max_feats, min_feats, mode_feats]
        feats = np.concatenate(conc_list, axis=1)

    return feats, targets, formulas, skipped_formula
    

#UTILITIES FUNCTIONS

class CompositionError(Exception):
    """Exception class for composition errors"""
    pass


def get_sym_dict(f, factor):
    sym_dict = collections.defaultdict(float)
    # compile regex for speedup
    regex = r"([A-Z][a-z]*)\s*([-*\.\d]*)"
    r = re.compile(regex)
    for m in re.finditer(r, f):
        el = m.group(1)
        amt = 1
        if m.group(2).strip() != "":
            amt = float(m.group(2))
        sym_dict[el] += amt * factor
        f = f.replace(m.group(), "", 1)
    if f.strip():
        raise CompositionError(f'{f} is an invalid formula!')
    return sym_dict


def parse_formula(formula):
    '''
    Parameters
    ----------
        formula: str
            A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
    Return
    ----------
        sym_dict: dict
            A dictionary recording the composition of that formula.
    Notes
    ----------
        In the case of Metallofullerene formula (e.g. Y3N@C80),
        the @ mark will be dropped and passed to parser.
    '''
    # for Metallofullerene like "Y3N@C80"
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')
    # compile regex for speedup
    regex = r"\(([^\(\)]+)\)\s*([\.\d]*)"
    r = re.compile(regex)
    m = re.search(r, formula)
    if m:
        factor = 1
        if m.group(2) != "":
            factor = float(m.group(2))
        unit_sym_dict = get_sym_dict(m.group(1), factor)
        expanded_sym = "".join(["{}{}".format(el, amt)
                                for el, amt in unit_sym_dict.items()])
        expanded_formula = formula.replace(m.group(), expanded_sym)
        return parse_formula(expanded_formula)
    sym_dict = get_sym_dict(formula, 1)
    return sym_dict


def _fractional_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    comp_frac = {key: elamt[key] / natoms for key in elamt}
    return comp_frac


def _fractional_composition_L(formula):
    comp_frac = _fractional_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts


def _element_composition(formula):
    elmap = parse_formula(formula)
    elamt = {}
    natoms = 0
    for k, v in elmap.items():
        if abs(v) >= 1e-6:
            elamt[k] = v
            natoms += abs(v)
    return elamt


def _element_composition_L(formula):
    comp_frac = _element_composition(formula)
    atoms = list(comp_frac.keys())
    counts = list(comp_frac.values())
    return atoms, counts
























