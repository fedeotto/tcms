

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error as mae, \
                            r2_score as r2, \
                            accuracy_score as acc, \
                            matthews_corrcoef as mcc
import os
from common.chem import _element_composition_L
import pandas as pd

device = torch.device('cuda')

def get_models_results(dataset_name:str = 'conductivity',
                       model_name :str  = 'random_forest',
                       cv               ='Kfold'):        
    
    path     = f'eval_results/{dataset_name}/{model_name}/'
    filelist = [f for f in os.listdir(path) if f.startswith(f'{cv}')]
    
    maes, r2s = [], []
    for file in filelist:
        res_df = pd.read_excel(os.path.join(path,file))        
        mean_abs_error = mae(res_df['real'], res_df['pred'])
        r2_score = r2(res_df['real'], res_df['pred'])
        maes.append(mean_abs_error)
        r2s.append(r2_score)
    
    mean_mae = np.mean(np.array(maes))
    mean_r2 = np.mean(np.array(r2s))
    std_mae = np.std(np.array(maes))
    std_r2 = np.std(np.array(r2s))
    print(f"MAE: {mean_mae} +/- {std_mae}")
    print(f"R2: {mean_r2} +/- {std_r2}") 

def preprocess_data_ml(df_,
                       dataset_name: str = 'conductivity',
                       elem_prop:    str = 'magpie',
                       screening: bool   = False,
                       shuffle  : bool   = True):
    
    """
    Utility function which prepares datasets to be fed to ML models.
    """

    df = df_.copy()

    if not screening:
        df = df.rename(columns={df.columns[1]:'formula',
                                df.columns[2]: 'target'})
        df = df.drop('Entry',axis=1)
        if dataset_name == 'conductivity':
            df['target'] = np.log10(df['target'])

    if shuffle:
        df = df.sample(frac=1, random_state=1234)

    idxs_to_drop = []

    if elem_prop != 'mendeleev':
        elem_props = pd.read_csv(f'assets/element_properties/{elem_prop}.csv', index_col='element')
        
        valid_elems= list(elem_props.index)

        for i,f in enumerate(df['formula']):
            elems, _ = _element_composition_L(f)
            for el in elems:
                if el not in valid_elems:
                    idxs_to_drop.append(i)
                    break
    else:
        atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
            'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
            'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
            'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
            'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}
        
        valid_elems = list(atom_nums.keys())
        dop_count = 0
        for i, f in enumerate(df['formula']):
            elems, atms = _element_composition_L(f)
            for el, frac in zip(elems,atms):
                if frac <0.1:
                    dop_count+=1
                if dop_count > 7 or el not in atom_nums.keys(): #7 is max dopants.
                    idxs_to_drop.append(i)
                    dop_count=0
                    break
    
    df = df.drop(index=idxs_to_drop)

    if not screening:
        #because we need the indexes for entries!
        df = df.reset_index(drop=True)
    
    return df

def clean_noble_gases(df): 
    noble_gases  = ['Kr','Ne','Xe','He','Ar','Og','Rn' ]
    idxs_to_drop = []
    
    for i,f in enumerate(df['formula']):
        try:
            elems, _ = _element_composition_L(f)
            for el in elems:
                if el in noble_gases:
                    idxs_to_drop.append(i)
                    break
        except:
            idxs_to_drop.append(i)
    df = df.drop(index=idxs_to_drop).reset_index(drop=True)
    
    return df

def clean_pure_elements(df): 
    idxs_to_drop = []
    for i,f in enumerate(df['formula']):
        try:
            elems, _ = _element_composition_L(f)
            if len(elems)==1:
                idxs_to_drop.append(i)
        except:
            idxs_to_drop.append(i)
    df = df.drop(index=idxs_to_drop).reset_index(drop=True)
    
    return df
    
def clean_outliers(df_, mult=4):
    # assumes last column is target.
    #Note:
    df     = df_.copy()
    target = df.columns[-1]
    std = np.std(np.log10(df[target]+1e-8))
    mean = np.mean(np.log10(df[target]+1e-8))
    output = df[np.log10(df[target]+1e-8)<=(mean+mult*std)]# drop above mean + x*std
    output = output[np.log10(output[target]+1e-8)>=(mean-mult*std)]  # drop below mean - x*std
    return output

def clean_formulas(df: pd.DataFrame):
    df_copy = df.copy()

    df_copy['formula'] = df_copy['formula'].str.split(' ',n=1).str[0]
    df_copy['formula'] = df_copy['formula'].str.split(',',n=1).str[0]
    # df_copy['formula'] = df_copy['formula'].map(lambda x: x.lstrip('[').rstrip(']'))
    # df_copy['formula'] = df_copy['formula'].str.replace('[','',regex=False)
    # df_copy['formula'] = df_copy['formula'].str.replace(']','',regex=False)
    df_copy['formula'] = df_copy['formula'].str.replace('rt','',regex=False)
    df_copy['formula'] = df_copy['formula'].str.split('+').str[0]
    # df_copy['formula'] = df_copy['formula'].str.split('(').str[0]
    df_copy['formula'] = df_copy['formula'].str.split('///').str[0]
    df_copy['formula'] = df_copy['formula'].str.split('///').str[0]
    df_copy['formula'] = df_copy['formula'].str.split(' ').str[0]
    df_copy = df_copy[~df_copy['formula'].str.contains('x')]
    mask = df_copy['formula'] == ''
    df_copy = df_copy[~mask]
    
    df_copy = df_copy.reset_index(drop=True)
    
    idxs_to_drop = []
    
    #Cleaning from D
    for i, formula in enumerate(df_copy['formula']):
        if 'D' in formula:
            try:
                if formula[formula.index('D') + 1] !='y':
                    idxs_to_drop.append(i)
            except:
                idxs_to_drop.append(i)

    
    #Cleaning from G
    for i, formula in enumerate(df_copy['formula']):
        if 'G' in formula:
            try:
                if (formula[formula.index('G') + 1] not in ['a','e','d']) \
                    or (formula=='GaAs0.1P0.9G1128'):
                    idxs_to_drop.append(i)
            except:
                idxs_to_drop.append(i)
                
    idxs_to_drop = list(set(idxs_to_drop))
    df_copy = df_copy.drop(index=idxs_to_drop)
    df_copy = df_copy.reset_index(drop=True)
    return df_copy

if __name__ == '__main__':
    get_models_results(dataset_name='bandgap',
                        model_name='rf',
                        cv='LOCO')
        
# def dopnet_preprocessing(df_: pd.DataFrame):
#     #elements on which Dopnet has been trained
#     atom_nums = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O':8, 'F': 9, 'Ne': 10,
#         'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
#         'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
#         'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
#         'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
#         'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
#         'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
#         'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
#         'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
#         'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100}

#     df = df_.copy()
#     idxs_to_drop = []
#     counter = 0
#     for i, f in enumerate(df['formula']):
#         elems, atms = _element_composition_L(f)
#         for el, frac in zip(elems,atms):
#             if frac <0.1:
#                 counter+=1
#             if counter > 7 or el not in atom_nums.keys():
#                 idxs_to_drop.append(i)
#                 break
    
#     df.drop(idxs_to_drop, inplace=True)

#     return df