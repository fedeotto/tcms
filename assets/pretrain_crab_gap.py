#IMPORTS
import pandas as pd
from mp_api.client import MPRester
import numpy as np
import os
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from assets.CrabNet.kingcrab import CrabNet
from common.utils import preprocess_data_ml
import seaborn as sns
from assets.CrabNet.model import Model
import torch
import random

plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

random_seed = 1234

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

# #Querying band gaps of all materials in the MP
# with MPRester("qy6DSKfr2ZjbCDIhOXgwgaacAqLK75BI") as mpr:
#     docs = mpr.materials.summary.search(band_gap=(0,5000),
#                                         fields=["material_id", "formula_pretty", "band_gap"])

# # Sort the results by material ID to ensure consistency
# docs      = sorted(docs, key=lambda x: x.material_id)

# # #Converting the data to a pandas dataframe
# formulas  = [doc.formula_pretty for doc in docs]
# band_gaps = [doc.band_gap for doc in docs]
# mp_id     = [doc.material_id for doc in docs]

# mp_band_gaps = pd.DataFrame({"mp_id": mp_id, "formula": formulas, "Eg (eV)": band_gaps})
# mp_band_gaps.to_csv('datasets/mp_band_gaps.csv', index=False)

# Simple function to get the reduced formula
def get_reduced_formula(formula):
    try:
        comp = Composition(formula)
        return comp.reduced_formula
    except:
        return np.nan

#LOADING MP BAND GAPS and EXP. BAND GAPS
mp_band_gaps  = pd.read_csv('datasets/mp_band_gaps.csv')
mp_band_gaps  = mp_band_gaps[['formula','Eg (eV)']]
exp_band_gaps = pd.read_excel('datasets/bandgap.xlsx') #experimental Eg

#Applying pymatgen Composition to formulas to handle permutation invariance
mp_band_gaps['composition'] = mp_band_gaps['formula'].apply(get_reduced_formula)
exp_band_gaps['composition'] = exp_band_gaps['formula'].apply(get_reduced_formula)

mp_band_gaps  = mp_band_gaps.dropna().reset_index(drop=True)
exp_band_gaps = exp_band_gaps.dropna().reset_index(drop=True)

#Checking common compositions between the two datasets
common_compositions = set(mp_band_gaps['composition']).intersection(set(exp_band_gaps['composition']))

print('Size of MP band gaps dataset:', mp_band_gaps.shape[0])
print('Size of experimental band gaps dataset:', exp_band_gaps.shape[0])
print('Number of common compositions:', len(common_compositions))

#Removing compositions from MP data that are in the experimental data
mp_band_gaps = mp_band_gaps[~mp_band_gaps['composition'].isin(common_compositions)].reset_index(drop=True)

#quick double-check
assert set(mp_band_gaps['composition']).intersection(set(exp_band_gaps['composition'])) == set()

#removing the composition column (not needed anymore)
mp_band_gaps   = mp_band_gaps.drop('composition',axis=1)
expt_band_gaps = exp_band_gaps.drop('composition',axis=1)
print('Size of MP band gaps dataset after removing common compositions:', mp_band_gaps.shape[0])

#Preprocessing duplicates
# mp_band_gaps = mp_band_gaps[['formula','Eg (eV)']]
# median       = mp_band_gaps.groupby(['formula']).transform('median')
# std          = mp_band_gaps.groupby(['formula']).transform('std')
# std          = std.fillna(0)
# #
# mp_band_gaps['median'] = median
# mp_band_gaps['std']    = std

# mask = mp_band_gaps['std']<=0.1 #processing in a similar way as matbench_expt_gap
# mp_band_gaps = mp_band_gaps[mask].drop(['median','std'], axis=1).reset_index(drop=True)

# #Dropping remaining duplicates
# mp_band_gaps = mp_band_gaps.sort_values(by='Eg (eV)').drop_duplicates(subset='formula', keep='first').reset_index(drop=True)
mp_band_gaps.head()
mp_band_gaps.rename(columns={'Eg (eV)':'target'}, inplace=True)

#Pretraining CrabNet model on MP data
model = Model(CrabNet(compute_device=device).to(device))

#10% validation data
val_data     = mp_band_gaps.sample(frac=0.1, random_state=random_seed)
mp_band_gaps = mp_band_gaps.drop(val_data.index).reset_index(drop=True)

#loading data
model.load_data(mp_band_gaps, train=True, batch_size=512)
model.load_data(val_data, train=False, batch_size=512)

#Training done in a separate script due to time.
model.fit(epochs=300)
model.save_network('transfer_models/transfer_crab_bandgap_mp_unprocessed.pt')
