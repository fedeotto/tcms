import pandas as pd
from matminer.datasets import load_dataset

# Processing conductivity
print("\n📊 Processing conductivity data:")
sigma = pd.read_excel('data/te_expt.xlsx')

sigma = sigma[['Formula', 'T (K)', 'Conduct. (S/cm)']]

sigma = sigma[sigma['T (K)'] == 300]

sigma = sigma.drop(columns=['T (K)'],axis=1)

std    = sigma.groupby('Formula').transform('std')
sigma['std']    = std
sigma['std'] = sigma['std'].fillna(0)

mask = sigma['std'] <10
sigma = sigma[mask]

sigma = sigma.drop_duplicates(subset=['Formula'])
sigma =sigma.drop(columns=['std'],axis=1)

sigma.rename(columns={'Formula': 'formula', 'Conduct. (S/cm)': 'Sigma (S/cm)'}, inplace=True)

sigma.to_excel('data/conductivity.xlsx', index=False)
print(f"     ✅ Saved {len(sigma)} conductivity records")

# Processing band gap
print("\n📊 Processing band gap data:")
bandgap = load_dataset('matbench_expt_gap')

std = bandgap.groupby('composition').transform('std')
bandgap['std'] = std
bandgap['std'] = bandgap['std'].fillna(0)
mask = bandgap['std'] < 0.1
bandgap = bandgap[mask]

bandgap = bandgap.drop_duplicates(subset=['composition'])

bandgap.rename(columns={'composition': 'formula', 'gap_expt': 'Eg (eV)'}, inplace=True)
bandgap.to_excel('data/bandgap.xlsx', index=False)
print(f"     ✅ Saved {len(bandgap)} band gap records")


