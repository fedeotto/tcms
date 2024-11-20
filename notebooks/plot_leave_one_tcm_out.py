import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors
import matplotlib
import warnings

import plotly.graph_objects as go
import plotly.io as pio

warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 500
plt.rcParams['font.size'] = 18

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="simple_white"

result_names  = os.listdir(f'./LOTCMO')
ind_dops      = ['Ga', 'In','Mn','Sb','Ta', 'Ti','W']
zn_dops       = ['Al-Sn', 'Al', 'Ga']
model         = 'crab' #rf or dop

def plot_lotcmo():
    fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(9,8))
    names  = os.listdir(f'./LOTCMO')
    names  = [name for name in names if name !='table_lotcmo.xlsx']

    dfs_dict = {'SnO2':{}, 'ZnO':{}}
    for family in dfs_dict.keys():
        if family == 'SnO2':
            dops = ind_dops
        else:
            dops = zn_dops
        for dop in dops:
            dfs_dict[family][dop] = {'sigma':{}, 'gap':{}}
    
    for key in dfs_dict.keys():
        for dop in dfs_dict[key].keys():
            for prop in dfs_dict[key][dop].keys():
                raw_data      = pd.read_excel('./LOTCMO/'+f'tcms_{prop}_{key}_{dop}.xlsx')
                formulas      = raw_data['Unnamed: 0'].loc[2:].values
                preds_crab    = raw_data['crabnet'].loc[2:].values
                uncert_crab   = raw_data['Unnamed: 2'].loc[2:].values
                preds_rf      = raw_data['rf'].loc[2:].values
                uncert_rf     = raw_data['Unnamed: 4'].loc[2:].values
                preds_dop     = raw_data['dopnet'].loc[2:].values
                uncert_dop    = raw_data['Unnamed: 6'].loc[2:].values

                if prop == 'sigma':
                    real          = raw_data['real_sigma log10 (S/cm)'].loc[2:].values
                else:
                    real          = raw_data['real_eg (eV)'].loc[2:].values
                
                df = pd.DataFrame({'formula':formulas,
                                    'preds_crab':preds_crab,
                                    'uncert_crab':uncert_crab,
                                    'preds_rf':preds_rf,
                                    'uncert_rf':uncert_rf,
                                    'preds_dop':preds_dop,
                                    'uncert_dop':uncert_dop,
                                    'real':real})
                
                dfs_dict[key][dop][prop] = df

    # # Get the Set3 colormap
    cmap = get_cmap("Paired")
    # Define the number of colors you need
    num_colors = 10

    # Generate a list of colors from the colormap
    colors_rgb = [cmap(i) for i in range(num_colors)]
    colors     = [mcolors.to_hex(color) for color in colors_rgb]

    labels_set = set()
    count      = 0 # to move colors
    for family in dfs_dict.keys():
        for dop in dfs_dict[family].keys():
            color = colors[count]
            df_sigma = dfs_dict[family][dop]['sigma']
            df_gap   = dfs_dict[family][dop]['gap']

            common_formulas = set(df_sigma['formula'].values).intersection(set(df_gap['formula'].values))
            df_sigma = df_sigma[df_sigma['formula'].isin(common_formulas)]
            df_gap   = df_gap[df_gap['formula'].isin(common_formulas)]

            df_sigma = df_sigma.sort_values(by='formula')
            df_gap   = df_gap.sort_values(by='formula')

            label    = f'{family}:{dop}'
            ax.scatter(df_gap[f'preds_{model}'], 
                        10**df_sigma[f'preds_{model}'], 
                        c=color,
                        label=label if label not in labels_set else None)
            ax.set_yscale('log')
            labels_set.add(label)
            count += 1
    ax.axhline(y=10**2, color='red', linestyle='--')
    ax.axvline(x=3, color='red', linestyle='--')
    ax.grid()

    ax.set_xlabel('Predicted E$_g$ (eV)', labelpad=10)
    ax.set_ylabel('Predicted $\sigma$ (S/cm)', labelpad=10)
    plt.legend(loc='upper center', 
                bbox_to_anchor=(0.5, 1.17),
                ncol=4,
                prop={'weight': 'bold', 'size':12},fancybox=True)
    # bbox_to_anchor=(1.0, 0.9),

def plot_latcmo(ax):
    names  = os.listdir(f'./LATCMO')
    names  = [name for name in names if name !='table_latcmo.xlsx']

    dfs_dict = {'SnO2':{}, 'ZnO':{}}
    for family in dfs_dict.keys():
        if family == 'SnO2':
            dops = ind_dops
        else:
            dops = zn_dops
        for dop in dops:
            dfs_dict[family][dop] = {'sigma':{}, 'gap':{}}

    for key in dfs_dict.keys():
        for dop in dfs_dict[key].keys():
            for prop in dfs_dict[key][dop].keys():
                raw_data      = pd.read_excel('./LATCMO/'+f'tcms_{prop}_{key}.xlsx')
                formulas      = raw_data['Unnamed: 0'].loc[2:].values
                preds_crab    = raw_data['crabnet'].loc[2:].values
                uncert_crab   = raw_data['Unnamed: 2'].loc[2:].values
                preds_rf      = raw_data['rf'].loc[2:].values
                uncert_rf     = raw_data['Unnamed: 4'].loc[2:].values
                preds_dop     = raw_data['dopnet'].loc[2:].values
                uncert_dop    = raw_data['Unnamed: 6'].loc[2:].values
    
                if prop == 'sigma':
                    real          = raw_data['real_sigma log10 (S/cm)'].loc[2:].values
                else:
                    real          = raw_data['real_eg (eV)'].loc[2:].values
                
                df = pd.DataFrame({'formula':formulas,
                                    'preds_crab':preds_crab,
                                    'uncert_crab':uncert_crab,
                                    'preds_rf':preds_rf,
                                    'uncert_rf':uncert_rf,
                                    'preds_dop':preds_dop,
                                    'uncert_dop':uncert_dop,
                                    'real':real})
                
                if key == 'ZnO' and dop == 'Al':
                    mask     = df['formula'].str.contains('Al') & ~df['formula'].str.contains('Sn')
                    df_dop   = df[mask]
                elif key == 'ZnO' and dop == 'Al-Sn':
                    mask     = df['formula'].str.contains('Al') & df['formula'].str.contains('Sn')
                    df_dop   = df[mask]
                else:
                    df_dop = df[df['formula'].str.contains(dop)]

                dfs_dict[key][dop][prop] = df_dop
    
    # # Get the Set3 colormap
    cmap = get_cmap("Paired")
    # Define the number of colors you need
    num_colors = 10

    # Generate a list of colors from the colormap
    colors_rgb = [cmap(i) for i in range(num_colors)]
    colors     = [mcolors.to_hex(color) for color in colors_rgb]

    labels_set = set()
    count      = 0 # to move colors
    for family in dfs_dict.keys():
        for dop in dfs_dict[family].keys():
            color = colors[count]
            df_sigma = dfs_dict[family][dop]['sigma']
            df_gap   = dfs_dict[family][dop]['gap']

            common_formulas = set(df_sigma['formula'].values).intersection(set(df_gap['formula'].values))
            df_sigma = df_sigma[df_sigma['formula'].isin(common_formulas)]
            df_gap   = df_gap[df_gap['formula'].isin(common_formulas)]

            df_sigma = df_sigma.sort_values(by='formula')
            df_gap   = df_gap.sort_values(by='formula')

            label    = f'{family}:{dop}'
            ax.scatter(df_gap[f'preds_{model}'], 
                        10**df_sigma[f'preds_{model}'], 
                        c=color,
                        label=label if label not in labels_set else None)
            ax.set_yscale('log')
            labels_set.add(label)
            count += 1
    ax.axhline(y=10**2, color='red', linestyle='--')
    ax.axvline(x=3, color='red', linestyle='--')
    ax.grid()

    ax.set_xlabel('Predicted E$_g$ (eV)', labelpad=10)


def plot_lotcmo_plotly():
    names  = os.listdir(f'./LOTCMO')
    names  = [name for name in names if name !='table_lotcmo.xlsx']

    dfs_dict = {'SnO2':{}, 'ZnO':{}}
    for family in dfs_dict.keys():
        if family == 'SnO2':
            dops = ind_dops
        else:
            dops = zn_dops
        for dop in dops:
            dfs_dict[family][dop] = {'sigma':{}, 'gap':{}}
    
    for key in dfs_dict.keys():
        for dop in dfs_dict[key].keys():
            for prop in dfs_dict[key][dop].keys():
                raw_data      = pd.read_excel('./LOTCMO/'+f'tcms_{prop}_{key}_{dop}.xlsx')
                formulas      = raw_data['Unnamed: 0'].loc[2:].values
                preds_crab    = raw_data['crabnet'].loc[2:].values
                uncert_crab   = raw_data['Unnamed: 2'].loc[2:].values
                preds_rf      = raw_data['rf'].loc[2:].values
                uncert_rf     = raw_data['Unnamed: 4'].loc[2:].values
                preds_dop     = raw_data['dopnet'].loc[2:].values
                uncert_dop    = raw_data['Unnamed: 6'].loc[2:].values

                if prop == 'sigma':
                    real          = raw_data['real_sigma log10 (S/cm)'].loc[2:].values
                else:
                    real          = raw_data['real_eg (eV)'].loc[2:].values
                
                df = pd.DataFrame({'formula':formulas,
                                    'preds_crab':preds_crab,
                                    'uncert_crab':uncert_crab,
                                    'preds_rf':preds_rf,
                                    'uncert_rf':uncert_rf,
                                    'preds_dop':preds_dop,
                                    'uncert_dop':uncert_dop,
                                    'real':real})
                
                dfs_dict[key][dop][prop] = df

    # # Get the Set3 colormap
    # Define your own custom colors
    custom_colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928', '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f']

    fig = go.Figure()

    labels_set = set()
    for family in dfs_dict.keys():
        for dop in dfs_dict[family].keys():
            color = custom_colors[len(labels_set) % len(custom_colors)]
            df_sigma = dfs_dict[family][dop]['sigma']
            df_gap = dfs_dict[family][dop]['gap']

            common_formulas = set(df_sigma['formula'].values).intersection(set(df_gap['formula'].values))
            df_sigma = df_sigma[df_sigma['formula'].isin(common_formulas)]
            df_gap = df_gap[df_gap['formula'].isin(common_formulas)]

            df_sigma = df_sigma.sort_values(by='formula')
            df_gap = df_gap.sort_values(by='formula')

            label = f'{family}:{dop}'
            fig.add_trace(go.Scatter(
                x=df_gap[f'preds_{model}'],
                y=10**df_sigma[f'preds_{model}'],
                mode='markers',
                marker=dict(color=color,symbol='circle'),
                name=label if label not in labels_set else None
            ))
            labels_set.add(label)

    # Add horizontal and vertical lines
    fig.add_vline(x=3, line_width=2, line_dash="dash", line_color="red",opacity=0.7)
    fig.add_hline(y=100, line_width=2, line_dash="dash", line_color="red",opacity=0.7)

    fig.update_yaxes(type="log")
    # Update layout
    fig.update_layout(
        width=700,
        height=600,
        font=dict(
            size=20,         # Set the overall font size
            ),
        xaxis=dict(title=r'$\textrm{Predicted } E_g \textrm{ (eV)}$',
                   showgrid=True),
        yaxis=dict(title=r'$\textrm{Predicted } \sigma \textrm{ (S/cm)}$', 
                   showgrid=True,
                   type='log',
                    tickvals=[1, 10, 100, 1000, 10000],
                    ticktext=['$10^0$','$10^1$', '$10^2$', '$10^3$', '$10^4$'],
                    tickmode='array'),

        legend=dict(orientation="h",
            yanchor="bottom",
            y=1.06,
            font=dict(size=15),
            xanchor="right",
            x=0.95)
        )
    fig.write_image("./figures/lotcmo_plotly.png", width=700, height=600, scale=2)
    fig.show()

def plot_all():
    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(18,8))
    plot_latcmo(ax=ax[0])  # Pass the axis to the plot_latcmo function
    plot_lotcmo(ax=ax[1])  # Pass the axis to the plot_lotcmo function
    plt.legend(loc='upper center', bbox_to_anchor=(-0.15, 1.17), fancybox=True, ncol=5)

if __name__ == "__main__":
    plot_lotcmo_plotly()