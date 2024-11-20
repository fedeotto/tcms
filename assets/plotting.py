#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:39:22 2023
@author: federico
"""
import os
# os.chdir('..')
import plotly.graph_objects as go
import matplotlib as mpl
import plotly.colors as colors
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import pandas as pd
import joblib
# from scripts.loco_cv import generate_compvec, color_map_gap, color_map_sigma
import pickle
from common.utils import clean_formulas
from collections import Counter
from scipy.stats import johnsonsu, betaprime
# from fitter import Fitter
# from analysis.data_analysis import master_list_analysis
import plotly.io as pio
import scipy.stats as st
import numpy as np
from chem_wasserstein.ElM2D_ import ElM2D
from common.chem import _element_composition
import warnings

pio.renderers.default="svg"    # 'svg' or 'browser'
pio.templates.default="plotly_white"

warnings.filterwarnings('ignore')

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

plt.rcParams['font.size'] = 20
plt.rcParams['figure.dpi'] = 700
    
def plot_predictions(dataset_name : str = 'conductivity_room_temp',
                     n_fold : int = 0):
    
    models = {'roost':{'formula':[], 'real' : [], 'pred': [] , 'uncert':[]}, 
              'crabnet': {'formula':[], 'real' : [], 'pred': [] , 'uncert':[]}, 
              'random_forest' : {'formula':[],'real' : [], 'pred': [] , 'uncert':[]}}
    
    for model_name in models.keys():
        path = f'./results/{dataset_name}/regression/{model_name}'
        filelist = [file for file in os.listdir(path) if file.startswith('Kfold')]
        fold_name = f'Kfold_{model_name}_run_0_fold_{n_fold}'
        fold = pd.read_csv(path + '/' + fold_name+'.csv')
        
        mean_uncert = fold['uncert'].mean()
        std_uncert  = fold['uncert'].std()
        
        lower_bound = mean_uncert-4*std_uncert
        upper_bound = mean_uncert+4*std_uncert
        
        fold = fold[(fold['uncert']>lower_bound) & (fold['uncert']<upper_bound)]
        
        models[model_name]['formula'] = fold['composition']
        models[model_name]['real']    = fold['real']
        models[model_name]['pred']    = fold['pred']
        models[model_name]['uncert']  = fold['uncert']
        
    # plotting
    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.dpi'] = 700
    fig, ax = plt.subplots(figsize=(20,6),
                           nrows=1,
                           ncols=3)
    
    fig.subplots_adjust(right=1.0)
    
    for a in ax:
        a.yaxis.set_minor_locator(tck.AutoMinorLocator())
        a.xaxis.set_minor_locator(tck.AutoMinorLocator())
        a.tick_params(direction='in', which='both')

        
    # roost
    ax[0].scatter(models['roost']['real'], 
                models['roost']['pred'], 
                c=models['roost']['uncert'],
                s=10
                )
    
    ax[0].set_ylim(-13.0,7.0)
    
    ax[0].axline((1,1),
                 c='orange',
                 linestyle='--',
                 linewidth=2,
                 slope=1)
    
    ax[1].scatter(models['crabnet']['real'], 
                  models['crabnet']['pred'], 
                  c=models['crabnet']['uncert'],
                  s=10)
    
    ax[1].set_ylim(-13.0,7.0)
    ax[1].axline((1,1),
                 c='orange',
                 linestyle='--',
                 linewidth=2,
                 slope=1)
    
    im = ax[2].scatter(models['random_forest']['real'], 
                       models['random_forest']['pred'], 
                       c=models['random_forest']['uncert'],
                       s=10)
        
    ax[2].set_ylim(-13.0,7.0)
    cb_ax = fig.add_axes([1.01, 0.125, 0.01, 0.750])
    fig.colorbar(im,cax=cb_ax)
    ax[2].axline((1,1),
                 c='orange',
                 linestyle='--',
                 linewidth=2,
                 slope=1)


def plot_distribution_plotly(data: np.array,
                             prop_name: str = 'conductivity'):
    
    fig = go.Figure()

    # # Plot histogram
    fig.add_trace(go.Histogram(x=data,
                               histnorm='density',
                               marker=dict(color='#58c7d5',
                                            opacity=0.5,
                                            line_width=0),
                               name=r'$σ$ \textrm{ distribution}'))

    fig.add_trace(go.Histogram(x=data,
                               histnorm='density',
                               marker=dict(color='#58c7d5',
                                            opacity=0.5,
                                            line_width=0),
                               name=r'$σ$ \textrm{ distribution}'))

    # Plot mean and median lines
    mean = np.mean(data)
    median = np.median(data)
    fig.add_shape(go.layout.Shape(type='line',
                                  x0=mean, x1=mean,
                                  y0=0, y1=0.035,
                                  line=dict(color='black', dash='dash', width=3)))
    fig.add_shape(go.layout.Shape(type='line',
                                  x0=median, x1=median,
                                  y0=0, y1=0.035,
                                  line=dict(color='black', dash='dash', width=3)))

    # Set axis labels
    fig.update_layout(xaxis_title=r'$\textrm{log}_{10}(σ)$',
                      yaxis_title='Count',
                      bargap=0,
                      width=800,
                        height=600,
                      )

    # Calculate and print percentiles
    data_series = pd.Series(data)
    quantiles = data_series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])

    print('Info:')
    print('Mean: ' + str(mean))
    print('Median: ' + str(median))

    for q, value in quantiles.items():
        print(f'{q * 100}th percentile: {value}')

    print('IQR:' + str(quantiles[0.75] - quantiles[0.25]))
    fig.show()
    # Save figure
    # fig.write_image('sigma_dist.png', width=800, height=600, scale=2)


    
def plot_distribution(  data: np.array,
                        prop_name: str = 'conductivity',
                        # dist_name: str = 'asymmetric_laplacian'
                        ):
    
    """
    Plots an asymmetric laplacian distribution to sigma dataset.
    """
    def fit_distribution(data: np.array,
                         prop_name: str = 'conductivity'
                         ):
        
        if prop_name == 'conductivity':
            dist_name = 'johnsonsu'
        else:
            dist_name = 'betaprime'
        
        f = Fitter(data, distributions=[dist_name])
        f.fit()

        return f.get_best()
    
    if prop_name == 'conductivity':
        dist_name = 'johnsonsu'
        dist      = johnsonsu
        label     = 'Johnson SU'
    else:
        dist_name = 'betaprime'
        dist      = betaprime
        label     = 'BetaPrime'
        color_dist= '#31a354'
        color_data= '#fec44f'

    # dict_params = fit_distribution(data, prop_name=prop_name)
    # dist_name   = list(dict_params.keys())[0]
    # a           = dict_params[dist_name]['a']
    # b           = dict_params[dist_name]['b']
    # loc         = dict_params[dist_name]['loc']
    # scale       = dict_params[dist_name]['scale']
    
    fig, ax = plt.subplots(figsize=(10,8))
    mean = np.mean(data)
    median = np.median(data)

    ax.hist(data,
            histtype='step',
            linewidth=4.0,
            color = '#fec44f', #fec44f bandgap '#4f99fe conductivity
            # color=color_data,
            bins=40)
    
    ax.hist(data,
            linewidth=1.0,
            alpha=0.4,
            color = '#fec44f', #fec44f bandgap #4f99fe conductivity
            # color=color_data,
            # label='$\sigma$ distribution',
            label='E${}_g$ distribution',
            bins=40)
        
    ax.axvline(mean, color='k',linewidth=3, ymin=0, ymax= 0.025)
    ax.axvline(median, color='k',linewidth=3, ymin=0, ymax=0.025)

    if prop_name == 'conductivity':
        ax.axvline(3, color='#dd1c77', linestyle='--', label='MMC')
        ax.set_xlabel('log$_{10}$ ($\sigma$)', labelpad=10)
    else:
        ax.set_xlabel('E${}_g$ (eV)', labelpad=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    
    # x = np.linspace(np.min(data), np.max(data), num=5000)
    data = pd.Series(data)
    
    # ax.plot(x, 
    #         dist.pdf(x, a=a, b=b, loc=loc, scale=scale),
    #         color_dist, linestyle='--',lw=3, alpha=0.9, label=label)
    
    # ax.set_ylabel('Frequency', labelpad=10)

    # y_ticklabels = np.arange(0,1600,200)
    # ax.set_yticklabels(y_ticklabels)

    (quant_5, quant_25, 
    quant_50, quant_75, 
    quant_95) = (data.quantile(0.05), data.quantile(0.25), 
                data.quantile(0.5), data.quantile(0.75), data.quantile(0.95))
    
    print('Info:')
    print('Mean: ' + str(mean))
    print('Median: ' + str(median))

    print(f'5th percentile: {quant_5}')
    print(f'25th percentile: {quant_25}')
    print(f'50th percentile: {quant_50}')
    print(f'75th percentile: {quant_75}')

    print('IQR:' + str(quant_75 - quant_25))
                 
    # y_start = 0.15

    # ax.axvline(quant_5, linestyle='--',color='red')
    # ax.axvline(quant_25, linestyle='--',linewidth=1.5,color='#0052bf')
    # ax.axvline(quant_50, linestyle='--',color='red')
    # ax.axvline(quant_75, linestyle='--',linewidth=1.5,color='#0052bf')

    # ax.hlines(y=0.34, xmin=quant_5, xmax=quant_25)
    # ax.hlines(y=0.35, xmin=quant_5, xmax=quant_50)
    # ax.hlines(y=0.36, xmin=quant_5, xmax=quant_75)
    # ax.hlines(y=0.37, xmin=quant_5, xmax=quant_95)
    plt.legend()
    plt.savefig('sigma_dist.png', dpi=600)
    
def plot_interactive_screening(cfg: DictConfig):
    # tcms = pd.read_csv(f'./screening/screen_{which}.csv')
    res = pd.read_excel(f'{cfg.action.screen_result_path}')

    fig = go.Figure()
    # Create a grid for the contour plot
    x_range = np.linspace(res['eg_pred (eV)'].min(), res['eg_pred (eV)'].max(), 100)
    y_range = np.linspace(res['sigma_pred log10 (S/cm)'].min(), res['sigma_pred log10 (S/cm)'].max(), 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Calculate Phi_M for each grid point
    Z = X * Y  # Since conductivity is in log scale, use 10**Y to convert it back

    # Add contour plot with only lines colored
    fig.add_trace(go.Contour(x=x_range,
                             y=y_range,
                             z=Z,
                             colorscale='Viridis',
                             showscale=True,
                             colorbar=dict(title='Phi_M'),
                             contours=dict(coloring='lines', showlines=True),
                             hoverinfo='none'))
    
    fig.add_trace(go.Scatter(x=res['eg_pred (eV)'],
                             y=res['sigma_pred log10 (S/cm)'],
                             customdata=np.stack((res['sigma_pred log10 (S/cm)'],
                                                  res['sigma_uncert log10 (S/cm)'],
                                                  res['eg_pred (eV)'],
                                                  res['eg_uncert (eV)'],
                                                  res['model'],
                                                  res['Phi_M'],
                                                  res['Phi_M_std']
                                                  ), axis=-1),
                             textposition='top right',
                             mode='markers',
                             marker=dict(color=res['Phi_M_std'],
                                         colorscale='ice',
                                         showscale=True,
                                         colorbar=dict(title='Phi_M_std'),
                                         size=5),
                             hovertemplate='Formula: %{text}<br>' +
                                           'Pred. conductivity [S/cm] (log10): %{customdata[0]:.2f}<br>'+
                                           'Unc. conductivity [S/cm] (log10): %{customdata[1]:.2f}<br>'+
                                           'Pred. band gap [eV]: %{customdata[2]:.2f}<br>'+
                                           'Unc. band gap [eV]: %{customdata[3]:.2f}<br>'+
                                           'Model: %{customdata[4]}<br>'+
                                           'Phi_M: %{customdata[5]:.2f}<br>'+
                                           'Phi_M std: %{customdata[6]:.2f}',
                             text=res['formula']))

    fig.update_layout(hoverlabel=dict(bgcolor="white",font_size=13))
    # fig.update_layout(coloraxis_colorbar_title_text="$\sigma$ uncertainty")
    fig.update_xaxes(title='Pred. Eg [eV]')
    fig.update_yaxes(title='Pred. Sigma [S/cm] (log10)')
        
    pio.write_html(fig, file='tcms_candidates.html', auto_open=True)
    fig.write_html('tcms_candidates.html', auto_open=True)
    fig.show()   
        
def plot_elem_hist(dataset_name: str = 'conductivity_room_temp',
                   top_k=5, 
                   color='#2c7fb8'):
    
    df = pd.read_csv(f'./datasets/{dataset_name}.csv')
    
    plt.rcParams['font.size'] = 18
    plt.rcParams['figure.dpi'] = 700
        
    elems_dict = dict.fromkeys(all_symbols, 0)
    train_elems_frequency = []
    
    formulae = df['formula']
    
    formulae_dict = formulae.apply(_element_composition)
    formulae_list = [item for row in formulae_dict for item in row.keys()]
    counter = Counter(formulae_list)
    
    freq_df = pd.DataFrame(None)
    freq_df['elem'] = formulae_list
    freq_df = freq_df.drop_duplicates('elem')
    freq_df['freq'] = [count for count in counter.values()]
    freq_df = freq_df.sort_values('freq', ascending=False)
    freq_df = freq_df.iloc[:top_k]
    
    top_k_elems = freq_df['elem']
    top_k_values = freq_df['freq']
    
    fig, ax = plt.subplots(figsize=(8,6))
        
    ax.bar(top_k_elems, 
           top_k_values,
           width=0.6,
           edgecolor='k',
           color='#43a2ca')
        
    ax.set_title(f'Top {top_k} elemental prevalence')
    ax.set_xlabel('Element', labelpad=10)
    ax.set_ylabel('Frequency', labelpad=10)
    ax.tick_params(which='major',
                   axis='x',
                   right=False,
                   top=False,
                   bottom=True,
                   direction='in',
                   size=10,
                   width=2,
                   length=7)
    
def plot_loco_clusters(dataset_name:str = 'conductivity_room_temp',
                       web=True):
    
    embs_path     = f'./embeddings/{dataset_name}_embeddings.pkl'
    clusters_path = f'./loco_clusters/{dataset_name}_clusters.pkl'
    formulae = pd.read_csv(f'./datasets/{dataset_name}.csv')['formula']
    
    embs     = joblib.load(embs_path)
    clusters = joblib.load(clusters_path)

    colorscale = colors.qualitative.Plotly
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=embs[:,0],
                             y=embs[:,1],
                             mode='markers',
                             marker=dict(color=clusters,
                                         colorscale=colorscale,
                                         showscale=True,
                                         size=10),
                             hovertext=formulae))
    
    fig.update_xaxes(title='UMAP 1')
    fig.update_yaxes(title='UMAP 2')
    fig.update_layout(title=f'{dataset_name}_clusters')
    pio.write_html(fig, file='scatterplot.html', auto_open=True)
    if web: fig.write_html('figure.html', auto_open=True)
    fig.show()
    
def plot_dopant_levels(conduct_df_path : str = './datasets/conductivity_room_temp.csv'):
    conduct_df = pd.read_csv(conduct_df_path)
    gammas = [0.4, 0.3, 0.1, 0.05, 0.03, 0.01]
    intr_dop = {}
    
    for gamma in gammas:
        doptool = DopantsTool(conduct_df, dopant_thresh=gamma)
        intr_dop[f'{gamma}'] =[len(doptool.doped), len(doptool.intr)]
        
    labels = ['0.4', '0.3', '0.1', '0.05','0.03','0.01']
    
    dop_values = [val[0] for val in intr_dop.values()]
    intr_values = [val[1] for val in intr_dop.values()]
    
    x = np.arange(len(labels))
    width = 0.35  
    
    fig, ax = plt.subplots(figsize=(8,6))
    rects1 = ax.bar(x - width/2, 
                    dop_values, 
                    width,
                    color='#a6bddb',
                    edgecolor='k',
                    label='Doped materials')
    
    rects2 = ax.bar(x + width/2, 
                    intr_values, 
                    width,
                    color='#2b8cbe',
                    edgecolor='k',
                    label='Intrinsic materials')
    
    ax.set_xticks(x, labels)
    ax.set_xlabel('Dopant concentration $\gamma$ (%)')
    ax.tick_params(axis='y', direction='in')
    yticks = ax.get_yticks()
    
    for tick in yticks:
        ax.axhline(tick, 
                   linestyle='--', 
                   alpha=0.25,
                   color='gray')
    
    ax.legend()
    plt.rcParams['figure.dpi'] = 700
    plt.rcParams['font.size'] = 14
    
    
def plot_class_target_maes(dataset_name='conductivity_room_temp'):
    path = './results_analysis'
    
    binned_maes_rf      = pd.read_csv(os.path.join(path, f'binned_maes_{dataset_name}_random_forest.csv'))
    binned_maes_crabnet = pd.read_csv(os.path.join(path, f'binned_maes_{dataset_name}_crabnet.csv'))
    binned_maes_roost   = pd.read_csv(os.path.join(path, f'binned_maes_{dataset_name}_roost.csv'))
    
    maes_rf      = binned_maes_rf['mae']
    maes_crabnet = binned_maes_crabnet['mae']
    maes_roost   = binned_maes_roost['mae']
    
    std_rf      = binned_maes_rf['std']
    std_crabnet = binned_maes_crabnet['std']
    std_roost   = binned_maes_roost['std']
    
    labels = list(binned_maes_rf['bins'])
    
    x = np.arange(len(labels))
    width = 0.25  #width of bars.
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    bar1 = ax.bar(x, 
                  maes_rf, 
                  width,
                  color='#7fcdbb',
                  edgecolor='k',
                  yerr=std_rf, 
                  label='Random Forest')
    
    bar2 = ax.bar(x+width,
                  maes_crabnet,
                  width,
                  color='#addd8e',
                  edgecolor='k',
                  yerr=std_crabnet,
                  label='CrabNet'
                  )
    
    bar2 = ax.bar(x+width*2,
                  maes_roost,
                  width,
                  color='#fec44f',
                  edgecolor='k',
                  yerr=std_roost,
                  label='Roost')
    
    ax.set_title('Target-conditional MAE')
    
    if dataset_name=='conductivity_room_temp':
        ax.set_xlabel('Conductivity ranges (log$_{10}$)')
    else:
        ax.set_xlabel('Band gap ranges (eV)')
    
    ax.set_ylabel('MAE')
    ax.xaxis.labelpad = 15
    ax.set_xticks(x+width,labels)
    plt.legend(bbox_to_anchor=(1.6,1.0))
    
    
def plot_error_dist(all_errors,prop, ax):
    ax.hist(all_errors,
            color = 'white',
            edgecolor='k',
            linewidth=2,
            density=True,
            bins=30)
    
    ax.set_xlabel(f'Error {prop}')
    
    # estimate distribution parameters, in this case (a, loc, scale)
    params = st.t.fit(all_errors)
    
    # evaluate PDF
    if prop == '$E_g$ (eV)':
        x = np.linspace(-4, 4, 1000)
    else:
        x = np.linspace(-15, 15, 1000)
        
    pdf = st.t.pdf(x, *params)
    ax.plot(x, pdf, '--r', color='#ffc200', linewidth=4, zorder=3)
    
    # First Confidence Interval
    upper, lower = st.t.interval(0.68, len(all_errors), params[1], params[2])
    ax.axvline(x=lower, color='#009eff', linestyle='--', zorder=1)
    ax.axvline(x=upper, color='#009eff', linestyle='--', zorder=1)
    
    # Second Confidence Interval
    upper, lower = st.t.interval(0.95, len(all_errors), params[1], params[2])
    ax.axvline(x=lower, color='#00b4ff', linestyle='--', zorder=1)
    ax.axvline(x=upper, color='#00b4ff', linestyle='--', zorder=1)
    
    # Third Confidence Interval
    upper, lower = st.t.interval(0.99, len(all_errors), params[1], params[2])
    ax.axvline(x=lower, color='#43c4ef', linestyle='--', zorder=1)
    ax.axvline(x=upper, color='#43c4ef', linestyle='--', zorder=1)
    
    # Mean Line of PDF
    ax.axvline(x=params[1], color='#0066ff', linestyle='--', zorder=1)
    lower, upper = st.t.interval(0.67, len(all_errors), params[1], params[2])
    
    print(f'--- T-Student mean: {params[1]}')
    print(f'--- 68% Confidence interval: [{lower}, {upper}]')
    

def plot_act_pred(act, pred, prop, diag, ax, colors=None):
    ax.scatter(act,
                pred,
                alpha=0.6,
                s=5,
                c=colors)
    if diag:
        ax.axline((1,1),
                  c='red',
                  linestyle='--',
                  linewidth=2.5,
                  slope=1)        
    
    
def plot_control_studies_info(dataset_name  ='conductivity_room_temp'):
    if dataset_name == 'conductivity_room_temp':
        prop = '$\sigma$ log$_{10}$'
    else:
        prop = '$E_g$ (eV)'
    
    shuffled_control = pd.read_csv(f'./results_analysis/mean_shuffled_control_{dataset_name}.csv')
    mean_control     = pd.read_csv(f'./results_analysis/dummy_model_{dataset_name}.csv')
    
    mean_control['error']     = mean_control['real']     - mean_control['pred']
    shuffled_control['error'] = shuffled_control['real'] - shuffled_control['pred']
    
    fig, ax = plt.subplots(figsize=(14,12), nrows=2, ncols=2)
    
    #Shuffled control study
    plot_act_pred(shuffled_control['real'], 
                  shuffled_control['pred'], 
                  prop=prop, 
                  diag=True,
                  ax=ax[0][0])
    
    ax[0][0].set_xlabel(f'Actual {prop}')
    ax[0][0].set_ylabel(f'Predicted {prop}')
    ax[0][0].set_title('Mean shuffled control study')
    
    plot_error_dist(shuffled_control['error'],prop=prop,ax=ax[1][0])    
    
    # Dummy model
    plot_act_pred(mean_control['real'], 
                  mean_control['pred'],
                  prop=prop, 
                  diag=False,
                  ax=ax[0][1])
    
    ax[0][1].set_xlabel(f'Actual {prop}')
    ax[0][1].set_title('Dummy model')
    
    plot_error_dist(mean_control['error'],
                    prop=prop,
                    ax=ax[1][1])
    
    ax[1][1].set_xlabel('Error {prop}')
    
def plot_embs_screening(filename: str = 'embs_master_list_50000_samples.pkl',
                        embs_path = None,
                        create_html=True):
        
    with open(f'./results_analysis/{filename}', 'rb') as handle:
        embs = pickle.load(handle)
        
    fig, ax = plt.subplots(figsize=(14,8))
                      
    scatter = ax.scatter(embs['PCA_1'], 
                         embs['PCA_2'],
                         s=1,
                         marker='o',
                         cmap='viridis',
                         c=embs['crabnet_sigma_pred'])
    
    ax.set_xlabel('PCA_1', labelpad=10)
    ax.set_ylabel('PCA_2')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Predicted $\sigma$ (S/cm) log10', labelpad=20)
    
    if create_html:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=embs['PCA_1'],
                                 y=embs['PCA_2'],
                                 textposition='top right',
                                 mode='markers',
                                 marker=dict(color=embs['crabnet_sigma_pred'],
                                             colorscale='Viridis',
                                             showscale=True,
                                             size=4),
                                 hovertemplate='Formula: %{text}<br>' +
                                               'Predicted Conductivity [S/cm] (log10): %{customdata:.2f}',
                                               customdata=embs['crabnet_sigma_pred'],
                                               text=embs['formula']))
        
        fig.update_layout(
            coloraxis_colorbar=dict(title="NON POSSO FARLO")
        )
                
        fig.update_xaxes(title='PCA_1')
        fig.update_yaxes(title='PCA_2')
        pio.write_html(fig, file='screening_result.html', auto_open=True)
        fig.show()
    
def plot_models_loco_info(dataset_name  ='conductivity_room_temp'):
    crab_path = f'./eval_results/{dataset_name}/regression/crabnet'
    rf_path   = f'./eval_results/{dataset_name}/regression/random_forest'
    dop_path  = f'./eval_results/{dataset_name}/regression/dopnet'
    
    if dataset_name=='conductivity':
        prop='$\sigma$ log$_{10}$'
    else:
        prop='$E_g$ (eV)'
    
    result_crab = pd.DataFrame(data=None, columns=['composition','real','pred','cluster'])
    for fold in range(5):
        name = f'LOCO_crabnet_finetune_run_0_fold_{fold}.csv'
        result_fold = pd.read_csv(os.path.join(crab_path,name))
        result_fold = result_fold.drop('uncert',axis=1)
        result_fold['cluster'] = fold
        result_crab= pd.concat([result_crab,result_fold],axis=0)

    result_rf = pd.DataFrame(data=None, columns=['composition','real','pred','cluster'])
    for fold in range(5):
        name = f'LOCO_random_forest_run_0_fold_{fold}.csv'
        result_fold = pd.read_csv(os.path.join(rf_path,name))
        result_fold = result_fold.drop('uncert',axis=1)
        result_fold['cluster'] = fold
        result_rf = pd.concat([result_rf,result_fold],axis=0)

    result_dop = pd.DataFrame(data=None, columns=['composition','real','pred','cluster'])
    for fold in range(5):
        name = f'LOCO_dopnet_run_0_fold_{fold}.csv'
        result_fold = pd.read_csv(os.path.join(dop_path,name))
        result_fold = result_fold.drop('uncert',axis=1)
        result_fold['cluster'] = fold
        result_dop = pd.concat([result_dop,result_fold],axis=0)
            
    result_crab = result_crab.reset_index(drop=True)
    result_rfv  = result_rf.reset_index(drop=True)
    result_dop  = result_dop.reset_index(drop=True)

    result_crab['error'] = result_crab['real'] - result_crab['pred']
    result_rf['error']   = result_rf['real']   - result_rf['pred']
    result_dop['error']  = result_dop['real']   - result_dop['pred']

    if dataset_name=='conductivity':
        color_map = color_map_sigma
    else:
        color_map = color_map_gap

    colors  = np.array([color_map[label] for label in result_crab['cluster'].values])
    fig, ax = plt.subplots(figsize=(21,16), nrows=2, ncols=3)

    # CRABNET LOCO
    plot_act_pred(result_crab['real'],
                  result_crab['pred'], 
                  prop,
                  colors =colors,
                  diag=True, 
                  ax=ax[0][0])
    
    ax[0][0].set_xlabel(f'Actual {prop}', labelpad=8)
    ax[0][0].set_ylabel(f'Predicted {prop}')    
    ax[0][0].set_title('CrabNet (LOCO)')
    
    if prop=='$\sigma$ log$_{10}$':
        xticks = np.arange(-20,10,5)
        yticks = np.arange(-15,7,2.5)
        ax[0][0].set_xticks(xticks)
        ax[0][0].set_yticks(yticks)
        ax[0][0].set_xticklabels(xticks)
        ax[0][0].set_yticklabels(yticks)

    print('--- Distribution errors CrabNet (LOCO):')
    plot_error_dist(result_crab['error'], 
                    prop,
                    ax=ax[1][0])
    
    if prop=='$\sigma$ log$_{10}$':
        xticks = np.arange(-15,15,5)
        ax[1][0].set_xticks(xticks)
        ax[1][0].set_xticklabels(xticks)

    # RF LOCO
    plot_act_pred(result_rf['real'],
                  result_rf['pred'],
                  prop, 
                  colors=colors,
                  diag=True, 
                  ax=ax[0][1])
    
    ax[0][1].set_title(f'RF (LOCO)')
    ax[0][1].set_xlabel(f'Actual {prop}', labelpad=8)
    
    if prop == '$\sigma$ log$_{10}$':
        xticks = np.arange(-20,10,5)
        yticks = np.arange(-15,7,2.5)
        ax[0][1].set_xticks(xticks)
        ax[0][1].set_yticks(yticks)
        ax[0][1].set_xticklabels(xticks)
        ax[0][1].set_yticklabels(yticks)
    
    print(f'--- Distribution errors RF:')
    plot_error_dist(result_rf['error'], 
                    prop,
                    ax=ax[1][1])

    if prop == '$\sigma$ log$_{10}$':
        xticks = np.arange(-15,15,5)
        ax[1][1].set_xticks(xticks)
        ax[1][1].set_xticklabels(xticks)

    plot_act_pred(result_dop['real'],
                  result_dop['pred'],
                  prop,
                  colors=colors,
                  diag=True, 
                  ax=ax[0][2])
    
    ax[0][2].set_title(f'DopNet (LOCO)')
    ax[0][2].set_xlabel(f'Actual {prop}')

    if prop == '$\sigma$ log$_{10}$':
        xticks = np.arange(-20,10,5)
        yticks = np.arange(-15,7,2.5)
        ax[0][2].set_xticks(xticks)
        ax[0][2].set_yticks(yticks)
        ax[0][2].set_xticklabels(xticks)
        ax[0][2].set_yticklabels(yticks)
        
    print(f'--- Distribution errors DopNet:')
    plot_error_dist(result_dop['error'], 
                    prop,
                    ax=ax[1][2])
    
    if prop== '$\sigma$ log$_{10}$':
        ax[1][2].set_xticks(np.arange(-15,15,5))
    
if __name__ == '__main__':
    df = pd.read_excel('datasets/bandgap.xlsx')
    # data = np.log10(df['Sigma (S/cm)']).values
    data = df['Eg (eV)'].values

    plot_distribution(data, prop_name='bandgap')