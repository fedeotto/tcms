"""

Script to create LOCO-CV splits according to the method proposed in the following paper:
https://pubs.rsc.org/en/content/articlelanding/2022/dd/d2dd00039c

"""
import pandas as pd
import matplotlib.pyplot as plt
from assets.cbfv.composition import generate_features
from sklearn.kernel_approximation import RBFSampler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import warnings

plt.rcParams['figure.dpi'] = 700
plt.rcParams['font.size']  = 16

warnings.filterwarnings('ignore')

color_map_sigma = {
                        0: '#756bb1',
                        1: '#c51b8a',
                        2: '#e6550d',
                        3: '#43a2ca',
                        4: '#a8ddb5',
                    }

color_map_gap = {
                    0:'#f71010',
                    1:'#ffb52b',
                    2:'#27c1bd',
                    3:'#32b60f',
                    4:'#2b72ff' 
                }

def apply_loco_cv(df,
                  dataset_name : str = 'conductivity',
                  n_clusters :int = 5,
                  kernelized : bool =True):
    
    random_seed = 0
    np.random.seed(random_seed)

    if dataset_name == 'conductivity':
        color_map = color_map_sigma
    else:
        color_map = color_map_gap
    
    # obtain one-hot representation from chemical formulas
    comp_vecs = generate_compvec(df)
    km        = KMeans(n_clusters=n_clusters, random_state=random_seed)
    pca       = PCA(n_components=2, random_state=random_seed)
    
    if kernelized:
        rbf    = RBFSampler(random_state=random_seed)
        labels = km.fit_predict(rbf.fit_transform(comp_vecs))

        #NO PLOTTING
        embs   = pca.fit_transform(rbf.fit_transform(comp_vecs))
        colors = np.array([color_map[label] for label in labels])
                
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(embs[:,0], 
                        embs[:,1],
                        alpha=0.6,
                        label=labels,
                        s=3,
                        c=colors)
        
        #cluster counts
        cluster_counts = {cluster: len(np.where(labels == cluster)[0]) for cluster in np.unique(labels)}
        
        clusters      = np.unique(labels)
        legend_labels = [str(cluster) + ' (' + str(cluster_counts[cluster]) + ')' for cluster in clusters]
        legend_colors = [color_map[cluster] for cluster in clusters]
        
        # Create custom legend elements for the specified clusters
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markersize=7,
                                      markerfacecolor=color, label=label)
                           for label, color in zip(legend_labels, legend_colors)]

        if dataset_name == 'conductivity':
            legend_title = 'clusters $(\sigma)$'
        else:
            legend_title = 'clusters $(E_{g})$'
        # Add the legend using the custom legend elements
        ax.legend(handles=legend_elements,
                    ncol=5,
                    handletextpad=0.1,  # Adjust the space between handle and text
                    columnspacing=0.4,  # Adjust the space between columns
                    borderpad=0.3,
                    title=legend_title, 
                    fontsize=15,
                    loc='lower center',
                    bbox_to_anchor=(0.5, 1.0))

        ax.set_xlabel('PCA 1', labelpad=10)
        ax.set_ylabel('PCA 2')

        if dataset_name == 'conductivity':
            plt.savefig('sigma_loco.png', bbox_inches='tight',dpi=600)
        else:
            plt.savefig('gap_loco.png', bbox_inches='tight', dpi=600)
    else:
        labels = km.fit_predict(comp_vecs)
        # embs   = pca.fit_transform(comp_vecs)
        # fig, ax = plt.subplots(figsize=(8,6))
        # ax.scatter(embs[:,0], embs[:,1],s=7, c =labels)
        # ax.set_title(f'LOCO-CV {dataset_name}')
        # ax.set_xlabel('PCA 1')
        # ax.set_ylabel('PCA 2')
        # ax.legend(*sc.legend_elements(), 
        #           title='clusters',
        #           bbox_to_anchor=[1.2,1.0])
        
    df['loco'] = labels
    return df

def generate_compvec(df_ : pd.DataFrame):
    df       = df_.copy()
    formulae = df['formula']
    
    # to use generate_features()
    dummy_df = pd.DataFrame({'formula': formulae, 'target':[0 for _ in range(len(formulae))]})
    comp_vecs, _, _, skipped = generate_features(dummy_df, elem_prop='onehot')
    
    if len(skipped) != 0:
        print(f'Formulas {skipped} were skipped!')
    
    return comp_vecs
        

