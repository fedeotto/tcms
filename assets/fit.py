from omegaconf import DictConfig
import pandas as pd
from common.utils import preprocess_data_ml
from assets.CrabNet.kingcrab import CrabNet
from assets.CrabNet.model import Model
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from assets.CrabNet.utilities.utilities import RobustL1
from sklearn.preprocessing import MinMaxScaler
from assets.random_forest.RandomForest import RandomForest
from assets.cbfv.composition import generate_features
import pickle
import numpy as np
import random

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

def fit_model(cfg: DictConfig):
    '''FITS MODEL AND STORES MODELS IN ./trained_models/...'''
    if cfg.model.name == 'crabnet':
        fit_crabnet(cfg)
    elif cfg.model.name == 'rf':
        fit_rf(cfg)

def fit_crabnet(cfg: DictConfig,
                df = None,
                save_path: str = './trained_models/crabnet'):

    if df is None:
        df = pd.read_excel(f'./datasets/{cfg.data.name}.xlsx')

    df = preprocess_data_ml(df,
                            dataset_name= cfg.data.name,
                            elem_prop   = cfg.model.elem_prop,
                            shuffle     = True)
    
    for n in range(cfg.model.n_ensemble):
        name = f'{cfg.model.name}_{cfg.data.name}_run_{n}.pt'
        model = Model(CrabNet(compute_device=device,
                              d_model=cfg.model.d_model,
                              heads=cfg.model.heads,
                              N=cfg.model.N,
                              random_state=n)
                              .to(device),
                              fudge=cfg.data.crab_fudge)
        
        df = df.sample(frac=1, random_state=n)

        if cfg.data.name == 'bandgap':
            print('Loading transfer model for bandgap..')
            model.load_network('transfer_models/transfer_crab_bandgap_mp_processed.pt')
        
        df_val = df.sample(frac=0.15, random_state=n)
        df_train = df.drop(df_val.index)

        model.load_data(df_train, train=True, batch_size=cfg.model.batch_size)
        model.load_data(df_val, train=False, batch_size=cfg.model.batch_size)
        model.fit(epochs=cfg.model.max_epochs)
        model.save_network(f'{save_path}/{name}')

def predict_crabnet(test, 
                    cfg: DictConfig,
                    predict_path: str = './trained_models/crabnet'):
    '''PREDICTS CRABNET USING TRAINED MODELS'''
    preds   = []
    ale_unc = []
    for n in range(cfg.model.n_ensemble):
        name  = f'{cfg.model.name}_{cfg.data.name}_run_{n}.pt'
        model = Model(CrabNet(compute_device=device,
                            d_model=cfg.model.d_model,
                            heads=cfg.model.heads,
                            N=cfg.model.N,
                            random_state=n).to(device),
                            fudge=cfg.data.crab_fudge)
                
        model.load_network(f'{predict_path}/{name}')
        model.load_data(test, train=False, batch_size=cfg.model.batch_size)
        act,pred,_, unc = model.predict(model.data_loader)

        preds.append(pred)
        ale_unc.append(unc)

    # storing conductivity attention weights to check interpretability
    if cfg.action.name == 'lotcmo' and cfg.data.name == 'conductivity':
        store_attention_weights(model, test, cfg)
    
    epist_unc    = np.std(np.vstack(preds), axis=0)
    final_preds  = np.mean(np.vstack(preds), axis=0)

    final_ale_unc= np.mean(np.vstack(ale_unc), axis=0)
    final_unc    = epist_unc + final_ale_unc
    return final_preds, final_unc

def store_attention_weights(crab_model,
                            test,
                            cfg:DictConfig):
    '''STORES ATTENTION WEIGHTS FOR CRABNET'''
    dataloader = crab_model.data_loader
    batch=next(iter(dataloader))
    X,y,formula = batch

    src, frac = X.squeeze(-1).chunk(2, dim=1)
    frac = frac * (1 + (torch.randn_like(frac))*cfg.data.crab_fudge)  # normal
    frac = torch.clamp(frac, 0, 1)
    frac[src == 0] = 0
    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

    #src is crab_vector.
    src = src.to(device,
                dtype=torch.long,
                non_blocking=True)

    #frac are fractions.
    frac = frac.to(device,
                dtype=torch.float32,
                non_blocking=True)
    _, attn = crab_model.model.encoder(src, frac, return_att=True)
    attn = attn.cpu().detach().numpy()

    # current_time= datetime.now()
    # time_index  = current_time.strftime("%Y%m%d%H%M%S")
    ex_formula  = formula[0]
    results     = (formula, attn)

    with open(f'saved_attentions/att_{ex_formula}.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def fit_rf(cfg: DictConfig,
           df = None,
           save_path: str = './trained_models/rf'):
    
    if df is None:
        df = pd.read_excel(f'./datasets/{cfg.data.name}.xlsx')

    df = preprocess_data_ml(df,
                            dataset_name= cfg.data.name,
                            elem_prop   = cfg.model.elem_prop,
                            shuffle     = True)

    X, y, _, skipped = generate_features(df, 
                                         elem_prop=cfg.model.elem_prop)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    with open(f'{save_path}/scaler_{cfg.data.name}.pkl', 'wb') as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #RF IS ALREADY AN ENSEMBLE METHOD
    model_pred = RandomForest(random_state=cfg.random_state)
    model_ale = RandomForest(min_samples_leaf=10, random_state=cfg.random_state)

    model_pred.fit(X,y)
    model_ale.fit(X,y)

    with open(f'{save_path}/{cfg.model.name}_pred_{cfg.data.name}_run_{cfg.random_state}.pkl', 'wb') as handle:
        pickle.dump(model_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(f'{save_path}/{cfg.model.name}_ale_{cfg.data.name}_run_{cfg.random_state}.pkl', 'wb') as handle:
        pickle.dump(model_ale, handle, protocol=pickle.HIGHEST_PROTOCOL)

def predict_rf(test, 
               cfg:DictConfig,
               predict_path: str = './trained_models/rf'):
    preds   = []
    ale_unc = []

    with open(f'{predict_path}/scaler_{cfg.data.name}.pkl', 'rb') as handle:
        scaler = pickle.load(handle)

    X, y, _, skipped = generate_features(test, 
                                         elem_prop=cfg.model.elem_prop)
    X = scaler.transform(X)

    with open(f'{predict_path}/rf_pred_{cfg.data.name}_run_{cfg.random_state}.pkl', 'rb') as handle:
        model_pred = pickle.load(handle)
    
    with open(f'{predict_path}/rf_ale_{cfg.data.name}_run_{cfg.random_state}.pkl', 'rb') as handle:
        model_ale = pickle.load(handle)

    preds, epis_unc = model_pred.predict_with_uncertainty(X, uncertainty='epistemic')
    _, ale_unc      = model_ale.predict_with_uncertainty(X, uncertainty='aleatoric')
    ale_unc[np.isnan(ale_unc)]=0
    total_uncert = epis_unc + ale_unc
    return preds, total_uncert