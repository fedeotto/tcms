import pandas as pd
from common.utils import preprocess_data_ml
from assets.loco_cv import apply_loco_cv
from sklearn.model_selection import LeaveOneGroupOut, KFold
from omegaconf import DictConfig
# import pytorch_lightning as pl
from assets.CrabNet.utilities.utilities import RobustL1
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from assets.cbfv.composition import generate_features
from assets.random_forest.RandomForest import RandomForest
from assets.CrabNet.kingcrab import CrabNet
from assets.CrabNet.model import Model
import numpy as np
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cross_validate(cfg: DictConfig):
    df = pd.read_excel(cfg.data.path)
    df = preprocess_data_ml(df,
                            dataset_name= cfg.data.name,
                            elem_prop   = cfg.model.elem_prop,
                            shuffle     = True,
                            screening   = False)
    
    cv         = cfg.action.cv.method
    model_name = cfg.model.name

    if cv == 'LOCO':
        df = apply_loco_cv(df, dataset_name = cfg.data.name)
    
    for run in range(cfg.action.n_runs): #just 1
        # Use two data holdout techniques
        loco  = LeaveOneGroupOut()
        kfold = KFold(n_splits=5, 
                        shuffle=True, 
                        random_state=cfg.random_state)
        if cv == "LOCO":
            cv_indices = loco.split(df["formula"], df["target"], df["loco"])
        elif cv == "Kfold":
            cv_indices = kfold.split(df["formula"], df["target"])
        # Generate models for each train/test split and save the test predictions
        for i, (train_index, test_index) in enumerate(cv_indices):
            train, test = df.iloc[train_index], df.iloc[test_index]

            if cv == 'LOCO':
                train = train.drop('loco', axis=1)
                test  = test.drop('loco', axis=1)

            formulae = test['formula']

            if model_name == 'crabnet':
                act, pred, formulae = eval_crabnet(train, test, cfg)
            elif model_name == 'rf':
                act, pred = eval_rf(train, test, cfg)
                act = act.values
            elif model_name == 'dopnet':
                act, pred = eval_dopnet(train, test, cfg)

            res_df = pd.DataFrame({"composition": formulae, 
                                    "real": act, 
                                    "pred": pred})

            res_df.to_excel(f"eval_results/{cfg.data.name}/{cfg.model.name}/{cv}_{i}.xlsx", index=False)

def eval_crabnet(train, test, cfg: DictConfig):
    val_set    = train.sample(frac=0.1, random_state=cfg.random_state)
    train      = train.drop(val_set.index)
    assert set(train['formula']) & set(test['formula']) & set(val_set['formula']) == set()

    model = Model(CrabNet(d_model=cfg.model.d_model,
                          compute_device=device,
                          N=cfg.model.N,
                          heads=cfg.model.heads,
                          random_state=cfg.random_state).to(device),
                          fudge=cfg.data.crab_fudge)
    
    if cfg.data.name == 'bandgap':
        if cfg.data.path_ft != '':
            print('Loading transfer model for bandgap..')
            model.load_network(cfg.data.path_ft)

    model.load_data(train, train=True, batch_size=cfg.model.batch_size)
    model.load_data(val_set, train=False, batch_size=cfg.model.batch_size)
    model.fit(epochs=cfg.model.max_epochs)

    model.load_data(test, train=False)
    act,pred,formulae,_ = model.predict(model.data_loader)
    return act, pred, formulae

def eval_rf(train, test, cfg: DictConfig):
    rf= MyRandomForest(n_estimators=cfg.model.n_estimators,random_state=cfg.random_state)
    X_train, y_train, _, skipped = generate_features(train,
                                                     elem_prop=cfg.model.elem_prop)

    X_test, y_test, _, _ = generate_features(test,
                                            elem_prop=cfg.model.elem_prop)
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    rf.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    preds  = rf.predict(X_test)
    return y_test, preds