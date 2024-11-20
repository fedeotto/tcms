import pandas as pd
from common.utils import preprocess_data_ml
from assets.loco_cv import apply_loco_cv
from sklearn.model_selection import LeaveOneGroupOut, KFold
from omegaconf import DictConfig
# import pytorch_lightning as pl
from assets.CrabNet.utilities.utilities import RobustL1
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from data_modules.cbfv.composition import generate_features
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

def eval_dopnet(train, test, cfg:DictConfig):
    # max_dops = dp.get_max_dops(train)

    val   = train.sample(frac=0.1, random_state=cfg.random_state)
    train = train.drop(val.index)

    
    train_dataset = dp.load_dataset(train, 
                                    comp_idx=0, 
                                    target_idx=1,
                                    max_dops=cfg.model.max_dops, 
                                    feat_type = cfg.model.elem_prop, 
                                    cond_idx=None)

    val_dataset = dp.load_dataset(val, 
                                comp_idx=0, 
                                target_idx=1,
                                max_dops=cfg.model.max_dops, 
                                feat_type = cfg.model.elem_prop, 
                                cond_idx=None)

    dop_dataset_train      = dp.get_dataset(train_dataset, cfg.model.max_dops) 
    dop_dataset_val        = dp.get_dataset(val_dataset, cfg.model.max_dops)

    data_loader_train       = DataLoader(dop_dataset_train, batch_size=cfg.model.batch_size, shuffle=True)
    data_loader_calc_train  = DataLoader(dop_dataset_train, batch_size=32, shuffle=False)

    data_loader_val         = DataLoader(dop_dataset_val, batch_size=cfg.model.batch_size, shuffle=True)
    data_loader_calc_val    = DataLoader(dop_dataset_val, batch_size=32, shuffle=False)

    emb_host         = ae.Autoencoder(train_dataset[0].host_feat.shape[0], 64)
    emb_host         = emb_host.to(device)
    optimizer_emb    = torch.optim.Adam(emb_host.parameters(), lr=cfg.model.lr_host, weight_decay=cfg.model.wd_host)

    trigger       = 0
    best_val_loss = np.inf
    for epoch in range(0, cfg.model.epochs_host):
        train_loss, val_loss = ae.train(emb_host, data_loader_train, data_loader_val, optimizer_emb)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\t Val loss: {:.4f}'.format(epoch + 1, cfg.model.epochs_host, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger > cfg.model.patience:
                print('Early stopping!')
                break

    #Using trained AE to generate host features to train DP
    host_embs_train = ae.test(emb_host, data_loader_calc_train)
    host_embs_val   = ae.test(emb_host, data_loader_calc_val)

    dop_dataset_train.host_feats = host_embs_train
    dop_dataset_val.host_feats   = host_embs_val

    data_loader_train       = DataLoader(dop_dataset_train, batch_size=cfg.model.batch_size, shuffle=True)
    data_loader_calc_train  = DataLoader(dop_dataset_train, batch_size=32, shuffle = False)

    data_loader_val         = DataLoader(dop_dataset_val, batch_size=cfg.model.batch_size, shuffle=True)
    data_loader_calc_val    = DataLoader(dop_dataset_val, batch_size=32, shuffle=False)

    # define DopNet and its optimizer
    pred_model = dp.DopNet(host_embs_train.shape[1], 
                            train_dataset[0].dop_feats.shape[1],
                            dim_out=2, 
                            max_dops=cfg.model.max_dops)
    
    pred_model = pred_model.to(device)
    optimizer  = torch.optim.SGD(pred_model.parameters(), lr=cfg.model.lr_pred, weight_decay=cfg.model.wd_pred)
    criterion  = RobustL1

    trigger       = 0
    best_val_loss = 1e10

    # train DopNet
    for epoch in range(0, cfg.model.epochs_pred):
        if (epoch + 1) % 200 == 0:
            for g in optimizer.param_groups:
                g['lr'] *= 0.5

        train_loss, val_loss = dp.train(pred_model, 
                                        data_loader_train, 
                                        data_loader_val, 
                                        optimizer, 
                                        criterion)
        print('Epoch [{}/{}]\tTrain loss: {:.4f}\tVal loss: {:.4f}'.format(epoch + 1, cfg.model.epochs_pred, train_loss, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger = 0
        else:
            trigger += 1
            if trigger > cfg.model.patience:
                print('Early stopping!')
                break

    # INFERENCE
    dataset = dp.load_dataset(test, comp_idx=0, target_idx=1, max_dops=cfg.model.max_dops, feat_type = cfg.model.elem_prop, cond_idx=None)
    dop_dataset_test  = dp.get_dataset(dataset, cfg.model.max_dops)
    data_loader_test  = DataLoader(dop_dataset_test, batch_size=32, shuffle=False)

    host_embs_test = ae.test(emb_host, data_loader_test)
    dop_dataset_test.host_feats = host_embs_test
    data_loader_test  = DataLoader(dop_dataset_test, batch_size=32, shuffle = False)
    pred, ale        = dp.test(pred_model, data_loader_test)
    act = np.hstack([d.target for d in dataset])
    return act, pred