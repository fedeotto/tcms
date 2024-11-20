import os
from pathlib import Path
from assets import evaluate, fit, screen, plotting
from assets.leave_one_tcm_out import leave_one_tcm_out
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import hydra
import os
import warnings

os.environ['HYDRA_FULL_ERROR'] = '1'
warnings.filterwarnings("ignore")

@hydra.main(config_path="conf", config_name="default")
def run(cfg: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    os.chdir(original_cwd)

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "run_conf.yaml").write_text(yaml_conf)

    if cfg.action.name == 'eval':
        evaluate.cross_validate(cfg) #results stored in eval_results folder
    elif cfg.action.name == 'fit':
        fit.fit_model(cfg)
    elif cfg.action.name == 'screen':
        screen.screen_materials_list(cfg)

    elif cfg.action.name == 'lotcmo':
        families = ['ZnO','SnO2','In2O3']
        for family in families:
            if family == 'ZnO':
                dopants = ['Al', 'Ga', 'Al-Sn']
            elif family == 'In2O3':
                dopants = ['Sn']
            elif family == 'SnO2':
                dopants = ['Ga', 'In', 'Mn','Ta','Ti','W']
            for dopant in dopants:
                leave_one_tcm_out(family, dopant, cfg)

if __name__=='__main__':
    run()
