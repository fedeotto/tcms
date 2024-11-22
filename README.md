# Assessing data-driven predictions of band gap and conductivity for transparent conducting materials
[![arXiv](https://img.shields.io/badge/arXiv-2411.14034-b31b1b.svg)](https://arxiv.org/abs/2411.14034)
This repository contains python code accompanying the paper **Assessing data-driven predictions of band gap and electrical conductivity for transparent conducting materials**.

## Installation
1. Clone this repository:
   ```git
   git clone https://github.com/fedeotto/nsg
   ```
2. Install a new `conda` environment from `env.yml`:
   ```git
   conda env create -f env.yml
   ```
3. Activate the new environment:
   ```git
   conda activate tcms
   ```

## Usage
### Data
Both `conductivity.xlsx` and `bandgap.xlsx` in the `datasets` folder are open-access and can be derived from matminer datasets (<a href="https://hackingmaterials.lbl.gov/matminer/dataset_summary.html">link</a>) under the names `ucsb_thermoelectrics` and `matbench_expt_gap`, respectively. However, these are provided solely for demonstration purposes to run the code, while the full datasets presented in the paper *cannot* be disclosed due to confidentiality agreements and restrictions associated with the use of proprietary commercial databases. To access the full raw data, an API license for the Materials Platform for Data Science (MPDS) can be purchased at the following <a href="https://mpds.io">link</a>.

### Trained models
We provide access to trained models (CrabNet and Random Forest) on the full data presented in the papers at the following <a href="https://drive.google.com/drive/folders/16cIHWnbz585LBH1cTGj3jXh9TgGLtbNV?usp=drive_link">link</a> (GDrive).

### Screening new materials from custom materials lists
Trained models can be used to predict electrical conductivity and band gap from arbitrary chemical compositions. You can reproduce the results illustrated in `Table 4` of the paper using trained models via
```git
python main.py action=screen model=crabnet ++screen.screen_path=datasets/tcms.xlsx
```

### Jupyter Notebooks


### Other 