[![arXiv](https://img.shields.io/badge/arXiv-2411.14034-b31b1b.svg)](https://arxiv.org/abs/2411.14034)
# Assessing data-driven predictions of band gap and conductivity for transparent conducting materials
This repository contains python code accompanying the paper **Assessing data-driven predictions of band gap and electrical conductivity for transparent conducting materials**.

## Installation
1. Clone this repository:
   ```git
   git clone https://github.com/fedeotto/tcms
   ```
2. Install a new `conda` environment from `env.yml`:
   ```git
   conda env create -f env.yml
   ```
3. Activate the new environment:
   ```git
   conda activate tcms
   ```
Change the `.env.template` file to `.env` , modifying the necessary paths accordingly.
## Usage
### Data
Due to licensing restrictions associated with MPDS data and confidentiality agreements tied to funding, we are unable to publicly release the full experimental datasets used in this study (band gap and conductivity). However, we provide representative data that can be used as demonstration to run the proposed pipeline.

First, you can download the updated version of UCSB dataset at the following <a href="https://zenodo.org/records/15365345">link</a> (`te_expt.xlsx`) and move it into the `data` folder. Then, run the script `prepare_data.py` via:
```git
python -m data.prepare_data
```
This will automatically create `conductivity.xlsx` and `bandgap.xlsx` datasets in the `data` folder that can be used to run the code. Band gap data is automatically extracted using matminer (`matbench_expt_gap`).

However, these can be utilized only for demonstration purposes, while the full datasets presented in the paper *cannot* be disclosed due to confidentiality agreements and restrictions associated with the use of commercial databases. To access the full raw data, an API license for the Materials Platform for Data Science (MPDS) can be purchased at the following <a href="https://mpds.io">link</a>.

### Trained models
We provide access to trained models (CrabNet and Random Forest) on the full data presented in the paper at the following <a href="https://drive.google.com/drive/folders/1pe5J-yAY4s7wtDOItUkfz-BMdqCzAjS3?usp=sharing">link</a> (GDrive).

### Screening new materials from custom material lists
Trained models can be used to predict electrical conductivity and band gap from arbitrary chemical compositions. You can reproduce the results illustrated in `Table 4` of the paper using trained models via
```git
python main.py action=screen model=crabnet ++screen.screen_path=datasets/tcms.xlsx
```

### Jupyter Notebooks
We include jupyter notebooks illustrating the analysis presented in the paper:

  1. `att_coeff.ipynb`: contains the analysis relative to the attention coefficients of CrabNet in the task of leave-one-TCM-out described in the paper.

  2. `bandgap_prediction.ipynb`: illustrates the comparison on the task of band gap prediction across different settings, in particular Random Forest, CrabNet with no fine-tuning and CrabNet fine-tuned on Materials project band gap data.

  3. `parity.ipynb` contains additional visualization and plotting corresponding to electrical conductivity and band gap predictions.

### Other usage
It is also possible to evaluate/fit new machine learning models on available data. For example, to fit a new CrabNet model on conductivity data you could simply do it via:

```git
python main.py action=fit model=crabnet data=conductivity
```
