# Assessing data-driven predictions of band gap and conductivity for transparent conducting materials

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
We provide restricted, open-access version of the datasets discussed in the paper. `conductivity.xlsx` is derived from the `ucsb_thermoelectrics` dataset while `bandgap.xlsx` is derived from `matbench_expt_gap` dataset from matminer. for demonstration purposes to run the code. However, the full datasets presented in the paper cannot be disclosed due to confidentiality agreements and restrictions associated with the use of proprietary commercial databases. An API license for the Materials Platform for Data Science can be purchased at the following link.

### Train ML models on available data
It is possible to fit available ML models on custom data of $\sigma$ and $E_g$. <br> 
Below, you can find an example of fitting CrabNet using a custom `bandgap.xlsx` dataset:

#### Example: Fitting a CrabNet model on custom Band Gap data
```git
python run.py action=fit model=crabnet data=bandgap
```
Trained models are stored into `trained_models/{model_name}` (Available models are `crabnet`, `rf`, and `dopnet`). It is recommended to utilize either `crabnet` or `rf` as `dopnet` is in a more experimental stage and considered to be less reliable than the other two. Unless a new substantial volume of data becomes available, it might be **not needed** to fit the models, as the latest ones (trained on top of our validated databases at UoL) are available here: https://drive.google.com/drive/folders/1zhbSQgu4TSjLzQzzC67ats7tvlyU7Pc7?usp=drive_link

### Screening new materials from custom materials lists
Fitted ML models can be used to predict electrical conductivity and band gap of new materials. To do so, you have to place a separate `csv` file named `materials_list.csv` that you want to screen, with two columns `Entry` and `formula` :

<table>
  <thead>
    <tr>
      <th>Source</th>
      <th>Entry</th>
      <th>Formula</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MPDS</td>
      <td>P1522899-1</td>
      <td>Sr1.33Mn4O8</td>
    </tr>
    <tr>
       <td>ICSD</td>
      <td>P604346-1</td>
      <td>Al2Cu1</td>
    </tr>
    <tr>
      <td>MPDS</td>
      <td>P1916288-1</td>
      <td>Cr23C6</td>
    </tr>
  </tbody>
</table>

<!--
You can focus the screening procedure around two main material classes, **transparent conductors** (TCMs) and **correlated metals** (COMs):
   1. `tcms` : electrical conductivity > $10^2$ S/cm , band gap > 3eV
   2. `coms` : $10^4$ S/cm < electrical conductivity < $10^6$ S/cm, band gap ~ 0 eV
-->
If a specific entry (e.g. DOI) or source is not available, just any string can be used for the `Entry` and `Source` columns. Below an example of ML screening to find new TCMs using a fitted `crabnet` model:

#### Example: Screening new TCMs from a custom list using NSG filters and fitted CrabNet model.
```git
python run.py action=screen model=crabnet ++action.screen_path=datasets/materials_list.csv ++action.nsg_filter=False ++action.material_type=tcms
```
It is also possible to perform screening by using multiple trained ML models. For now, only `crabnet` and `rf` are utilized as `dopnet` is still experimental. This behavior is obtained by setting `++action.all_models=true`. See the example below:

#### Example: Screening new TCMs from a custom list using NSG filters and both RF and CrabNet models.
```git
python run.py action=screen ++action.all_models=True ++action.screen_path=datasets/materials_list.csv action.nsg_filter=True ++action.material_type=tcms
```

Setting `++action.material_type=tcms` will focus the screening over TCMs, returning all materials with predicted electrical conductivity > $10^2$ S/cm and band gap > $3$ eV. Setting instead `++action.material_type=all` will return all predictions for materials in `materials_list.csv`. <br>
`++action.nsg_filter=False` will filter out all chemical formulas containing unfeasible elements in line with NSG **Viability list of elements**. <br> 
Screening results will be stored into `screening_results` folder under the name `{model_name}_tcms` (or `{model_name}_all`). Below, an example of screening result:

<table>
  <thead>
    <tr>
      <th>Source</th>
      <th>Entry</th>
      <th>Formula</th>
      <th>eg_pred (eV)</th>
      <th>eg_uncert (eV)</th>
      <th>sigma_pred log10 (S/cm)</th>
      <th>sigma_uncert log10 (S/cm)</th>
      <th>Model</th>
      <th>Phi_M</th>
      <th>Phi_M_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MPDS</td>
      <td>entry1</td>
      <td>Al0.03Zn0.955O1</td>
      <td>3.57</td>
      <td>0.97</td>
      <td>3.50</td>
      <td>1.29</td>
      <td>crabnet</td>
      <td>12.50</td>
      <td>5.71</td>
    </tr>
    <tr>
      <td>MPDS</td>
      <td>entry2</td>
      <td>Mo1O2</td>
      <td>3.06</td>
      <td>1.53</td>
      <td>4</td>
      <td>3.20</td>
      <td>crabnet</td>
      <td>12.22</td>
      <td>11.52</td>
    </tr>
    <tr>
      <td>MPDS</td>
      <td>entry3</td>
      <td>Mg0.03Zn0.97O1</td>
      <td>3.67</td>
      <td>1.18</td>
      <td>3.21</td>
      <td>1.65</td>
      <td>crabnet</td>
      <td>11.79</td>
      <td>7.14</td>
    </tr>
  </tbody>
</table>

In the above table, `Phi_M` is an empirical figure of merit defined as the product or `sigma_pred log10 (S/cm) * eg_pred (eV)`, which ranks the high-reward materials (in terms of predicted conductivity and band gap). `Phi_M_std` is the corresponding standard deviation computed according the rules of <a href="https://chem.libretexts.org/Bookshelves/Analytical_Chemistry/Analytical_Chemistry_2.1_(Harvey)/04%3A_Evaluating_Analytical_Data/4.03%3A_Propagation_of_Uncertainty">uncertainty propagation</a>.

### Interactive plotting.
It is possible to create an interactive plot of predicted materials. Below an example:

#### Example: plotting TCM candidates identified by fitted CrabNet and RF models using a custom materials list
```git
python run.py action=plot ++action.screen_result_path=screen_results/all_tcms.xlsx
```
This will generate an `html` file named `tcms_candidates.html` that can be opened using a browser.

### New functionality for NSG

<!--
All the retrieved candidates are automatically attached to available references, stored in `./datasets/dois.xlsx`. Below an example of predicted candidates from a list of materials:

| Entry    | Formula | eg_pred (eV) | eg_uncert (eV) | sigma_pred log10 (S/cm) | sigma_uncert log10 (S/cm) | model_type | eg_exp | sigma_exp | charges |
|----------------------------------------------------------------------------------------------------------|---------|--------------|-----------------|-------------------------|----------------------------|------------|--------|-----------|----------------|
| DOI1  | Al0.1Zn0.85O1     | 3.38 | 0.50   | 3.11 | 0.54   | crabnet | 0  | 0  |{'Al3+': 0.1, 'Zn2+': 0.85, 'O2-': 1.0}|
| DOI2  | Mg0.04Zn0.96O1    | 3.56 | 0.4225 | 3.13 | 0.75   | crabnet | 0  | 0  |{'Mg2+': 0.04, 'Zn2+': 0.96, 'O2-': 1.0}|
| DOI3  | Ti0.15In4Sn2.85O12| 3.41 | 1.68   | 2.05 | 1.03   | dopnet  | 0  | 0  |{'Ti4+': 0.15, 'In3+': 4.0, 'Sn4+': 2.85, 'O2-': 12.0}|
| DOI4  | Zn2In2O5          | 3.11 | 1.40   | 3.30 | 0.31   | dopnet  | 0  | 1  |{'Zn2+': 2, 'In3+': 2, 'O2-': 5}|

### Cross-referencing original training datasets for extracting candidates
It is also possible to cross-reference directly the original training datasets in order to check if there are already materials matching the requested criteria. Additionally you can utilize trained ML models to provide the missing property ($\sigma$ or $E_g$) for each of the two datasets. Under the setting `mode` we can specify different cross-referencing strategies. For example, `orig_cond_orig_gap` will simply merge training datasets and looking for entries that match the criteria (either for `tcms` or `coms`):

```git
python run.py --action screening --mode orig_cond_orig_gap --material_type tcms
```

Possible modes are: 
1. `orig_cond_orig_gap` : cross-reference material candidates from available training data.
2. `orig_cond_pred_gap` : predicts band gap from conductivity dataset and search for candidates.
3. `orig_gap_pred_cond` : predicts conductivity from band gap dataset and search for candidates.
-->