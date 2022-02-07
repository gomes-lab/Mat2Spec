## Density of States Prediction for Materials Discovery via Contrastive Learning from Probabilistic Embeddings

Authors: Shufeng Kong <sup>1</sup>, Francesco Ricci <sup>2,4</sup>, Dan Guevarra <sup>3</sup>, Jeffrey B. Neaton <sup>2,5,6</sup>, Carla P. Gomes <sup>1</sup>, and John M. Gregoire <sup>3</sup>
1) Department of Computer Science, Cornell University, Ithaca, NY, USA
2) Material Science Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
3) Division of Engineering and Applied Science, California Institute of Technology, Pasadena, CA, USA
4) Chemical Science Division, Lawrence Berkeley National Laboratory, Berkeley, CA, USA
5) Department of Physics, University of California, Berkeley, Berkeley, CA, USA
6) Kavli Energy NanoSciences Institute at Berkeley, Berkeley, CA, USA

This a Pytorch implementation of the machine learning model "Mat2Spec" presented in this paper (https://arxiv.org/abs/2110.11444).
Any question or suggestion about the codes please directly send to sk2299@cornell.edu

### Installation
Install the following packages if not already installed: - may take 30 mins on typical machine to install all of them:
* Python  (tested on 3.8.11)
* Pytorch (tested on 1.4.0)
* Cuda    (tested on 10.0)
* Pandas  (tested on 1.3.3) 
* Pytmatgen (tested on 2022.0.14)
* PyTorch-Geometric (tested on 1.5.0)

Please follow these steps to create an environment:

1) Download packages - example:
https://download.pytorch.org/whl/cu100/torch-1.4.0%2Bcu100-cp38-cp38-linux_x86_64.whl
https://download.pytorch.org/whl/cu100/torchvision-0.5.0%2Bcu100-cp38-cp38-linux_x86_64.whl
https://data.pyg.org/whl/torch-1.4.0/torch_cluster-1.5.4%2Bcu100-cp38-cp38-linux_x86_64.whl
https://data.pyg.org/whl/torch-1.4.0/torch_scatter-2.0.4%2Bcu100-cp38-cp38-linux_x86_64.whl
https://data.pyg.org/whl/torch-1.4.0/torch_sparse-0.6.1%2Bcu100-cp38-cp38-linux_x86_64.whl
https://data.pyg.org/whl/torch-1.4.0/torch_spline_conv-1.2.0%2Bcu100-cp38-cp38-linux_x86_64.whl

2) Install packages - example

```bash
conda create --name mat2spec python=3.8
conda activate mat2spec
pip install torch-1.4.0+cu100-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.5.0+cu100-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.5.4+cu100-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.4+cu100-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.1+cu100-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.0+cu100-cp38-cp38-linux_x86_64.whl
pip install torch-geometric==1.5.0
pip install pandas
pip install pymatgen
```

When finish using our model, you can deactivate the environment:
```bash
conda deactivate
```

Remember to activate the environment before using our model next time:
```bash
conda activate mat2spec
```

### Datasets

1) Phonon density of state: see our data repository link below, or data can be downloaded from here  https://github.com/zhantaochen/phonondos_e3nn.
2) Electronic density of state: see our data repository link below, or data can be downloaded from the Materials Project. 
3) Initial element embeddings: please refer to "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties" by Tian Xie and Jeffrey C. Grossman.

These initial element embeddings include the embeddings of the following elements: 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'.

Datasets for this work are avaiable at https://data.caltech.edu/records/8975

Please download the data folder and unzip it under the main folder 'Mat2Spec'.

### Example Usage

Model training typically takes 20 min for phDOS and 3 hours for eDOS on a GPU. 

To train the model on phDOS with maxnorm and MSE:
```bash
bash SCRIPTS/train_phdos51_norm_max_mse.sh
```
Note that the bash scripts manually assign the CUDA device index via environment variable CUDA_VISIBLE_DEVICES and should be adjusted to the correct index (usually '0' for single GPU systems) prior to training or else Pytorch will only leverage CPU.

To train the model in eDOS with std and MAE:
```bash
bash SCRIPTS/train_dos128_std_mae.sh
```

To train the model in eDOS with norm sum and KL:
```bash
bash SCRIPTS/train_dos128_norm_sum_kl.sh
```

To test the trained models:
```bash
bash SCRIPTS/test_phdos51_norm_max_mse.sh
bash SCRIPTS/test_dos128_std_mae.sh
bash SCRIPTS/test_dos128_norm_sum_kl.sh
```

To use the trained models for predicting eDOS for material without label:

1) Place your json files under ./Mat2Spec_DATA/materials_without_dos/
Each json file should includes a key 'structure' which maps to a material in the pymatgen format.

2) Place a csv file named 'mpids.csv' that contains all your json files' names under ./DATA/20210623_no_label 

3) If you want to use trained models with std and MAE:

```bash
bash SCRIPTS/test_nolabel128_std_mae.sh
```

4) If you want to use trained models with norm sum and KL:

```bash
bash SCRIPTS/test_nolabel128_std_mae.sh
bash SCRIPTS/test_nolabel128_norm_sum_kl.sh
```

Then rescale the KL prediction with the std prediction:
```bash
x_sd = np.load('prediction_Mat2Spec_no_label_128_standardized_MAE_trainsize1.0.npy')
x_kl = np.load('prediction_Mat2Spec_no_label_128_normalized_sum_KL_trainsize1.0.npy')
x = x_kl*np.sum(x_sd, axis=-1, keepdims=True)
```


All test results (model-predicted DOS) are placed under ./RESULT

 
### Disclaimer
This is research code shared without support or any guarantee on its quality. However, please do raise an issue or submit a pull request if you spot something wrong or that could be improved and I will try my best to solve it. 

### Acknowledgements
Implementation of the GNN is inspirated from GATGNN: https://github.com/superlouis/GATGNN.
