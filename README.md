# Collective Non-negative Tensor Factorization (AAAI-19)

This repo contains the PyTorch implementation of the paper `Learning Phenotypes and Dynamic Patient Representations via RNN Regularized Collective Non-negative Tensor Factorization` in AAAI-19. [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/3920) [[dataset]](https://mimic.physionet.org/)

## Requirements
The codes have been tested with the following packages:
- Python 3.7   
- PyTorch 0.4.1  

## Quick Demo
To run the model with a quick demo data, simply clone the repo and decompress the data archive by executing the following commands:
```bash
git clone git@github.com:jakeykj/cntf.git
cd cntf
tar -xzvf demo_data.tar.gz
python train.py demo_data/
```
A folder `./results/cntf_hitf/` will be automatically created and the results will be saved there.

Use `python train.py --help` to obtain more information about setting the parameters of the model.

## Data Format and Organization
The data are stored in two seperate files contained in a folder (we refer to its path by `<DATA_PATH>`): `<DATA_PATH>/data.pkl` and `<DATA_PATH>/list.pkl`. They should be saved using the `pickle` package in Python, and the format of the two files are described as follows:
- **`data.pkl`**: contains a list of the temporal sparse tensor, one for each patient. It is a python list of dictionary with six keys. `subs` is a `torch.LongTensor` object containing the subscripts of the non-zero elements of the tensor (for a 3rd-order tensor, its size should be 3-by-K, where K is the number of non-zero elements). `vals` is a `torch.FloatTensor` object containing the values of the tensor that are corresponding to the non-zero subscripts (size: K). `size` is a `torch.Size` object describing the size of the temporal sparse tensor. `hadm_id` is the unique ID of the patient (a python interger). `dx_vector` is a `torch.FloatTensor` object containing the one-hot vector of the diagnoses of the patient. `rx_vector` is a `torch.FloatTensor` object containing the counting vector of the medications of the patient.
- **`list.pkl`**: contains the temporal length and label information of the patient. It is a python list of nested python lists, one for each patient. The format is: `[[<hadm_id>, <temporal_length>, <label>], ...]`.  

If you use other datasets, you can organize the input data in the same format described above, and pass the `<DATA_PATH>` as a parameter to the training script:
```bash
python train.py <DATA_PATH>
```


## Citation
If you find the paper or the implementation helpful, please cite the following paper:
```bib
@inproceedings{yin2019learning,
  title={Learning phenotypes and dynamic patient representations via {RNN} regularized collective non-negative tensor factorization},
  author={Yin, Kejing and Qian, Dong and Cheung, William K. and Fung, Benjamin C. M. and Poon, Jonathan},
  booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence},
  pages={1246--1253},
  year={2019},
  organization={AAAI Press}
}
```

## Contact
If you have any enquires, please contact Mr. Kejing Yin by email: cskjyin [AT] comp [DOT] hkbu.edu.hk, or leave your questions in issues. 

---
:point_right: Check out [my home page](https://kejing.me) for more research work by us.

