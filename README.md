# Collective Non-negative Tensor Factorization (AAAI-19)

This repo contains the PyTorch implementation of the paper `Learning Phenotypes and Dynamic Patient Representations via RNN Regularized Collective Non-negative Tensor Factorization` in AAAI-19. [[paper]](https://aaai.org/ojs/index.php/AAAI/article/view/3920) [[dataset]](https://mimic.physionet.org/)

## Requirements
The codes have been tested with the following packages:
- Python 3.6  
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
For relevant enquires, please contact Mr. Kejing Yin by email: cskjyin [AT] comp [DOT] hkbu.edu.hk
