# ConSlide
[ICCV 2023] ConSlide: Asynchronous Hierarchical Interaction Transformer with Breakup-Reorganize Rehearsal for Continual Whole Slide Image Analysis.

## Training Data Preparation

We mainly follow the pipeline of [CLAM](https://github.com/mahmoodlab/CLAM). The modified version of the CLAM code for data preparation will be released later.

## Training Example

```
python -u utils/main.py --model cocoopmil_continual --dataset seq-wsi --exp_desc cocoopmil_continual --n_epochs 50
python -u utils/main.py --model cocoopmil_naive --dataset seq-wsi --exp_desc cocoopmil_naive --n_epochs 50
python -u utils/main.py --model cocoopmil_joint --dataset seq-wsi --exp_desc cocoopmil_joint --n_epochs 200

python -u utils/main.py --model gdumb --dataset seq-wsi --exp_desc gdumb --buffer_size 1100 
python -u utils/main.py --model er_ace --dataset seq-wsi --exp_desc er_ace --buffer_size 1100 --n_epochs 50 

python -u utils/main.py --model lwf --dataset seq-wsi --exp_desc lwf --alpha 0.2
python -u utils/main.py --model ewc_on --dataset seq-wsi --exp_desc ewc_on --e_lambda 0.1 --gamma 0.1
python -u utils/main.py --model derpp --dataset seq-wsi --exp_desc derpp --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 1100
python -u utils/main.py --model derpp --dataset seq-wsi --exp_desc derpp --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 0

python -u utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 1100
python -u utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --alpha 0.2 --beta 0.2 --n_epochs 50 --buffer_size 0 

```



## Acknowledgements

Framework code for Continual Learning was largely adapted via making modifications to [Mammoth](https://github.com/aimagelab/mammoth)