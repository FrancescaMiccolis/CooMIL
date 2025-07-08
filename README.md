# CooMIL
Context-guided Prompt-learning for Continual WSI Classification

## Training Data Preparation

We mainly follow the pipeline of [CLAM](https://github.com/mahmoodlab/CLAM). The modified version of the CLAM code for data preparation will be released later.

## Training Example

```
python utils/main.py --model conslide --dataset seq-wsi --exp_desc conslide --buffer_size 1100 --alpha 0.2 --beta 0.2
```

## Updates / TODOs
Please follow this GitHub for more updates.

- [ ] Refine the code.
- [ ] Provide code for data preparation.
- [ ] Remove dead code.
- [ ] Better documentation on interpretability code example.


```}
```

## Acknowledgements

Framework code for Continual Learning was largely adapted via making modifications to [ConSlide]([https://github.com/HKU-MedAI/ConSlide])
