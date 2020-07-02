# The HAM models for Sequential Recommendation
The pytorch implementation of the paper:

HAM: Hybrid Associations Models for Sequential Recommendation
Arxiv: https://arxiv.org/pdf/2002.11890.pdf

Author: Bo Peng (peng.707@buckeyemail.osu.edu)

**Feel free to send me an email if you have any questions.**

## Environments

- python 3.7.3
- PyTorch (version: 1.2.0)
- numpy (version: 1.16.2)
- scipy (version: 1.2.1)
- sklearn (version: 0.20.3)


## Dataset and Data preprocessing

Please refer to our paper for the details of datasets and the preprocessing procedure.
we upload the CDs and ML-1M datasets for the seek of reproducibility.
Please feel free to contact me if you need more preprocessed data.

## Example
Please refer to the following example on how to train and evaluate the model (you are recommended to run the code using GPUs).

```
python run.py --data=CDs --n_iter=300 --L=5 --T=3 --d=400 --model=xHAM --neg_samples=3 --P=2 --isTrain=0 --setting=CUT --l2=1e-3 --order=2
```

## Acknowledgment
The training framework is primarily built on [HGN](https://github.com/allenjack/HGN). Thanks for the great work!

