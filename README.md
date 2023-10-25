# CCINet

The official repo of the paper `A Cascaded Consensus Interaction Network for Co-saliency Object Detection`.

## Environment Requirement

create environment and install as following: `pip install -r requirements.txt`

## Data Format

trainset: COCO-SEG

testset: CoCA, CoSOD3k, Cosal2015

Put the [CoCo-SEG](https://pan.baidu.com/s/1-nxUm-3v24QfNaFnkDl6xA?pwd=mb5s), [CoCA](https://pan.baidu.com/s/1-nxUm-3v24QfNaFnkDl6xA?pwd=mb5s), [CoSOD3k](https://pan.baidu.com/s/1-nxUm-3v24QfNaFnkDl6xA?pwd=mb5s) and [Cosal2015](https://pan.baidu.com/s/1-nxUm-3v24QfNaFnkDl6xA?pwd=mb5s) datasets to `CCINet/data` as the following structure:

```
CCINet
   ├── other codes
   ├── ...
   │ 
   └── data
         ├── gts
              ├── CoCo-SEG (CoCo-SEG's gt files)
         	  ├── CoCA (CoCA's gt files)
              ├── CoSOD3k (CoSOD3k's gt files)
              └── Cosal2015 (Cosal2015's gt files)
         ├── images
              ├── CoCo-SEG (CoCo-SEG's image files)
         	  ├── CoCA (CoCA's image files)
              ├── CoSOD3k (CoSOD3k's image files)
              └── Cosal2015 (Cosal2015's image files)
```

## Trained model

trained model can be downloaded from [papermodel](https://pan.baidu.com/s/1QcaoSwzq1ZAhDrmS2YZJKg?pwd=azxw ).

Run `test.py` for inference.

The evaluation tool please follow: https://github.com/zzhanghub/eval-co-sod

