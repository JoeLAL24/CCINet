# CCINet

The official repo of the paper `A Cascaded Consensus Interaction Network for Co-saliency Object Detection`.

## Abstract

Co-saliency object detection simulates human attention behavior, which is dedicated to find common salient objects in a group of related images. Previous methods usually lack the interaction among the extracted co-saliency information, leading to the detection maps being incomplete or redundant. In this paper, we propose a cascaded consensus interaction network (CCINet) for co-saliency object detection, which enhances the fusion and interaction among features to make full use of these co-saliency information. In the stage of encoding, we propose an edge semantic consensus (ESC) module, which integrates encoding information of the low layer and the high layer effectively, so that the obtained feature has both edge fine-grained information of the lower layer and rich semantic information of the high layer. ESC module initially screens and refines the co-saliency features, which helps to better detect the co-saliency region later. In the up-sampling stage, the cascaded contextual aggregation (CCA) module is proposed. It uses attention mechanism, adaptive average pooling and separation-dilated convolution to extract features thoroughly, which greatly reduces the noise interference while greatly controlling the number of parameters well. Extensive experiments show that our model achieves better performance than the excellent CoSOD methods in recent years under the three most popular benchmark datasets. Source code is available at https://github.com/JoeLAL24/CCINet.git.

## Framework Overview

The pipeline of CCINet is shown in the figure below.

<img width="519" alt="fig2" src="https://github.com/JoeLAL24/CCINet/assets/100739402/8bdef124-631e-46bb-8f5c-0efe51700c4e">

## Result

- Comparisons with the excellent CoSOD methods in recent years on three benchmarks, CoCA, Cosal2015 and CoSOD3k. The value of MAE is the smaller, the better. While others are the larger, the better. Black bold fonts indicates the best performance.

<img width="468" alt="result_all" src="https://github.com/JoeLAL24/CCINet/assets/100739402/a4688225-8bd5-483a-976c-b30706a07e63">

- Ablation study:

<img width="259" alt="tab2" src="https://github.com/JoeLAL24/CCINet/assets/100739402/3ae56ddb-da40-423c-8afb-9d353346ba80"> <img width="257" alt="tab3" src="https://github.com/JoeLAL24/CCINet/assets/100739402/14c8a49c-e8fe-4546-a605-53beaccf1e7a">

<img width="256" alt="tab4" src="https://github.com/JoeLAL24/CCINet/assets/100739402/3c0bcb22-c82e-4b53-9f38-f70663efc576"> <img width="259" alt="tab5" src="https://github.com/JoeLAL24/CCINet/assets/100739402/3087f007-5eef-4835-a6ab-f9cbbe2b4cff">

## Prediction

![fig5high](https://github.com/JoeLAL24/CCINet/assets/100739402/706d4e35-fcd9-4bb2-8fbf-9781fade913c)

## Environment Requirement

create environment and install as following: `pip install -r requirements.txt`

## Data Format

trainset: COCO-SEG

testset: CoCA, CoSOD3k, Cosal2015

Put the [CoCo-SEG](https://drive.google.com/file/d/1GbA_WKvJm04Z1tR8pTSzBdYVQ75avg4f/view), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015](https://drive.google.com/u/0/uc?id=1mmYpGx17t8WocdPcw2WKeuFpz6VHoZ6K&export=download) datasets to `CCINet/data` as the following structure:

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

trained model can be downloaded from [papermodel](https://drive.google.com/file/d/1cfuq4eJoCwvFR9W1XOJX7Y0ttd8TGjlp/view?usp=sharing).（要改）

Run `test.py` for inference.

The evaluation tool please follow: https://github.com/zzhanghub/eval-co-sod

## Reproduction

reproductions by myself on RTX3090 can be found at [reproduction](https://drive.google.com/file/d/1cfuq4eJoCwvFR9W1XOJX7Y0ttd8TGjlp/view?usp=sharing).
