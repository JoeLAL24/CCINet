# CCINet

The official repo of the paper `A Cascaded Consensus Interaction Network for Co-saliency Object Detection`.

## Abstract

Co-saliency object detection simulates human attention behavior, which is dedicated to find common salient objects in a group of related images. Previous methods usually lack the interaction among the extracted co-saliency information, leading to the detection maps being incomplete or redundant. In this paper, we propose a cascaded consensus interaction network (CCINet) for co-saliency object detection, which enhances the fusion and interaction among features to make full use of these co-saliency information. In the stage of encoding, we propose an edge semantic consensus (ESC) module, which integrates encoding information of the low layer and the high layer effectively, so that the obtained feature has both edge fine-grained information of the lower layer and rich semantic information of the high layer. ESC module initially screens and refines the co-saliency features, which helps to better detect the co-saliency region later. In the up-sampling stage, the cascaded contextual aggregation (CCA) module is proposed. It uses attention mechanism, adaptive average pooling and separation-dilated convolution to extract features thoroughly, which greatly reduces the noise interference while greatly controlling the number of parameters well. Extensive experiments show that our model achieves better performance than the excellent CoSOD methods in recent years under the three most popular benchmark datasets. Source code is available at https://github.com/JoeLAL24/CCINet.git.

## Framework Overview

The pipeline of CCINet is shown in the figure below.

![fig2high](https://github.com/JoeLAL24/CCINet/assets/100739402/4a388a2e-f405-4854-8f7a-aa656233434c)

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

reproductions by myself on RTX3090 can be found at [reproduction1](https://drive.google.com/file/d/1vovii0RtYR_EC0Y2zxjY_cTWKWM3WaxP/view?usp=sharing).
