# CCINet

The official repo of the paper `A Cascaded Consensus Interaction Network for Co-saliency Object Detection`.

## Abstract

Co-saliency object detection imitates human attention behavior, with the aim of identifying common salient objects in a set of related images. Previous approaches generally have a lack of interaction among the extracted co-saliency information. As a result, the detection maps turn out to be incomplete or redundant. In this paper, we propose a Cascaded Consensus Interaction Network (CCINet) for co-saliency object detection. This network improves the fusion and interaction among features, thus making full use of the co-saliency information. In the encoding stage, we introduce an Edge Semantic Consensus (ESC) module. It can effectively integrate low-level and high-level encoding information. In this way, it is able to capture both fine edge details and rich semantics. Meanwhile, the ESC module refines the co-saliency features, which enhances the detection of co-saliency regions. During the up-sampling stage, the Cascaded Contextual Aggregation (CCA) module employs attention mechanisms, adaptive pooling, and separation - dilated convolution for comprehensive feature extraction. This approach can effectively reduce background noise and control the number of parameters. Extensive experiments indicate that our model outperforms many excellent CoSOD methods in recent years on the three most popular benchmark datasets

## Framework Overview

The pipeline of CCINet is shown in the figure below.

![image](https://github.com/user-attachments/assets/d716812c-6f2c-4137-ab9d-47cf57e43ce2)


## Result

- Comparisons with the excellent CoSOD methods in recent years on three benchmarks, CoCA, Cosal2015 and CoSOD3k. The value of MAE is the smaller, the better. While others are the larger, the better. Black bold fonts indicates the best performance.

![image](https://github.com/user-attachments/assets/4dac2a94-2008-4932-ad08-f1651e8f35e1)

- Ablation study:

![image](https://github.com/user-attachments/assets/0195eef3-7510-4801-b15d-28d7aa3fafb7)  ![image](https://github.com/user-attachments/assets/0ae7f3b9-06e3-45f8-b0de-1079562319b2)
![image](https://github.com/user-attachments/assets/6802a6df-6888-4841-9ecd-6100849f9973)  ![image](https://github.com/user-attachments/assets/1c321be9-a8e1-4e6e-9a2b-8b5106acf3d3)

## Prediction

![image](https://github.com/user-attachments/assets/81841066-11eb-4377-b0fe-a24d12a42b42)

## Environment Requirement

create environment and install as following: `pip install -r requirements.txt`

## Data Format

trainset: COCO-SEG

testset: CoCA, CoSOD3k, Cosal2015

Put the datasets ([gts](https://pan.baidu.com/s/1A0cklgxqK2yPtYI7GNY62Q?pwd=7xo7) and [imgs](https://pan.baidu.com/s/1Bf3HfdDWMiV4MIaHu2MJQQ?pwd=scub)) to `CCINet/data` as the following structure:

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

trained model can be downloaded from [papermodel](https://pan.baidu.com/s/1R-isw86_4UrCGNo2T2ubSg?pwd=koy2).

Run `test.py` for inference.

The evaluation tool please follow: https://github.com/zzhanghub/eval-co-sod

## Reproduction

reproductions by myself on RTX3090 can be found at [reproduction](https://pan.baidu.com/s/1R-isw86_4UrCGNo2T2ubSg?pwd=koy2).
