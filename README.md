# FTMI
Multi-hop interpretable meta learning for few-shot temporal knowledge graph completion

This repository contains the implementation of the FTMI architectures described in the paper.

# Installation
 Install tensorflow (>= 1.3.0)
 ```
pip install tensorflow
 ```
 Python 3.6.5
  ```
pip install python 3.6.5
  ```
 Numpy
  ```
pip install numpy
  ```
 Pandas
  ```
pip install pandas
  ```
 tqdm
  ```
pip install tqdm
  ```
# How to use
run the code:
```
python trainer_FTMI.py --parameters
```

# Parameters setting

1. The embedding dimension of ICEWS18-few and GDELT-few is set to 100.
2. For both datasets, the task entity's maximum number of single-hop neighbors is set to 60, and maximum number of two-hop neighbors is set to 20. If a sufficient number of neighbors is missing, padding is performed. 
3. The number of Transform tiers is set to 6, and number of Transform headers is set to 4.  
4.  The Adam optimizer initial learning rate for updating model parameters was set to 0.001.
5. Initial learning rate was set to 0.001 and Dropout rate was set to 0.5. The boundary value is set to 1.0. 
6. Applying small batch gradient descent to update parameters during training process. Batch size is 128.

# Dateprocess

To run our code, we need to divide the data set according to the data set partition file first, or divide it according to our own needs. If we want to get the best results, we need to use Complex to pre-train and then embed it into the model.

| Baselines   | Code                                                         |
| ----------- | ------------------------------------------------------------ |
| TransE      | [Link](https://github.com/jimmywangheng/knowledge_representation_pytorch) |
| TTransE     | [link](https://github.com/INK-USC/RE-Net)                    |
| DistMult    | [link](https://github.com/BorealisAI/DE-SimplE)              |
| TA-DistMult | [link](https://github.com/INK-USC/RE-Net)                    |
| TA-TransE   | [link](https://github.com/INK-USC/RE-Net)                    |
| Gamtching   | [link](https://github.com/xwhan/One-shot-Relational-Learning) |
| MateR       | [link](https://github.com/AnselCmy/MetaR)                    |
| FSRL        | [link](https://github.com/chuxuzhang/AAAI2020_FSRL)          |
| FAAN        | [link](https://github.com/JiaweiSheng/FAAN)                  |
| TFSC        | [link](https://github.com/DMKE-Lab/TFSC)                     |

TTransE, TA-Distmult, and TA-TransE have been implemented in baselines in [Re-Net]((https://github.com/INK-USC/RE-Net/tree/master/baselines)). The user can run the baselines by the following command.

```
CUDA_VISIBLE_DEVICES=0 python3 TTransE.py -f 1 -d ICEWS18 -L 1 -bs 1024 -n 1000`
```

We have implemented DistMult refer to [RotatE](: https://github.com/DeepGraphLearning/ KnowledgeGraphEmbedding.).

```
cd ./baselines
bash run.sh train MODEL_NAME DATA_NAME 0 0 512 1024 512 200.0 0.0005 10000 8 0
```

# Output

The metrics used for evaluation are Hits@{1,3,5} and MRR. The results show that our model works better than most models.



