# AEP
This is the source code for paper AEP: Cross-lingual Knowledge Graph Alignment via Embedding Propagation.

## Requirement
Python3 (tested on 3.6.9)

Pytorch (tested on 1.7.0+cu101)

gensim (tested on 3.8.3)

## Code
WordEmbedding.py generates word embeddings and provides an evaluation function to evaluate the hits@1 of summation of word vectors.

main.py generates the alignment results of AEP. 

At first, you need to edit include.Config.py to set the parameters of AEP. 
Then, running main.py will generate word embeddings and attention-based GCN model and evaluate the hits@1 of whole AEP

If you want to train AEP(Bert), you could acquire the entity embeddings from a pre-trained Bert by [Bert-int](https://github.com/kosugi11037/bert-int)
## Acknowledgement
The code is based on the old version of [KECG](https://github.com/THU-KEG/KECG). 
The datasets are obtained from [JAPE](https://github.com/nju-websoft/JAPE) and [RSN](https://github.com/nju-websoft/RSN).



