# Predicting efficiency/effectiveness trade-offs for retrieval strategy selection
This repository includes codes and data regarding the paper "Predicting efficiency/effectiveness trade-offs for retrieval strategy selection".
In this work, we propose two classifiers in order to select a suitable retrieval strategy to maintain the trade-off between the cost and utility of  sparse vs. dense vs. hybrid retrievers for individual queries.

## Sparse vs Dense

###### Train
In order to train the classifier to decide the appropriate retrieval strategy per query i.e., sparse vs dense. First we need to label the training queries. Since sparse retriever has lower cost ciompared to dense retriever, we set the query class=0 (sparse retriever) if the sparse retriever can retrieve and rank any relevant document among top-T retrieved documents. Otherwise, we prefer the more expensive and complex dense retriever with class label =1. Labels for all queries in MSMARCO training set can be found in ```train_labels_T50.tsv``` when T=50 by ```query<\t>label```. 

Further, ```train_sparse_vs_dense.py``` will train a cross-encoder model with ```bert-base-uncased``` and save it under ```models``` directory. The following parameter can be changed for the training  in ```train_sparse_vs_dense.py```.:
*  ```model_name``` : The initial pretrained model can be changed undervariable
*  ```epoch_num``` : number of epochs
*  ```batch_size``` : batch size for training

###### Test

## Sparse vs Hybrid

###### Train

###### Test

