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

Training hould take less than 1 hour on RTX2080 GPU.
If you are not willing to train the model, you can [download the trained sparse vs dense model from here]()
###### Test
we test our trained sparse vs dense classifier on MSMARCO small dev set queriees (```queries.dev.small.tsv```). Run ```test_sparse_vs_dense.py``` and the trained model can be changed under ```model_name```. The results should be saved under ```results``` repository as ```prediction_sparse_vs_dense.dev.small.tsv``` in the following format:
```qid<\t>query<\t>sparse_prob<\t>dense_prob```
Based on the sparse vs dense classifier prediction, the query should be retrieved by the retriever with higher probability.
In addition, if you are willing to rank queries based on their probability of being assigned to sparse retriever, you can set ```num_labels=1``` when training and testing. As a result, for each query, you will get probability of success when retrieviing with dense retriever, you can rank them on descending order to rank queeires based on how good they can be retrieve with sparse retriever. 

## Sparse vs Hybrid

###### Train
Training the sparse vs hybrid classifier is similar to sparse vs dense classifier, however, since in this seeting, we would retrieve with sparse retriever anyway, we would like to utilie another piece of information i.e., retrieved documents by sparse retrievers. Therefore, we train a similar cross encoder by using queries as well as first retrieved documents by sparse retrievers. The labeling schema and training setting is the same.

Train label for sparse vs hybrid includes the query, first retreievd docuent , and the label, in ```train_label_sparse_vs_hybrid_T50.tsv```

Further, run ``````train_sparse_vs_hybrid.py``````  and the following parameter can be changed for the training  in ```train_sparse_vs_dense.py```.:
*  ```model_name``` : The initial pretrained model can be changed undervariable e.e., ```bert-based-uncased```
*  ```epoch_num``` : number of epochs
*  ```batch_size``` : batch size for training
The model will be saved under ```models``` directory. e.g., ```sparse_vs_hybrid_model_berrt-based-uncased_e1_b32```
If you are not willing to train the model, you can [download the trained sparse vs hybrid model from here]()

###### Test
we test our trained sparse vs dense classifier on MSMARCO small dev set queriees adn theri first retrieved documents by bm25  (```queries+firstdoc.small.dev.tsv```). Run ```test_sparse_vs_hybrid.py``` and the trained model can be changed under ```model_name```. The results should be saved under ```results``` repository as ```prediction_sparse_vs_hybrid.dev.small.tsv``` in the following format:
```qid<\t>query<\t>sparse_prob<\t>hybrid_prob```
Based on the sparse vs hybrid classifier prediction, the query should be retrieved by the retriever with higher probability.

