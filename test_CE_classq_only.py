from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging
#### Just some code to print debug information to stdout


batch_size=32


model = CrossEncoder('models/only_Q_tuned_model_bert-base-uncased_2_b32', num_labels=2)




with open('test_class_sparse_vs_dense_only_sparse.pkl', 'rb') as f:
    q_map_first_doc_test=pickle.load(f)

sentences = []
map_value_test=[]
qs=[]
for key in q_map_first_doc_test:
    qs.append(key)
    sentences.append([q_map_first_doc_test[key]["query_text"]])

scores=model.predict(sentences)

actual=[]
out=open('out_onlyq_e2','w')
for i in range(len(sentences)):
    if scores[i][0]> scores[i][1]:
        out.write(qs[i]+'\t'+str((scores[i]))+'\t0\n')
    else:
        out.write(qs[i]+'\t'+str((scores[i]))+'\t1\n')
out.close()