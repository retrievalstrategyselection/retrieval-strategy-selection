from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging
#### Just some code to print debug information to stdout


batch_size=32

model_name="sparse_vs_dense_bert-base-uncased_e2_b32"
model = CrossEncoder('models/'+model_name, num_labels=2)
sentences = []
qs=[]

labels=open('queries.dev.small.tsv','r').readlines()
for line in labels:
    qid,qtext=line.rstrip().split('\t')
    sentences.append([qtext])
    qs.append(qid)
    

scores=model.predict(sentences)
actual=[]
out=open('results/prediction_'+model_name+'.dev.small.tsv','w')
out.write('qid\tquery\tsparse_prob\tdense_prob\n')
for i in range(len(sentences)):
    out.write(qs[i]+'\t'+sentences[i][0]+'\t'+str(scores[i][0])+'\t'+str(scores[i][1])+'\n')
out.close()
