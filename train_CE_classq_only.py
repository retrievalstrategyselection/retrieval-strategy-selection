from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging
#### Just some code to print debug information to stdout

with open('train_sparse_vs_dense_50only_sparse.pkl', 'rb') as f:
    q_map_first_doc_train=pickle.load(f)


train_set=[]

for key in q_map_first_doc_train:
    if "first_doc_text" in q_map_first_doc_train[key].keys():
        qtext=q_map_first_doc_train[key]["query_text"]
        label_=q_map_first_doc_train[key]["label"]
        train_set.append( InputExample(texts=[qtext],label=label_ ))
        
print(len(train_set))

batch_size=32
# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)


# We add an evaluator, which evaluates the performance during training

epoch_num=2

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1) #10% of train data for warm-up
model_name='bert-base-uncased'

model = CrossEncoder(model_name, num_labels=2)

# Train the model
model.fit(train_dataloader=train_dataloader,
          epochs=epoch_num,
          warmup_steps=warmup_steps,
          output_path="models/only_Q_tuned_model_"+model_name+"_e"+str(epoch_num)+'_b'+str(batch_size))


model.save("models/only_Q_tuned_model_"+model_name+"_"+str(epoch_num)+'_b'+str(batch_size))

with open('test_clas.pkl', 'rb') as f:
    q_map_first_doc_test=pickle.load(f)

sentences = []
map_value_test=[]
qs=[]
for key in q_map_first_doc_test:
    if "first_doc" in q_map_first_doc_test[key].keys():
        qs.append(key)
        sentences.append([q_map_first_doc_test[key]["query_text"],q_map_first_doc_test[key]["first_doc_text"]])
        map_value_test.append(q_map_first_doc_test[key]["label"])

scores=model.predict(sentences)

actual=[]
predicted=[]
out=open('out','w')
for i in range(len(sentences)):
    out.write(qs+'\t'+str((scores[i]))+'\n')
    actual.append(map_value_test[i])
    predicted.append(float(scores[i]))
out.close()