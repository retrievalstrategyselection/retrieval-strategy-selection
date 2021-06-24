from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging

train_set=[]


labels=open('train_labels_T50.tsv','r').readlines()
for line in labels:
    qtext,label=line.rstrip().split('\t')
    train_set.append( InputExample(texts=[qtext],label=int(label) ))

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


model.save("models/sparse_vs_dense_model_"+model_name+"_"+str(epoch_num)+'_b'+str(batch_size))
