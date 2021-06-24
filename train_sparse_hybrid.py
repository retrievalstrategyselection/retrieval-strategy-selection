from sentence_transformers import SentenceTransformer, InputExample, losses, util,evaluation
from torch.utils.data import DataLoader
import pickle 
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
import math,logging
#### Just some code to print debug information to stdout
for k in ['50','100','150','200']:
    with open('new_labels_train_sparse_vs_dense_'+k+'only_sparse.pkl', 'rb') as f:
        dic_class=pickle.load(f)

    train_set=[]

    for key in dic_class:
        if "first_doc_text" in dic_class[key].keys():
            qtext=dic_class[key]["query_text"]
            firstdoctext=dic_class[key]["first_doc_text"]
            train_set.append( InputExample(texts=[qtext,firstdoctext],label=dic_class[key]['label'] ))
            
    print(len(train_set))

    batch_size=16
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size)

    epoch_num=1

    warmup_steps = math.ceil(len(train_dataloader) * epoch_num * 0.1) #10% of train data for warm-up
    model_name='bert-base-uncased'

    model = CrossEncoder(model_name, num_labels=1)
    model_name="models/new_labels_tuned_model_sparse_only_"+k+'_'+model_name+"_"+str(epoch_num)+'_b'+str(batch_size)
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              epochs=epoch_num,
              warmup_steps=warmup_steps,
              output_path=model_name)

    model.save(model_name)
