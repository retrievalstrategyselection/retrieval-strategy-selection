import pickle ,sys
from scipy.stats import kendalltau,pearsonr
from scipy.stats import kendalltau,pearsonr

sys.path.append('/home/ir-bias/Negar/Sbert/sentence-transformers')

from sentence_transformers.cross_encoder import CrossEncoder
for k in ['50','100','150','200']:
    trained_model_name="new_labels_tuned_model_sparse_only_"+k+"_bert-base-uncased_1_b16"
    model = CrossEncoder("models/"+trained_model_name, num_labels=1)

    with open('test_class_sparse_vs_dense_only_sparse.pkl', 'rb') as f:
        q_map_first_doc_test=pickle.load(f)

    sentences = []
    map_value_test=[]
    queries=[]
    for key in q_map_first_doc_test:
        sentences.append([q_map_first_doc_test[key]["query_text"],q_map_first_doc_test[key]["first_doc_text"]])
        queries.append(key)

    scores=model.predict(sentences)


    actual=[]
    predicted=[]
    out=open('results/new_labels_trained_only_sparse'+trained_model_name,'w')
    for i in range(len(sentences)):
        predicted.append(float(scores[i]))
        out.write(queries[i]+'\t'+str(predicted[i])+'\n')
    out.close()
