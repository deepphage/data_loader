from dna2vec.dna2vec.multi_k_model import MultiKModel
import numpy as np
import pickle
from DnaLoad import DnaLoad

k = 6
stride = 3

filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
mk_model = MultiKModel(filepath)

dna_load = DnaLoad("data/input_file.txt", 6, 3, n_genomes=1)
kmers = dna_load.get_kmer(0)
labels = dna_load.get_labels(0)


embeddings = np.empty([len(kmers), 100])
for i, kmer in enumerate(kmers):
	embed = mk_model.vector(kmer)
	embeddings[i] = embed


with open('data/dna2vec/embedding_'+str(k)+'_'+str(stride)+'.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/dna2vec/embedding_'+str(k)+'_'+str(stride)+'_labels.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)






