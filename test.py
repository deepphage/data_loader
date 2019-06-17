#####################################
# 			EXAMPLE USE 			#
#####################################

from data_loader.data_loader import PhageLoader
from torch.utils.data import DataLoader

'''
get_kmer parameters in order:

1) k -> size of window
2) stride
3) genome number in .txt file
4) embedding: OPTIONAL (String) {'dna2vec', 'kmer_embed'}
5) if you chose embedding of type kmer_embed, 
   then you need to pass also the embedding size: embed_size 

example use: dataset = loader.get_kmers(3,2,1, embedding='kmer_embed', embed_size=5)	
'''


loader = PhageLoader("data/")
l = loader.get_data_set(n_files=2, n='all',read_length=100, batch_size=32, k=3, stride=1, embedding="dict", embed_size=None, drop_last=False)
ids = loader.get_data_set_ids(n_files=2, n='all',read_length=100, batch_size=32, k=3, stride=1, embedding="dict", embed_size=None, drop_last=False)

print(ids[25])



print(len(l))
# for i, data in enumerate(loader, 0):
# 	X, y = data