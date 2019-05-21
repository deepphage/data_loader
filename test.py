#####################################
# 			EXAMPLE USE 			#
#####################################

from data_loader import PhageLoader
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

loader = PhageLoader("data/input_file.txt")
dataset = loader.get_kmers(3,2,1)
train_loader = DataLoader(dataset=dataset, batch_size=5)

for i, data in enumerate(train_loader, 0):
	X, y = data



