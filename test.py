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

# loader = PhageLoader("data/input_file.txt")
# dataset = loader.get_kmers(3,2,0, embedding='kmer_embed', embed_size=5)
# train_loader = DataLoader(dataset=dataset, batch_size=32)

# print(len(train_loader))
# for i, data in enumerate(train_loader, 0):
# 	X, y = data


loader = PhageLoader("data/")
loaders = loader.get_data_loaders()
# for each file 
for l in loaders:
	# if you want to get all genomes use "All"
	dataLoaders = l.get_n_loaders("all",32,3,2, embedding="dict", embed_size=None)
	print(len(dataLoaders))
	for dl in dataLoaders:
		for i, data in enumerate(dl, 0):
			X, y = data



