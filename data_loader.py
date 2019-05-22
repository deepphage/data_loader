import torch
from torch.utils.data import Dataset, DataLoader
from DnaLoad import DnaLoad

class DnaDataSet(Dataset):

	def __init__(self, kmers, labels):
		self.kmers = kmers
		self.labels = labels
		
	def __len__(self):
		return len(self.labels)


	def __getitem__(self, index):
		X = self.kmers[index]
		y = self.labels[index]
		return X, y

class PhageLoader():
	def __init__(self, file):
		self.file = file
		self.dna_load = DnaLoad(file)

	def get_kmers_for_read(self, k, stride, read_n, embedding=None, embed_size=None):
		kmers = self.dna_load.get_kmer(k, stride, read_n, embedding, embed_size)
		labels = self.dna_load.get_labels(k, stride, read_n)
		return DnaDataSet(kmers, labels)

	def get_n_loaders(self, n, batch_size, k, stride, embedding=None, embed_size=None):
		loaders = []
		for i in range(n):
			dataset = self.get_kmers_for_read(k, stride, i, embedding, embed_size)
			loaders.append(DataLoader(dataset=dataset, batch_size=batch_size))
		return loaders


# loader = PhageLoader("data/input_file.txt")
# dataLoaders = loader.get_n_loaders(3,32,3,2, embedding="dict", embed_size=None)
# train_loader = dataLoaders[0]
# for i, data in enumerate(train_loader, 0):
# 	X, y = data

loader = PhageLoader("data/input_file_reads.txt")
dataset = loader.get_kmers_for_read(3,2,0, embedding=None, embed_size=None)
train_loader = DataLoader(dataset=dataset, batch_size=32)
for i, data in enumerate(train_loader, 0):
	X, y = data
	print(y)