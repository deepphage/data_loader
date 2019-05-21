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

	def get_kmers(self, k, stride, genome_n, embedding=None, embed_size=None):
		kmers = self.dna_load.get_kmer(k, stride, genome_n, embedding, embed_size)
		labels = self.dna_load.get_labels(k, stride, genome_n)
		return DnaDataSet(kmers, labels)