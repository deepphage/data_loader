import torch
from os import listdir
import re
import numpy as np
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader
from data_loader.DnaLoad import DnaLoad

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

class PhageFileLoader():
	def __init__(self, file):
		self.file = file
		self.dna_load = DnaLoad(file)

	def get_kmers_for_read(self, k, stride, read_n, embedding=None, embed_size=None):
		kmers = self.dna_load.get_kmer(k, stride, read_n, embedding, embed_size)
		labels = self.dna_load.get_labels(k, stride, read_n)
		return DnaDataSet(kmers, labels)

	def get_n_loaders(self, n='all', batch_size=32, k=3, stride=1, embedding=None, embed_size=None, drop_last=False):
		loaders = []
		if(n=="all"):
			n = self.dna_load.get_number_genomes()
		for i in range(n):
			dataset = self.get_kmers_for_read(k, stride, i, embedding, embed_size)
			loaders.append(DataLoader(dataset=dataset, batch_size=batch_size, drop_last=drop_last))
		return loaders


# loader = PhageLoader("data/input_file.txt")
# dataLoaders = loader.get_n_loaders(3,32,3,2, embedding="dict", embed_size=None)
# train_loader = dataLoaders[0]
# for i, data in enumerate(train_loader, 0):
# 	X, y = data

# loader = PhageFileLoader("data/Input_NC_013693.txt")
# dataset = loader.get_kmers_for_read(3,2,0, embedding=None, embed_size=None)
# train_loader = DataLoader(dataset=dataset, batch_size=32)
# for i, data in enumerate(train_loader, 0):
# 	X, y = data
# 	print(y)


class PhageLoader():
	def __init__(self, datafolder):
		self.datafolder = datafolder
		self.filenames = self.get_file_names()
		self.phageLoaders = np.empty(len(self.filenames), dtype=object)

	def get_file_names(self):
		files = listdir(self.datafolder)
		regex = re.compile('Input*')
		selected_files = list(filter(regex.search, files))
		return selected_files

	def get_data_loaders(self):
		for i in range(len(self.filenames)):
			dl = PhageFileLoader(self.datafolder+self.filenames[i])
			self.phageLoaders[i] = dl
		return self.phageLoaders










