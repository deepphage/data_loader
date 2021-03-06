import torch
from os import listdir
import re
from itertools import product
import numpy as np
from os.path import isfile, join
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from data_loader.DnaLoad import DnaLoad

from dna2vec.dna2vec.multi_k_model import MultiKModel

class DnaDataSet(Dataset):

	def __init__(self, reads, read_labels):
		self.reads = reads
		self.read_labels = read_labels
		
	def __len__(self):
		return len(self.read_labels)


	def __getitem__(self, index):
		X = self.reads[index]
		y = self.read_labels[index]
		return X, y

class PhageFileLoader():
	def __init__(self, file):
		self.file = file
		self.dna_load = DnaLoad(file)

	def get_kmers_for_read(self, k, stride, read_n, read_length=100, embedding=None, embed_size=None):
		kmers = self.dna_load.get_kmer(k, stride, read_n, embedding, embed_size)
		labels = self.dna_load.get_labels(k, stride, read_n)
		# split into reads of length read_length
		reads = kmers[:read_length * (kmers.shape[0] // read_length)].reshape(-1, read_length)
		read_labels = labels[:read_length * (labels.shape[0] // read_length)].reshape(-1, read_length)
		return DnaDataSet(reads, read_labels)

	def get_kmers_for_read_w_id(self, k, stride, read_n, read_length=100, embedding=None, embed_size=None):
		kmers = self.dna_load.get_kmer(k, stride, read_n, embedding, embed_size)
		labels = self.dna_load.get_labels(k, stride, read_n)
		k_id = self.dna_load.get_k_id(read_n)
		# split into reads of length read_length
		reads = kmers[:read_length * (kmers.shape[0] // read_length)].reshape(-1, read_length)
		read_labels = labels[:read_length * (labels.shape[0] // read_length)].reshape(-1, read_length)
		return (DnaDataSet(reads, read_labels), k_id)

	def get_n_loaders(self, n='all', read_length=100, batch_size=32, k=3, stride=1, embedding=None, embed_size=None):
		loaders = []
		if(n=="all"):
			n = self.dna_load.get_number_genomes()
		datasets = np.empty(n, dtype=object)
		for i in range(n):
			dataset = self.get_kmers_for_read(k, stride, i, read_length, embedding=embedding, embed_size=embed_size)
			datasets[i] = dataset
		d = ConcatDataset(datasets)
		return d

	def get_n_loaders_w_id(self, n='all', read_length=100, batch_size=32, k=3, stride=1, embedding=None, embed_size=None):
		loaders = []
		if(n=="all"):
			n = self.dna_load.get_number_genomes()
		k_ids = []
		datasets = np.empty(n, dtype=object)
		for i in range(n):
			(dataset, k_id) = self.get_kmers_for_read_w_id(k, stride, i, read_length, embedding=embedding, embed_size=embed_size)
			_k_ids = [k_id]*len(dataset)
			k_ids.extend(_k_ids)
			datasets[i] = dataset
		d = ConcatDataset(datasets)
		# print("Lenghts of 1 file")
		# print(len(d))
		# print(len(k_ids))
		return (d, k_ids)
	

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

	def get_data_loader(self, n_files='all' ,n='all', read_length=100, batch_size=32, k=3, stride=1, embedding=None, embed_size=None, drop_last=False):
		if(n_files=='all'):
			n_files = len(self.filenames)
		datasets = np.empty(n_files, dtype=object)
		for i in range(n_files):
			dl = PhageFileLoader(self.datafolder+self.filenames[i])
			dataset = dl.get_n_loaders(n=n,read_length=read_length, batch_size=batch_size,k=k,stride=stride,embedding=embedding,embed_size=embed_size)
			datasets[i] = dataset
		d = ConcatDataset(datasets)
		return DataLoader(dataset=d, batch_size=batch_size, drop_last=drop_last)


	def get_data_set(self, n_files='all' ,n='all', read_length=100, batch_size=32, k=3, stride=1, embedding=None, embed_size=None, drop_last=False):
		if(n_files=='all'):
			n_files = len(self.filenames)
		datasets = np.empty(n_files,  dtype=object)
		for i in range(n_files):
			dl = PhageFileLoader(self.datafolder+self.filenames[i])
			dataset = dl.get_n_loaders(n=n,read_length=read_length, batch_size=batch_size,k=k,stride=stride,embedding=embedding,embed_size=embed_size)
			datasets[i] = dataset
		d = ConcatDataset(datasets)
		return d

	def get_data_set_ids(self, n_files='all' ,n='all', read_length=100, batch_size=32, k=3, stride=1, embedding=None, embed_size=None, drop_last=False):
		if(n_files=='all'):
			n_files = len(self.filenames)
		datasets = np.empty(n_files,  dtype=object)
		k_ids = []
		for i in range(n_files):
			dl = PhageFileLoader(self.datafolder+self.filenames[i])
			(dataset, k_id) = dl.get_n_loaders_w_id(n=n,read_length=read_length, batch_size=batch_size,k=k,stride=stride,embedding=embedding,embed_size=embed_size)
			k_ids.extend(k_id)
			datasets[i] = dataset
		d = ConcatDataset(datasets)
		# print(len(d))
		# print(len(k_ids))
		return k_ids

	def get_dict(self, window_size, embedding='dict'):
		letters = 'AGCT'
		vocab = [''.join(i) for i in product(letters, repeat = window_size)]
		kmers_dict = {}
		for i in range(len(vocab)):
			kmers_dict[vocab[i]] = i
		if(embedding=='dna2vec'):
			print("loading DNA 2 vec model")
			filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
			mk_model = MultiKModel(filepath)
			for kmer in kmers_dict.keys():
				kmers_dict[kmer] = mk_model.vector(kmer)
		return kmers_dict












