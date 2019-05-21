import numpy as np
import pickle
import torch.nn as nn
import torch
from itertools import product
from dna2vec.dna2vec.multi_k_model import MultiKModel
from statistics import mode 

class DnaLoad:

	def __init__(self, file_name, dna2vec=False): 
		self.input_file = file_name
		self.f = None
		self.dna2vec = dna2vec

		if (self.dna2vec == True):
			self.init_dna2vec()

		self.read_from_file()
		self.sequence = np.empty(len(self.f), dtype=object)
		self.lab = np.empty(len(self.f), dtype=object)
		self.generate_sequence()
	
	def init_dna2vec(self):
		print("Loading dna2vec matrix")
		self.filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
		self.mk_model = MultiKModel(self.filepath)
		print("Done loading .w2v")

	def read_from_file(self):
		self.f = open(self.input_file, 'r').readlines()

	def generate_sequence(self):
		for i in range(len(self.f)):
			self.sequence[i] = self.f[i].split(';')[1]
			self.lab[i] = self.f[i].split(';')[3]

	def get_labels(self, k, stride, genome_number):
		l = self.windows(k, stride, self.lab[genome_number])
		arr = np.empty(len(l))
		for i in range(len(l)):
			lab_ = self.split(l[i])
			lab = list(map(int, lab_))
			overall_lab = mode(lab)
			arr[i] = overall_lab
		return arr


	def split(self, l): 
		return [char for char in l] 

	def windows(self, k, stride, fseq):
		list_of_k_mers = []
		for i,seq in enumerate(self.window_samples(k, stride, fseq)):
			list_of_k_mers.append(seq)
		return list_of_k_mers


	def window_samples(self, k, stride, fseq):
		if len(fseq)>2*k:
			for i in range(0, len(fseq) - k + 1, stride):
				yield fseq[i:i+k]

	def get_kmer(self, k, stride, genome_number, embedding=None, embed_size=None):
		kmers = np.empty(int(((len(self.sequence[genome_number])-k)/stride)+1), dtype=object)
		for i, seq in enumerate(self.window_samples(k, stride, self.sequence[genome_number])):
				kmers[i] = seq
		if(embedding=='dna2vec'):
			self.init_dna2vec()
			embeddings = torch.ones([len(kmers), 100], dtype=torch.float64)
			for i, kmer in enumerate(kmers):
				embed = self.mk_model.vector(kmer)
				embeddings[i] = torch.tensor(embed, dtype=torch.float64)
			return embeddings
		elif(embedding=='kmer_embed'):
			assert embed_size != None, "When choosing kmer_embed, also provide an embed_size argument (int)"
			kmer_dict = self.generate_kmer_dict(k)
			vocab_size = len(kmer_dict)
			kmer2embedding = nn.Embedding(vocab_size, embed_size)
			print("embedding...")
			embedded_kmers = torch.ones([len(kmers), embed_size], dtype=torch.float64)
			for i, kmer in enumerate(kmers):
				kmer_ = torch.tensor([kmer_dict[kmer]], dtype=torch.long)
				embedded_kmers[i] = kmer2embedding(kmer_)
			return embedded_kmers
		else:
			return kmers
	

	def generate_kmer_dict(self, window_size):
		letters = 'AGCT'
		vocab = [''.join(i) for i in product(letters, repeat = window_size)]
		kmers_dict = {}
		for i in range(len(vocab)):
			kmers_dict[vocab[i]] = i
		return kmers_dict

