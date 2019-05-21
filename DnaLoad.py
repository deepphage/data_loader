import numpy as np
import pickle
from dna2vec.dna2vec.multi_k_model import MultiKModel

class DnaLoad:

	def __init__(self, file_name, dna2vec=True): 
		self.input_file = file_name
		self.f = None
		self.dna2vec = dna2vec
		# self.k = k
		# self.stride = stride
		if (self.dna2vec == True):
			self.filepath = 'dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v'
			self.mk_model = MultiKModel(self.filepath)

		self.read_from_file()
		self.sequence = np.empty(len(self.f), dtype=object)
		self.lab = np.empty(len(self.f), dtype=object)
		self.generate_sequence()

	def read_from_file(self):
		self.f = open(self.input_file, 'r').readlines()

	def generate_sequence(self):
		for i in range(len(self.f)):
			self.sequence[i] = self.f[i].split(';')[1]
			self.lab[i] = self.f[i].split(';')[3]

	def get_labels(self, k, stride, genome_number):
		l = self.windows(k, stride, self.sequence[genome_number])
		arr = np.zeros(len(l))
		for i in range(len(l)):
			ones = 0
			for j in range(len(l[i])):
				if l[i][j] == '1':
					ones+=1
			if ones>=2:
				arr[i] = 1
		return arr


	def windows(self, k, stride, fseq):
		list_of_k_mers = []
		for seq in self.window_samples(k, stride, fseq):
			list_of_k_mers.append(seq)
		return list_of_k_mers


	def window_samples(self, k, stride, fseq):
		if len(fseq)>2*k:
			for i in range(0, len(fseq) - k + 1, stride):
				yield fseq[i:i+k]

	def get_kmer(self, k, stride, genome_number, embedding=None):
		kmers = np.empty(int(((len(self.sequence[genome_number])-k)/stride)+1), dtype=object)
		for i, seq in enumerate(self.window_samples(k, stride, self.sequence[genome_number])):
				kmers[i] = seq
		if(embedding=='dna2vec'):
			embeddings = np.empty([len(kmers), 100])
			for i, kmer in enumerate(kmers):
				embed = self.mk_model.vector(kmer)
				embeddings[i] = embed
			return embeddings
		else:
			return kmers


# dna_load = DnaLoad("data/input_file.txt")
# kmers = dna_load.get_kmer(3, 2, 0)
# labels = dna_load.get_labels(3, 2, 0)
# print(kmers[len(kmers)-1])


