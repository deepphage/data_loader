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

dna_load = DnaLoad("data/input_file.txt")
kmers = dna_load.get_kmer(3, 2, 1)
labels = dna_load.get_labels(3, 2, 1)

dataset = DnaDataSet(kmers, labels)
train_loader = DataLoader(dataset=dataset, batch_size=100)
for i, data in enumerate(train_loader, 0):
	X, y = data
	print(X)







