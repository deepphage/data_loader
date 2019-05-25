
loader = PhageLoader("data/")
loader = loader.get_data_set(n='all',read_length=100, batch_size=32, k=3, stride=1, embedding="dict", embed_size=None, drop_last=False)
print(loader)
# for i, data in enumerate(loader, 0):
# 	X, y = data
# 	# one X is one batch of size 32 in this case
# 	# each element of the bacth is a sequence of length read_length
# 	print(X[0]) # <- this is a sequence of length 100 



