import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

# The data path is at:
PATH = './data/600196.csv'
STEP = 60

# Try to build the dataset us torch.dataset
from torch.utils.data import Dataset, DataLoader

class StockData (Dataset):
	def __init__ (self, path:str, step:int = 30):
		self.path = path
		self.step = step

		data = pd.read_csv(path).values[:,1:]

		self.len = len(data)

		self.y_max = data[:,3].max()
		self.y_min = data[:,3].min()

		# visualise the data
		data = self.normalise(data)+1e-5

		# print(data[:-1,:])
		self.X = torch.tensor(data[:-1,:].astype(np.float32))
		self.y = torch.tensor(data[1:,3].astype(np.float32))

		# plt.plot(self.y)
		plt.show()


	def __getitem__ (self, index):
		return self.X[index:index+self.step], self.y[index+self.step]

	def __len__ (self):
		return self.len-1-self.step

	def normalise (self, data):
		data = data.T
		for i in range(len(data)):
			data_min = data[i].min()
			data_max = data[i].max()
			data[i] = (data[i] - data_min) / (data_max - data_min)
		return data.T


stock_data = StockData(path = PATH, step = STEP)
data = DataLoader(dataset = stock_data, batch_size = 5, shuffle = False)


# now let us write a LSTM model
import torch.nn as nn
import torch.nn.functional as F

class Net (nn.Module):
	
	def __init__ (self, input_size = 6, hidden_size = 20, output_size = 1, layers = 3):
		super().__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first = True, bidirectional = True)
		self.linear = nn.Linear(hidden_size*2, output_size)
		self.function = torch.sigmoid

	def forward(self, X):
		X, hidden = self.lstm(X, None)
		X = X[:,-1,:]
		X = self.linear(X)
		X = self.function(X)
		return X

net = Net()

# # X,y = iter(data).next()
# # # print(X)
# # print(net(X))



# now Train the model
import torch.optim as op

# LR = 0.01
criteria = nn.MSELoss()
optimiser = op.Adam(net.parameters())
EPCHO = 10


for epcho in range(EPCHO):
	for i, Xy in enumerate (data):

		if i == 200: break

		X = Xy[0]
		y = Xy[1]

		predict = net(X)
		optimiser.zero_grad()
		loss = criteria(predict,y)
		loss.backward()
		optimiser.step()

	print('Epcho: {}.......... loss is {}'.format(epcho,loss))

torch.save(net, 'stock_predict_15.pkl')



# net = torch.load('stock_predict_15.pkl')

predict = np.array([])
actual = np.array([])

torch.no_grad() 


for X,y in data:
	predict = np.append(predict, net(X).data[0,0])
	actual = np.append(actual, y.data[0])


plt.plot(predict, label = 'prediction')
plt.plot(actual, label = 'actual')
plt.title('stock_predict step = '+ str(STEP))
plt.legend()
plt.show()














