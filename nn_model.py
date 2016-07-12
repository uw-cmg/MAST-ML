import neurolab as nl
import configuration_parser
import numpy as np
import ast

class model():
	def  __init__(self,net,train='bfgs',epochs=500,show=False,goal=0.1):
		self.epochs = epochs
		self.show = show
		self.goal = goal
		self.net = net
		self.trainAlgorithm = train

	def predict(self,x_test):
		return self.net.sim(np.array(x_test))

	def fit(self, x_train, y_train):
		exec('self.net.trainf = nl.train.train_' + self.trainAlgorithm)
		self.net.train(np.array(x_train), np.array(y_train).reshape(len(y_train), 1),epochs=self.epochs,show=self.show,goal=self.goal)


def get():
	config = configuration_parser.parse()
	minmax = ast.literal_eval(config.get(__name__, 'minmax'))
	size = ast.literal_eval(config.get(__name__, 'size'))
	epochs = config.getint(__name__, 'epochs')
	show = config.getboolean(__name__, 'show')
	goal = config.getfloat(__name__, 'goal')
	exec('transf = [nl.trans.' + config.get(__name__, 'transfer_function') + '()]*len(size)',locals(),globals())
	train = config.get(__name__, 'training_algorithm')
	return model(nl.net.newff(minmax,size,transf),train,epochs,show,goal)
