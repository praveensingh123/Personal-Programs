import numpy as np
np.random.seed(1234)
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
mnist = fetch_mldata('MNIST original')
# prepare data
N = 70000
data = np.float32(mnist.data[:]) / 255.
idx = np.random.choice(data.shape[0], N)
data = data[idx]
target = np.int32(mnist.target[idx]).reshape(N, 1)

train_idx, test_idx = train_test_split(np.array(range(N)), test_size=0.10)
train_data, test_data = data[train_idx], data[test_idx]
train_target, test_target = target[train_idx], target[test_idx]

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False).fit_transform(train_target))

# inputs
n_input = train_data.shape[1]
M = train_data.shape[0]

batch_size = 100 # 4
num_batches = train_data.shape[0]//batch_size


def sigmoid(x):
	return 1./(1+np.exp(-x))

def relu(x):
	x[x<0.0] = 0.0
	return x

def deriv_relu(x):
	x[x>0.0] = 1.0
	return x

def tanh(x):
	return np.tanh(x)

def deriv_tanh(x):
	return (1-x**2)

def py_cal(mu_1, sigma_1, eps_1, mu_2, sigma_2, eps_2, mu_3, sigma_3, eps_3, 
	b_b1_mu, b_b1_logsigma, b_epsilon_b1,b_b2_mu, b_b2_logsigma, b_epsilon_b2,
	b_b3_mu, b_b3_logsigma, b_epsilon_b3, x_in, y_output, batch_size, n_batches, lr, b, episode_num):
	weight_1 = mu_1 + np.log(1. + np.exp(sigma_1))*eps_1
	weight_2 = mu_2 + np.log(1. + np.exp(sigma_2))*eps_2
	weight_3 = mu_3 + np.log(1. + np.exp(sigma_3))*eps_3
	b_weight_1 = b_b1_mu + np.log(1+np.exp(b_b1_logsigma))*b_epsilon_b1
	b_weight_2 = b_b2_mu + np.log(1+np.exp(b_b2_logsigma))*b_epsilon_b2
	b_weight_3 = b_b3_mu + np.log(1+np.exp(b_b3_logsigma))*b_epsilon_b3
	z1 = np.dot(x_in, weight_1) + b_weight_1
	h1 = relu(z1)
	z2 = np.dot(h1, weight_2) + b_weight_2 
	h2 = relu(z2)
	z3 = np.dot(h2, weight_3) + b_weight_3
	z3 = np.exp(z3)/np.sum(np.exp(z3), axis=1, keepdims=True)

    
	# likelihood
	delta3 = -1.0*(1./x_in.shape[0])*(y_output-z3)/(np.exp(-3)**2) # we need to check (y-h) or (h-y)
	delta_b3_mu = np.sum(delta3, axis=0)
	delta_b3_logsigma = np.sum((delta3)*(1/(1+np.exp(-b_b3_logsigma)))*b_epsilon_b3, axis=0)
	delta_w3_mu = np.dot(h2.T, delta3)
	delta_w3_logsigma = np.dot(h2.T, delta3)*(1/(1+np.exp(-sigma_3)))*eps_3
	delta2 = np.dot(delta3, weight_3.T)*deriv_relu(h2)
	delta_b2_mu = np.sum(delta2, axis=0)
	delta_b2_logsigma = np.sum((delta2)*(1/(1+np.exp(-b_b2_logsigma)))*b_epsilon_b2, axis=0)
	delta_w2_mu = np.dot(h1.T, delta2)
	delta_w2_logsigma = np.dot(h1.T, delta2)*(1/(1+np.exp(-sigma_2)))*eps_2
	delta1 = np.dot(delta2, weight_2.T)*deriv_relu(h1)
	delta_b1_mu = np.sum(delta1, axis=0)
	delta_b1_logsigma = np.sum((delta1)*(1/(1+np.exp(-b_b1_logsigma)))*b_epsilon_b1, axis=0)
	delta_w1_mu = np.dot(x_in.T, delta1)
	delta_w1_logsigma = np.dot(x_in.T, delta1)*(1/(1+np.exp(-sigma_1)))*eps_1
	# prior 
	# wegihts
	w_gaussian_prior_mu_1 = -weight_1/(0.05**2)
	w_gaussian_prior_sigma_1 = (-weight_1/(0.05**2))* eps_1*(1./(1.+np.exp(-sigma_1)))
	w_gaussian_prior_mu_2 = -weight_2/(0.05**2)
	w_gaussian_prior_sigma_2 = (-weight_2/(0.05**2))* eps_2*(1./(1.+np.exp(-sigma_2)))
	w_gaussian_prior_mu_3 = -weight_3/(0.05**2)
	w_gaussian_prior_sigma_3 = (-weight_3/(0.05**2))* eps_3*(1./(1.+np.exp(-sigma_3)))
	# biases
	b_gaussian_prior_mu_1 = -b_weight_1/(0.05**2)
	b_gaussian_prior_sigma_1 = (-b_weight_1/(0.05**2))* b_epsilon_b1*(1./(1.+np.exp(-b_b1_logsigma)))
	b_gaussian_prior_mu_2 = -b_weight_2/(0.05**2)
	b_gaussian_prior_sigma_2 = (-b_weight_2/(0.05**2))* b_epsilon_b2*(1./(1.+np.exp(-b_b2_logsigma)))
	b_gaussian_prior_mu_3 = -b_weight_3/(0.05**2)
	b_gaussian_prior_sigma_3 = (-b_weight_3/(0.05**2))* b_epsilon_b3*(1./(1.+np.exp(-b_b3_logsigma)))
	
	# variational posterior only sigmas
	# sigmas for weights (mus for weights is zero in the case of Variational Posterior)
	'''
	varitional_sigma_1 = ((weight_1-mu_1)**2/(np.exp(2*sigma_1)))-1.0-(((weight_1-mu_1)/(np.exp(2*sigma_1)))*(1/(1+np.exp(-sigma_1)))*eps_1)
	varitional_sigma_2 = ((weight_2-mu_2)**2/(np.exp(2*sigma_2)))-1.0-(((weight_2-mu_2)/(np.exp(2*sigma_2)))*(1/(1+np.exp(-sigma_2)))*eps_2)
	varitional_sigma_3 = ((weight_3-mu_3)**2/(np.exp(2*sigma_3)))-1.0-(((weight_3-mu_3)/(np.exp(2*sigma_3)))*(1/(1+np.exp(-sigma_3)))*eps_3)
	
	# sigmas for bias terms
	b_varitional_sigma_1 = ((b_weight_1-b_b1_mu)**2/(np.exp(2*b_b1_logsigma)))-1.0-(((b_weight_1-b_b1_mu)/(np.exp(2*b_b1_logsigma)))*(1/(1+np.exp(-b_b1_logsigma)))*b_epsilon_b1)
	b_varitional_sigma_2 = ((b_weight_2-b_b2_mu)**2/(np.exp(2*b_b2_logsigma)))-1.0-(((b_weight_2-b_b2_mu)/(np.exp(2*b_b2_logsigma)))*(1/(1+np.exp(-b_b2_logsigma)))*b_epsilon_b2)
	b_varitional_sigma_3 = ((b_weight_3-b_b3_mu)**2/(np.exp(2*b_b3_logsigma)))-1.0-(((b_weight_3-b_b3_mu)/(np.exp(2*b_b3_logsigma)))*(1/(1+np.exp(-b_b3_logsigma)))*b_epsilon_b3) 
	'''
	varitional_sigma_1 = (-1.0/(np.log(1. + np.exp(sigma_1))))*(1/(1+np.exp(-sigma_1)))
	varitional_sigma_2 = (-1.0/(np.log(1. + np.exp(sigma_2))))*(1/(1+np.exp(-sigma_2)))
	varitional_sigma_3 = (-1.0/(np.log(1. + np.exp(sigma_3))))*(1/(1+np.exp(-sigma_3)))
	
	# sigmas for bias terms
	b_varitional_sigma_1 = (-1.0/(np.log(1+np.exp(b_b1_logsigma))))*(1/(1+np.exp(-b_b1_logsigma)))
	b_varitional_sigma_2 = (-1.0/(np.log(1+np.exp(b_b2_logsigma))))*(1/(1+np.exp(-b_b2_logsigma)))
	b_varitional_sigma_3 = (-1.0/(np.log(1+np.exp(b_b3_logsigma))))*(1/(1+np.exp(-b_b3_logsigma)))

	# mus and sigmas updates
	mu_1 = mu_1 - lr*((1./n_batches)*(0-w_gaussian_prior_mu_1)+(delta_w1_mu*x_in.shape[0]))*(1./x_in.shape[0])
	sigma_1 = sigma_1 - lr*((1./n_batches)*(varitional_sigma_1-w_gaussian_prior_sigma_1) + \
		(delta_w1_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	b_b1_mu = b_b1_mu - lr*((1./n_batches)*(0-b_gaussian_prior_mu_1)+(delta_b1_mu*x_in.shape[0]))*(1./x_in.shape[0])
	b_b1_logsigma = b_b1_logsigma - lr*((1./n_batches)*(b_varitional_sigma_1-b_gaussian_prior_sigma_1) + \
		(delta_b1_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	mu_2 = mu_2 - lr*((1./n_batches)*(0-w_gaussian_prior_mu_2)+ (delta_w2_mu*x_in.shape[0]))*(1./x_in.shape[0])
	sigma_2 = sigma_2 - lr*((1./n_batches)*(varitional_sigma_2-b_gaussian_prior_sigma_2) + \
		(delta_w2_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	b_b2_mu = b_b2_mu - lr*((1./n_batches)*(0-b_gaussian_prior_mu_2) + \
		(delta_b2_mu*x_in.shape[0]))*(1./x_in.shape[0])
	b_b2_logsigma = b_b2_logsigma - lr*((1./n_batches)*(b_varitional_sigma_2-b_gaussian_prior_sigma_2) + \
		(delta_b2_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	mu_3 = mu_3 - lr*((1./n_batches)*(0-w_gaussian_prior_mu_3)+(delta_w3_mu*x_in.shape[0]))*(1./x_in.shape[0])
	sigma_3 = sigma_3 - lr*((1./n_batches)*(varitional_sigma_3-w_gaussian_prior_sigma_3) + \
		(delta_w3_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	b_b3_mu = b_b3_mu - lr*((1./n_batches)*(0-b_gaussian_prior_mu_3)+(delta_b3_mu*x_in.shape[0]))*(1./x_in.shape[0])
	b_b3_logsigma = b_b3_logsigma - lr*((1./n_batches)*(b_varitional_sigma_3-b_gaussian_prior_sigma_3) + \
		(delta_b3_logsigma*x_in.shape[0]))*(1./x_in.shape[0])

	# this method returns the new mus and sigmas after updating
	return (mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, b_b1_mu, b_b1_logsigma, b_b2_mu, b_b2_logsigma, b_b3_mu, b_b3_logsigma)

if __name__ == '__main__':

	#train_x = np.linspace(0,2*np.pi, num=2000,endpoint=True).reshape(2000,1)
	#train_y = np.sin(train_x) + np.random.normal(loc=0.0, scale=0.01)

	sigma_prior = np.exp(-3)
	learning_rate = 0.001 # from 0.001, changed from 0.0005 of linear regression
	n_epochs = 50
	n_hidden_1 = 400
	n_hidden_2 = 400
	n_output = 10

	batch_size = 100
	n_batches = int(M / float(batch_size))

	mu_1 = np.random.normal(0, 0.05, size=(n_input, n_hidden_1))
	sigma_1 = np.random.normal(0, 0.05, size=(n_input, n_hidden_1))
	#eps_1 = np.random.normal(0, 0.05, size=(n_input, n_hidden_1))
	mu_2 = np.random.normal(0, 0.05, size=(n_hidden_1, n_hidden_2))
	sigma_2 = np.random.normal(0, 0.05, size=(n_hidden_1, n_hidden_2))
	#eps_2 = np.random.normal(0, 0.05, size=(n_hidden_1, n_hidden_2))
	mu_3 = np.random.normal(0, 0.05, size=(n_hidden_2, n_output))
	sigma_3 = np.random.normal(0, 0.05, size=(n_hidden_2, n_output))
	#eps_3 = np.random.normal(0, 0.05, size=(n_hidden_2, n_output))
	b_b1_mu = np.random.normal(0, 0.05, size=(1, n_hidden_1))
	b_b1_logsigma = np.random.normal(0, 0.05, size=(1, n_hidden_1))
	#b_epsilon_b1 = np.random.normal(0, 0.05, size=(1, n_hidden_1))
	b_b2_mu = np.random.normal(0, 0.05, size=(1, n_hidden_2))
	b_b2_logsigma = np.random.normal(0, 0.05, size=(1, n_hidden_2))
	#b_epsilon_b2 = np.random.normal(0, 0.05, size=(1, n_hidden_2))
	b_b3_mu = np.random.normal(0, 0.05, size=(1, n_output))
	b_b3_logsigma = np.random.normal(0, 0.05, size=(1, n_output))
	#b_epsilon_b3 = np.random.normal(0, 0.05, size=(1, n_output))

for e in range(n_epochs):
	test_accuracy = []
	train_accuracy = []

	for b in range(n_batches):
		x_input = np.reshape(train_data[b*batch_size:(b+1)*batch_size,:], (batch_size,784))
		y_true = np.reshape(train_target[b*batch_size:(b+1)*batch_size,:],(batch_size,10))
		eps_1 = np.random.normal(0, 0.05, size=(n_input, n_hidden_1))
		eps_2 = np.random.normal(0, 0.05, size=(n_hidden_1, n_hidden_2))
		eps_3 = np.random.normal(0, 0.05, size=(n_hidden_2, n_output))
		b_epsilon_b1 = np.random.normal(0, 0.05, size=(1, n_hidden_1))
		b_epsilon_b2 = np.random.normal(0, 0.05, size=(1, n_hidden_2))
		b_epsilon_b3 = np.random.normal(0, 0.05, size=(1, n_output))
		mu_1, sigma_1, mu_2, sigma_2, mu_3, sigma_3, b_b1_mu, b_b1_logsigma, b_b2_mu, b_b2_logsigma, b_b3_mu, \
        b_b3_logsigma = py_cal(mu_1, sigma_1, eps_1, mu_2, sigma_2, eps_2, mu_3, sigma_3, eps_3, b_b1_mu, \
                               b_b1_logsigma, b_epsilon_b1, b_b2_mu, b_b2_logsigma, b_epsilon_b2, b_b3_mu, b_b3_logsigma, b_epsilon_b3, x_input, y_true, batch_size, n_batches, learning_rate, b, e)

	if e%1 == 0.0:
		z1 = np.dot(train_data, mu_1) + b_b1_mu
		h1 = relu(z1)
		z2 = np.dot(h1, mu_2) + b_b2_mu 
		h2 = relu(z2)
		z3 = np.dot(h2, mu_3) + b_b3_mu
		out_1 = np.exp(z3)/np.sum(np.exp(z3), axis=1, keepdims=True)      

		soft_arg_max_1 = np.asarray(np.argmax(out_1, axis=1))
		test_numeric_1 = np.argmax(train_target, axis=1)
		acc_1 = np.count_nonzero(soft_arg_max_1 == test_numeric_1)
		#print('episode num', e)
		train_accuracy.append(acc_1/soft_arg_max_1.shape[0])
        
		z1 = np.dot(test_data, mu_1) + b_b1_mu
		h1 = relu(z1)
		z2 = np.dot(h1, mu_2) + b_b2_mu 
		h2 = relu(z2)
		z3 = np.dot(h2, mu_3) + b_b3_mu
		out_2 = np.exp(z3)/np.sum(np.exp(z3), axis=1, keepdims=True)      

		soft_arg_max_2 = np.asarray(np.argmax(out_2, axis=1))
		#test_numeric_2 = np.argmax(test_target, axis=1)
		count = 0

		for i in range(len(test_target)):
			if test_target[i] == soft_arg_max_2[i]:
				count += 1
		print("epoch", e,"test accuracy",(count/len(test_target)),"train accuracy",(acc_1/soft_arg_max_1.shape[0]))
		test_accuracy.append(count/len(test_target))










