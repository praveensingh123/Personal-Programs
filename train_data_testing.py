import numpy as np 

file = open('training-data-small.txt', 'r')
#file = open('test.txt','r')

label = [] # list for label
no_of_data = [] # for creating the dictionary, gather all the x_data
train_x = [] #
for line in file:
	lx = line.split('\t')
	ly = lx[1].rstrip().split(',')
	label.append(lx[0]) # labels appending
	train_x.append(ly) # used for labeling
	no_of_data.extend(ly)

max_len = max([len(i) for i in train_x])
min_len = min([len(i) for i in train_x])
mean_len = np.mean([len(i) for i in train_x])
std_len = np.std([len(i) for i in train_x])

print('max sentence length: ',max_len)
print('min sentence length: ', min_len)
print('mean:', mean_len)
print('std:', std_len)

# creating label array
train_label = np.asarray(label).reshape(-1,)
print('label array shape:', train_label.shape)

train_dict = dict()

for i,j in enumerate(no_of_data):
	train_dict[j] = i

print('unique words:',len(train_dict.keys()))

#print(train_dict)

# making the dictionary start with zero
for i,j in enumerate(train_dict.keys()):
	train_dict[j] = i

#print(train_dict)

file.close()

# making the train_data as integers

#for i in train_x:
#	for j 


#train_data = np.asarray(train_x)
#print(train_data)
#print(train_data.shape)
#print(train_data[0])


train_data = np.asarray(train_x).reshape(-1,)
for index,i in enumerate(train_data):
	for pos,k in enumerate(i):
		train_data[index][pos] = train_dict[train_data[index][pos]]
		#print(ls_ara[index][j])

print(train_data)
print(train_data.shape)








