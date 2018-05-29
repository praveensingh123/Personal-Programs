import numpy as np
import collections

file = open('test.txt','r')

label = []
train_x = []
for line in file:
	#print(type(line))
	lx = line.split('\t')
	#ly = lx[1].rstrip().split(',')
	label.append(tuple(lx[0]))
	train_x.extend(lx[1].rstrip().split(','))
	#print(len(ly))
print(label)
#print(train_x)
#print(collections.Counter(train_x))
di = collections.Counter(train_x)
#print(type(di))

train_dict = dict()

for i,j in enumerate(train_x):
	train_dict[j] = i

print(train_dict) 

test_list = ['x','y','z','x','y']
test_dict = dict()

for i, j in enumerate(test_list):
	test_dict[j] = i

print(test_dict)

print(len(test_dict.keys()))
print(len(train_dict.keys()))

dd = {'1':'x', '2':'y', '4':'z', '5':'o'}
ls = [['1','2','4'],['4','5']]
ls_ara = np.array(ls).reshape(-1,)
print(ls_ara)
for index,i in enumerate(ls_ara):
	for j,k in enumerate(i):
		ls_ara[index][j] = dd[ls_ara[index][j]]
		print(ls_ara[index][j])
	#print(i)

print(ls_ara)