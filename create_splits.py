import os
import numpy as np
from glob import glob
import random
random_index2 = random.sample(range(1,587),76)
f = open('/home/user1/4TBHD/COR19_exteranl_test/train_set.txt','w')
fv = open('/home/user1/4TBHD/COR19_exteranl_test/validation_set.txt','w')
add_cases = open('/home/user1/4TBHD/COR19_exteranl_test/add_cases.txt','r')
add_set = []
add_ = add_cases.readlines()
test_set = []
validation_set = []
train_set = []

ft_Ben_test = []
for line in add_:
    line = line.replace('\n','')
    add_set.append(line)


patho = '/home/user1/4TBHD/COR19/COR19_new/data'
basedir = os.path.normpath(patho)
files = glob(basedir+'/*.nrrd')
train_validation_set = []
count1 = 0
count2 = 0
tmp = []
for file in files:
    name = file.split('/')
    name = name[-1].split('.')
    name = name[0]
    train_validation_set.append(name)

for i in add_set:
    train_validation_set.append(i)
import random
random.shuffle(train_validation_set)

for i in train_validation_set:
    count2 += 1
    if count2 in random_index2:
        validation_set.append(name)
        fv.write(i+'\n')
    else:
        train_set.append(name)
        f.write(i+'\n')

print(len(train_validation_set))
print(len(test_set))
print(len(validation_set))
print(len(train_set))

fv.close()
f.close()

###in ben's test_set but not in demographics: MIX411,MIX381
#in ben's test_set but it is negative: COR047
#test_set:58-->55
#validation: 55
#train+validation: 361