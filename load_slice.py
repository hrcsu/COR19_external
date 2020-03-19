import os
import nrrd
import numpy as np
import csv
from glob import glob
pt = '/media/user1/externaldrive8TB/COR19_add/NRRD_Penn usable/2019 first half'
tmp = []
bs = os.path.normpath(pt)
tp = glob(bs+'/*.nrrd')
for i in tp:
    name = i.split('/')
    name = name[-1].split('.')
    name = name[0]
    tmp.append(name)
print(len(tmp))

f = open('/home/user1/4TBHD/COR19_without_seg/label2.csv')
f = csv.reader(f)
add_case = open('/home/user1/4TBHD/COR19_without_seg/add_cases.txt','a')
for index,row in enumerate(f):
    if index == 0:
        continue
    #print(row)
    #print(row[0],row[4],row[5],row[7],row[8],row[10],row[11],row[13],row[14])
    img_name = row[0]
    if img_name not in tmp:
        continue
    #img_name = '32229982'
    try:
        begin = int(row[4])
        end = int(row[5])+1
        img,_ = nrrd.read('/home/user1/4TBHD/COR19_without_seg/pre_nrrd/'+img_name+'.nrrd')
        if 'NCVP' in img_name or 'SUB' in img_name:
            img = img[:,:,::-1]
            begin = int(row[4])-1
            end = int(row[5])
        #print('hhh')
        for i in range(begin,end):
            if len(str(i)) == 1:
                np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-00' + str(i), img[:,:,i])
            if len(str(i)) == 2:
                np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-0' + str(i), img[:,:,i])
            if len(str(i)) == 3:
                np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-' + str(i), img[:,:,i])
        if row[7]!='':
            begin = int(row[7])
            end = int(row[8])+1
            if 'NCVP' in img_name or 'SUB' in img_name:
                begin = int(row[7]) - 1
                end = int(row[8])
            for i in range(begin, end):
                if len(str(i)) == 1:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-00' + str(i), img[:, :, i])
                if len(str(i)) == 2:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-0' + str(i), img[:, :, i])
                if len(str(i)) == 3:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-' + str(i), img[:, :, i])
        if row[10]!='':
            begin = int(row[10])
            end = int(row[11])+1
            if 'NCVP' in img_name or 'SUB' in img_name:
                begin = int(row[10]) - 1
                end = int(row[11])
            for i in range(begin, end):
                if len(str(i)) == 1:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-00' + str(i), img[:, :, i])
                if len(str(i)) == 2:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-0' + str(i), img[:, :, i])
                if len(str(i)) == 3:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-' + str(i), img[:, :, i])
        if row[13]!='':
            begin = int(row[13])
            end = int(row[14])+1
            if 'NCVP' in img_name or 'SUB' in img_name:
                begin = int(row[13]) - 1
                end = int(row[14])
            for i in range(begin, end):
                if len(str(i)) == 1:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-00' + str(i), img[:, :, i])
                if len(str(i)) == 2:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-0' + str(i), img[:, :, i])
                if len(str(i)) == 3:
                    np.save('/home/user1/4TBHD/COR19_without_seg/preprocessed/' + img_name + '-' + str(i), img[:, :, i])
        add_case.write(img_name+'\n')
    except Exception as e:
        print(img_name)
add_case.close()