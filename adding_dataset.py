import os
import numpy as np
from tqdm import tqdm
file_name = 'Final_second'
#training_data = list(np.load('dataset/Combined_dataset_v2.npy' ,allow_pickle=True))+list(np.load('Dataset/6.npy' ,allow_pickle=True))+list(np.load('Dataset/7.npy' ,allow_pickle=True)) + list(np.load('Dataset/8.npy' ,allow_pickle=True)) + list(np.load('Dataset/9.npy' ,allow_pickle=True))
#np.save(file_name,training_data)
lst = [i for i in os.listdir('Car-dataset/')]
print(lst)
mf = []
for i in tqdm(lst):
    ds = 'Car-dataset/{}'.format(i)
    ds = list(np.load(ds ,allow_pickle=True))
    mf+=ds
mf = np.array(mf)
np.save(file_name,mf)










