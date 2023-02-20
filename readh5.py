import h5py
import numpy as np
import torch

file_name = './my_data/test.h5'
f1 = h5py.File(file_name,'r')
print(f1.keys())
from DCGN import DCGN
a_group_key = list(f1.keys())[0]

# get the object type for a_group_key: usually group or dataset
#print(type(f1[a_group_key]))

# If a_group_key is a group name,
# this gets the object names in the group and returns as a list
data = list(f1[a_group_key])
#print(data)

# If a_group_key is a dataset name,

graph = f1['TEST_0000.mp4']['change_points'][...].astype(np.int32)
tensor = torch.from_numpy(f1['TEST_0000.mp4']['features'][...].astype(np.float32))
#print(f1['TEST_0000.mp4']['n_frames'][...])
#print(tensor.shape)


net = DCGN(1024, 13)
out = net(tensor,graph)

print(out)