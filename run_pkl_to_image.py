""" temporary script to transform samples from pkl to images """

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch

# path to model generated results
path_gan_sample_pkl = './assert_results/pkl/'
path_gan_sample_img = './sample_jpg/'

# if not os.path.exists(path_gan_sample_pkl):
#     os.mkdir(path_gan_sample_pkl)

# if not os.path.exists(path_gan_sample_img):
#     os.mkdir(path_gan_sample_img)


# name of new data files
def get_filename_from_idx(idx):
    return 'sample_{:0>6}'.format(idx)


filename_sample_z = './sample_z.h5'

# get the pkl file list
list_pathfile_pkl = glob.glob(os.path.join(path_gan_sample_pkl, '*.pkl'))
list_pathfile_pkl.sort()

# loop to transform data and save image
list_z = []
i_counter = 0
for pathfile_pkl in list_pathfile_pkl:
    print(pathfile_pkl)
    with open(pathfile_pkl, 'rb') as f:
        pkl_content = pickle.load(f)
    x = pkl_content['x'] # x.shape = [B, (3, 1024, 1024)]
    z = pkl_content['z']
    num_cur = x.shape[0]
    for i in range(num_cur):
        # wenjianming
        pathfile_cur = os.path.join(path_gan_sample_img, get_filename_from_idx(i_counter))
        plt.imsave(path_gan_sample_img + "sample_{:0>6}.png".format(i_counter), x[i])
        np.save(pathfile_cur + '_z.npy', z[i].cpu())
        i_counter += 1
    list_z.append(z)

# save z (latent variables)
z_concat = torch.cat(list_z, axis=0).cpu().numpy()
pathfile_sample_z = os.path.join(path_gan_sample_img, filename_sample_z)
with h5py.File(pathfile_sample_z, 'w') as f:
    f.create_dataset('z', data=z_concat)
