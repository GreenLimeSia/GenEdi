"""
try face tl_gan using pg-gan model, modified from
https://drive.google.com/drive/folders/1A79qKDTFp6pExe4gTSgBsEOkxwa2oes_
"""

"""
prerequsit: before running the code, download pre-trained model to project_root/asset_model/
pretrained model download url: https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU
model name: karras2018iclr-celebahq-1024x1024.pkl
"""

import os
import stylegan2
import time
import pickle
import numpy as np
import torch

torch.backends.cudnn.enabled = False

# path to model code and weight
""" load gan model """
path_model = './Gs.pth'
gpus = [0]  # 使用哪几个GPU进行训练，这里选择0号GPU
cuda_gpu = torch.cuda.is_available()  # 判断GPU是否存在可用
device = torch.device(gpus[0] if cuda_gpu else 'cpu')
print(cuda_gpu)
G = stylegan2.models.load(path_model)
assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
                                                  'stylegan2.models.Generator. Found {}.'.format(type(G))
G.to(device)
latent_size, label_size = G.latent_size, G.label_size

# path to model generated results
path_gen_sample = './assert_results/pkl'
# if not os.path.exists(path_gen_sample):
#     os.mkdir(path_gen_sample)

""" gen samples and save as pickle """

n_batch = 8000
batch_size = 16

for i_batch in range(n_batch):
    try:
        i_sample = i_batch * batch_size

        tic = time.time()

        latents = torch.from_numpy(np.random.randn(batch_size, latent_size)).to(device=device, dtype=torch.float32)

        # Generate dummy labels (not used by the official networks).
        labels = None

        # Run the generator to produce a set of images. tensor to numpy
        with torch.no_grad():
            images = G(latents, labels)

        images = np.clip((images.permute(0, 2, 3, 1).cpu().numpy() + 1.0) / 2.0, a_min=0.0, a_max=1.0)

        # images = images[:, 4::8, 4::8, :] (1024,1024,3)->(128,128,3)
        # images = images[:, 1::2, 1::2, :] # (1024,1024,3)->(512,512,3)

        with open(os.path.join(path_gen_sample, 'style_celeba_{:0>6d}.pkl'.format(i_sample)), 'wb') as f:
            pickle.dump({'z': latents, 'x': images}, f)

        toc = time.time()
        print(i_sample, toc - tic)

    except:
        print('error in {}'.format(i_sample))

""" view generated samples """
yn_view_sample = False
if yn_view_sample:
    with open(os.path.join(path_gen_sample, 'stylean_celeba_{:0>6d}.pkl'.format(0)), 'rb') as f:
        temp = pickle.load(f)

    import matplotlib.pyplot as plt

    plt.imshow(temp['x'][0])
    plt.show()
