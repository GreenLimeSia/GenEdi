""" generation of images interactively with ui control """
import time
import pickle
import os
import numpy as np
import torch
import datetime
import stylegan2

import matplotlib.pyplot as plt
import feature_axis as feature_axis
import feature_celeba_organize as feature_celeba_organize


def gen_time_str():
    """ tool function """
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


""" location to save images """
path_gan_explore_interactive = 'interactive'
if not os.path.exists(path_gan_explore_interactive):
    os.mkdir(path_gan_explore_interactive)

##
""" load feature directions """
pathfile_feature_direction = './assert_result/slope/slope.pkl'

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction'] * feature_celeba_organize.feature_reverse[None, :]
feature_name = feature_celeba_organize.feature_name_celeba_rename
num_feature = feature_direction.shape[1]

##
""" load gan model """

# path to model code and weight
path_model = './Gs.pth'
device = torch.device('cpu')
G = stylegan2.models.load(path_model)
assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
                                                  'stylegan2.models.Generator. Found {}.'.format(type(G))

latent_size, label_size = G.latent_size, G.label_size

##

# Generate random latent variables
latents_c = np.random.randn(1, latent_size)


# latents = G.G_mapping(torch.randn([1, 512])).to(device=device, dtype=torch.float32)
# Generate dummy labels
# labels = None


def gen_image(latents, labels):
    """
    tool funciton to generate image from latent variables
    :param latents: latent variables
    :return:
    """
    with torch.no_grad():
        images = G(latents, labels=labels)
    return torch.from_numpy(
        np.clip((images.permute(0, 2, 3, 1).numpy() + 1.0) / 2.0, a_min=0.0, a_max=1.0))


# img_cur = gen_image(latents)

batch_size = 10

step_size = 0.2

counter = 0

feature_lock_status = np.zeros(num_feature).astype('bool')
feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
    feature_direction, idx_base=np.flatnonzero(feature_lock_status))

for i_feature in range(feature_direction.shape[1]):
    latents_0 = latents_c - feature_directoion_disentangled[:, i_feature] * step_size
    latents_1 = latents_c + feature_directoion_disentangled[:, i_feature] * step_size

    print(np.mean(latents_0 - latents_1) ** 2)

    latents = np.zeros([batch_size, latent_size])

    for i_alpha, alpha in enumerate(np.linspace(0, 1, batch_size)):
        latents[i_alpha, :] = latents_0[0] * (1 - alpha) + latents_1[0] * alpha

    latents = torch.from_numpy(latents).to(device=device, dtype=torch.float32)
    # Generate dummy labels (not used by the official networks).
    labels = None

    # Run the generator to produce a set of images.
    images = gen_image(latents, labels)

    for idx in range(images.shape[0]):
        plt.imsave(os.path.join(path_gan_explore_interactive,
                                '{}_{}_{}.png'.format(counter, feature_name[i_feature],
                                                      ('pos' if step_size > 0 else 'neg'))), images[idx].cpu().numpy())
        counter += 1
