""" generation of images interactively with ui control """
import time
import pickle
import os
import numpy as np
import torch
import matplotlib

import stylegan2

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

plt.ion()

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
path_feature_direction = './assert_result/slope/slope.pkl'

with open(path_feature_direction, 'rb') as f:
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
# latents = torch.from_numpy(np.random.randn(1, latent_size)).to(device=device, dtype=torch.float32)
latents = G.G_mapping(torch.randn([1, 512])).to(device=device, dtype=torch.float32)
# Generate dummy labels
labels = None


def gen_image(latents):
    """
    tool funciton to generate image from latent variables
    :param latents: latent variables
    :return:
    """
    with torch.no_grad():
        images = G(latents, labels=labels)
    return torch.from_numpy(np.clip((images[0].permute(1,2,0).numpy()[1::2, 1::2, :] + 1.0)/2.0, a_min=0.0, a_max=1.0))


img_cur = gen_image(latents)

##
""" plot figure with GUI """
h_fig = plt.figure(figsize=[10, 10])
h_ax = plt.axes([0.0, 0.0, 0.5, 1.0])
h_ax.axis('off')
h_img = plt.imshow(img_cur)


class GuiCallback(object):
    counter = 0
    latents = latents

    def __init__(self):
        self.latents = torch.from_numpy(np.random.randn(1, G.latent_size)).to(device=device, dtype=torch.float32)
        self.feature_direction = feature_direction
        self.feature_lock_status = np.zeros(num_feature).astype('bool')
        self.feature_directoion_disentangled = torch.from_numpy(feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))).to(device=device, dtype=torch.float32)
        img_cur = gen_image(self.latents)
        h_img.set_data(img_cur)
        plt.draw()

    def random_gen(self, event):
        self.latents = torch.from_numpy(np.random.randn(1, G.latent_size)).to(device=device, dtype=torch.float32)
        img_cur = gen_image(self.latents)
        h_img.set_data(img_cur)
        plt.draw()

    def modify_along_feature(self, event, idx_feature, step_size=0.05):
        self.latents += self.feature_directoion_disentangled[:, idx_feature] * step_size #选择第几个特征
        img_cur = gen_image(self.latents)
        h_img.set_data(img_cur)
        plt.draw()
        plt.savefig(os.path.join(path_gan_explore_interactive,
                                 '{}_{}_{}.png'.format(gen_time_str(), feature_name[idx_feature],
                                                       ('pos' if step_size > 0 else 'neg'))))

    def set_feature_lock(self, event, idx_feature):
        self.feature_lock_status[idx_feature] = np.logical_not(self.feature_lock_status[idx_feature])
        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))


callback = GuiCallback()

ax_randgen = plt.axes([0.55, 0.90, 0.15, 0.05])
b_randgen = widgets.Button(ax_randgen, 'Random Generate')
b_randgen.on_clicked(callback.random_gen)


def get_loc_control(idx_feature, nrows=8, ncols=5,
                    xywh_range=(0.51, 0.05, 0.48, 0.8)):
    r = idx_feature // ncols
    c = idx_feature % ncols
    x, y, w, h = xywh_range
    xywh = x + c * w / ncols, y + (nrows - r - 1) * h / nrows, w / ncols, h / nrows
    return xywh


step_size = 0.8


def create_button(idx_feature):
    """ function to built button groups for one feature """
    x, y, w, h = get_loc_control(idx_feature)

    plt.text(x + w / 2, y + h / 2 + 0.01, feature_name[idx_feature], horizontalalignment='center',
             transform=plt.gcf().transFigure)

    ax_neg = plt.axes((x + w / 8, y, w / 4, h / 2))
    b_neg = widgets.Button(ax_neg, '-', hovercolor='0.1')
    b_neg.on_clicked(lambda event:
                     callback.modify_along_feature(event, idx_feature, step_size=-1 * step_size))

    ax_pos = plt.axes((x + w * 5 / 8, y, w / 4, h / 2))
    b_pos = widgets.Button(ax_pos, '+', hovercolor='0.1')
    b_pos.on_clicked(lambda event:
                     callback.modify_along_feature(event, idx_feature, step_size=+1 * step_size))

    ax_lock = plt.axes((x + w * 3 / 8, y, w / 4, h / 2))
    b_lock = widgets.CheckButtons(ax_lock, ['L'], [False])
    b_lock.on_clicked(lambda event:
                      callback.set_feature_lock(event, idx_feature))
    return b_neg, b_pos, b_lock


list_buttons = []
for idx_feature in range(num_feature):
    list_buttons.append(create_button(idx_feature))

plt.ioff()
plt.show()