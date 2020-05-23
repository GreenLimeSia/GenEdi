import matplotlib
import matplotlib.pyplot as plt
from torchsummaryX import summary
import pylab
from stylegan2 import utils
import numpy as np
import torch
import stylegan2
import pickle
import os

# matplotlib.use('TkAgg')

if __name__ == "__main__":
    # path to model code and weight
    plt.ion()
    path_model = './Gs.pth'
    device = torch.device('cpu')
    latens = torch.randn([3, 512])
    G = stylegan2.models.load(path_model)
    assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
                                                      'stylegan2.models.Generator. Found {}.'.format(type(G))
    G.to(device)
    latent_size, label_size = G.latent_size, G.label_size

    summary(G, torch.zeros(1, 13, 512))
    # for parameters in G.parameters():
    #     print(parameters)

    # for name, parameters in G.named_parameters():
    #     print(name, ':', parameters.size())

    # a = G.G_mapping(latens)

    ##

    # # Generate random latent variables
    # latents_ = []
    # rnd = np.random.RandomState(seed=6600)
    # latents_.append(torch.from_numpy(rnd.randn(latent_size)))
    # #latents = torch.from_numpy(np.random.randn(1, latent_size)).to(device=device, dtype=torch.float32)
    # # Generate dummy labels
    # latents = torch.stack(latents_, dim=0).to(device=device, dtype=torch.float32)
    # labels = None
    #
    #
    # def gen_image(latents):
    #     """
    #     tool funciton to generate image from latent variables
    #     :param latents: latent variables
    #     :return:
    #     """
    #     with torch.no_grad():
    #         images = G(latents, labels=labels)
    #     return np.clip((images[0].permute(1,2,0).numpy() + 1.0)/2.0, a_min=0.0, a_max=1.0)
    #
    #
    # img_cur = gen_image(latents)
    # # """ plot figure with GUI """
    # h_fig = plt.figure(figsize=[30, 30])
    # h_ax = plt.axes([0.0, 0.0, 0.5, 1.0])
    # h_ax.axis('off')
    # h_img = plt.imshow(img_cur)
    # plt.show()
    print()
