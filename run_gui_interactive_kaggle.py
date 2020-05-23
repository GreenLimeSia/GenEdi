""" import packages """

import numpy as np
import pickle
import torch
import PIL
import io
import stylegan2
import feature_axis as feature_axis
from torchvision import transforms
import feature_celeba_organize as feature_celeba_organize

import IPython
import ipywidgets
from IPython.display import display

def gen_image(latents):
    """
    tool funciton to generate image from latent variables
    :param latents: latent variables
    :return:
    """
    with torch.no_grad():
        images = G(latents, labels=labels)
    return torch.from_numpy(
        np.clip((images[0].permute(1, 2, 0).cpu().numpy()[1::2, 1::2, :] + 1.0) / 2.0, a_min=0.0, a_max=1.0))


""" load feature directions """
path_feature_direction = './assert_result/slope/slope.pkl'

with open(path_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction'] * feature_celeba_organize.feature_reverse[None, :]
feature_name = feature_celeba_organize.feature_name_celeba_rename
num_feature = feature_direction.shape[1]

feature_name = feature_celeba_organize.feature_name_celeba_rename
feature_direction = feature_direction_name['direction'] * feature_celeba_organize.feature_reverse[None, :]

##
""" load gan model """

# path to model code and weight
path_model = './Gs.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = stylegan2.models.load(path_model)
assert isinstance(G, stylegan2.models.Generator), 'Model type has to be ' + \
                                                  'stylegan2.models.Generator. Found {}.'.format(type(G))
G.to(device)
latent_size, label_size = G.latent_size, G.label_size

##

# Generate random latent variables
latents = torch.from_numpy(np.random.randn(1, latent_size)).to(device=device, dtype=torch.float32)
# Generate dummy labels
labels = None

x_sample = gen_image(latents)

""" ========== ipywigets gui interface ========== """


def img_to_bytes(x_sample):
    """ tool funcion to code image for using ipywidgets.widgets.Image plotting function """
    x_sample = x_sample.numpy()
    imgObj = PIL.Image.fromarray((x_sample * 255).astype(np.uint8)).convert('RGB')
    imgByteArr = io.BytesIO()
    imgObj.save(imgByteArr, format='PNG')
    imgBytes = imgByteArr.getvalue()
    return imgBytes


w_img = ipywidgets.widgets.Image(value=img_to_bytes(x_sample), fromat='png',
                                 width=512, height=512,
                                 layout=ipywidgets.Layout(height='512px', width='512px')
                                 )


class GuiCallback(object):
    """ call back functions for button click behaviour """
    counter = 0

    #     latents = z_sample
    def __init__(self):
        self.latents = torch.from_numpy(np.random.randn(1, G.latent_size)).to(device=device, dtype=torch.float32)
        self.feature_direction = feature_direction
        self.feature_lock_status = np.zeros(num_feature).astype('bool')
        self.feature_directoion_disentangled = torch.from_numpy(feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))).to(device=device,
                                                                                           dtype=torch.float32)

    def random_gen(self, event):
        self.latents = torch.from_numpy(np.random.randn(1, G.latent_size)).to(device=device, dtype=torch.float32)
        self.update_img()

    def modify_along_feature(self, event, idx_feature, step_size=0.3):
        self.latents += self.feature_directoion_disentangled[:, idx_feature] * step_size
        self.update_img()

    def set_feature_lock(self, event, idx_feature, set_to=None):
        if set_to is None:
            self.feature_lock_status[idx_feature] = np.logical_not(self.feature_lock_status[idx_feature])
        else:
            self.feature_lock_status[idx_feature] = set_to
        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))

    def update_img(self):
        x_sample = gen_image(self.latents)
        x_byte = img_to_bytes(x_sample)
        w_img.value = x_byte


guicallback = GuiCallback()

step_size = 0.4


def create_button(idx_feature, width=128, height=40):
    """ function to built button groups for one feature """
    w_name_toggle = ipywidgets.widgets.ToggleButton(
        value=False, description=feature_name[idx_feature],
        tooltip='{}, Press down to lock this feature'.format(feature_name[idx_feature]),
        layout=ipywidgets.Layout(height='{:.0f}px'.format(height / 2),
                                 width='{:.0f}px'.format(width),
                                 margin='2px 2px 2px 2px')
    )
    w_neg = ipywidgets.widgets.Button(description='-',
                                      layout=ipywidgets.Layout(height='{:.0f}px'.format(height / 2),
                                                               width='{:.0f}px'.format(width / 2),
                                                               margin='1px 1px 5px 1px'))
    w_pos = ipywidgets.widgets.Button(description='+',
                                      layout=ipywidgets.Layout(height='{:.0f}px'.format(height / 2),
                                                               width='{:.0f}px'.format(width / 2),
                                                               margin='1px 1px 5px 1px'))

    w_name_toggle.observe(lambda event:
                          guicallback.set_feature_lock(event, idx_feature))
    w_neg.on_click(lambda event:
                   guicallback.modify_along_feature(event, idx_feature, step_size=-1 * step_size))
    w_pos.on_click(lambda event:
                   guicallback.modify_along_feature(event, idx_feature, step_size=+1 * step_size))

    button_group = ipywidgets.VBox([w_name_toggle, ipywidgets.Box([w_neg, w_pos])],
                                   layout=ipywidgets.Layout(border='1px solid gray'))

    return button_group


list_buttons = []
for idx_feature in range(num_feature):
    list_buttons.append(create_button(idx_feature))

yn_button_select = False


def arrange_buttons(list_buttons, yn_button_select=True, ncol=4):
    num = len(list_buttons)
    if yn_button_select:
        feature_celeba_layout = feature_celeba_organize.feature_celeba_layout
        layout_all_buttons = ipywidgets.VBox(
            [ipywidgets.Box([list_buttons[item] for item in row]) for row in feature_celeba_layout])
    else:
        layout_all_buttons = ipywidgets.VBox(
            [ipywidgets.Box(list_buttons[i * ncol:(i + 1) * ncol]) for i in range(num // ncol + int(num % ncol > 0))])
    return layout_all_buttons


# w_button.on_click(on_button_clicked)
guicallback.update_img()
w_button_random = ipywidgets.widgets.Button(description='random face', button_style='success',
                                            layout=ipywidgets.Layout(height='40px',
                                                                     width='128px',
                                                                     margin='1px 1px 5px 1px'))
w_button_random.on_click(guicallback.random_gen)

w_box = ipywidgets.Box([w_img,
                        ipywidgets.VBox([w_button_random,
                                         arrange_buttons(list_buttons, yn_button_select=True)])
                        ], layout=ipywidgets.Layout(height='628px', width='1024px')
                       )

print('INSTRUCTION: press +/- to adjust feature, toggle feature name to lock the feature')
display(w_box)
