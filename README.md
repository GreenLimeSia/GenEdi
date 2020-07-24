## Generating and Editing Facial Images with StyleGAN2 &mdash; Pytorch1.5 Implementation
### About StyleGAN2
This is an unofficial port of the StyleGAN2 architecture and training procedure from [the official Tensorflow implementation](https://github.com/NVlabs/stylegan2) to Pytorch. Pretrained Tensorflow models can be converted into Pytorch models.

This model is built to be runnable for 1d, 2d and 3d data.

### Using pre-trained networks
Networks can be converted from their tensorflow version by running `run_convert_from_tf.py`. The networks have to be of type StyleGAN2, the baseline StyleGAN is not supported (config a-d). The network weights can be automatically downloaded if you specify `--download=NAME` where `NAME` is one of the following:

```
ffhq-config-e
ffhq-config-f
```
Alternatively, specify a file directly with `--input=FILE`.

### Generating images
```.bash
# Train a network or convert a pretrained one.
# Example of converting pretrained ffhq model:
python run_convert_from_tf.py --download ffhq-config-f --output G.pth D.pth Gs.pth

# Generate ffhq uncurated images (matches paper Figure 12)
python run_generator.py generate_images --network=Gs.pth --seeds=6600-6625 --truncation_psi=0.5

# Generate ffhq curated images (matches paper Figure 11)
python run_generator.py generate_images --network=Gs.pth --seeds=66,230,389,1518 --truncation_psi=1.0
```
The results are placed in `<RUNNING_DIR>/results/*.png`. You can change the location with `--output`. For example, `--output=~/my-stylegan2-results`.

### Projecting images to latent space
```.bash
# Train a network or convert a pretrained one.
# Example of converting pretrained ffhq model:
python run_convert_from_tf.py --download ffhq-config-f --output G.pth D.pth Gs.pth

# Project generated images
python run_projector.py project_generated_images --network=Gs.pth --seeds=0,1,5

# Project real images
python run_projector.py project_real_images --network=Gs.pth --data-dir=path/to/image_folder
```

### Training and Evaluating StyleGAN2
When specifying a location of your images it should be a root directory where images are located in the root directory or any subdirectories. If conditioning networks with labels, classes are interpreted as the subdirectory of the root directory that the image was loaded from.

Note that a GPU with less than 16 GB memory will not be able to train on images of size 1024x1024.
See `python run_training.py --help` for info on training. This script is unique compared to the other runnable python files in that the settings can be specified beforehand in a yaml file. Just pass the yaml file as the first positional argument. Any following keyword arguments will take precedent over the same argument if found in the yaml file.
Example:
```yaml
# Here is an example training config file for 512x512 images.
channels: [32, 64, 128, 256, 512, 512, 512, 512]
tensorboard_log_dir: 'runs/stylegan2_512x512'
tensorboard_image_interval: 500
checkpoint_dir: 'checkpoints/stylegan2_512x512'
checkpoint_interval: 10000
data_dir: /path/to/dataset/images512x512/
gpu: [0, 1, 2, 3]
```
This training setup can be ran by `python run_training.py my_settings.yaml --gpu 0 --resume`.
In this example we did override one of the options, the `--gpu` option which was set to 4 devices in the settings file. But since the keyword arguments take precedent over the config file we will instead train only on one GPU device. We also added the argument `--resume` which will look in the checkpoint directory if we specified one and attempt to load the latest checkpoint before continuing to train. If no checkpoints are available, the training is started from scratch.

See `python run_metrics.py --help` for info on metric evaluation (metric evaluation can also be performed during training at specified intervals). This script supports multi-GPU metric evaluations.
Metrics will be appended to the file `metrics.json` if it exists. Identical metrics will be overwritten by the one that was last evaluated.

### Kaggle Manipulating
Our method also upload to kaggle, you can copy to your own directory and use it. The kaggle link is available in https://www.kaggle.com/greenlime/genandedi/edit?rvi=1
### Code information
Base functionality is tested but there might be settings that raise exceptions as the repository contains a couple thousand lines of code. If you find one, please open an issue or pull request.


