from PIL import Image
import numpy as np
from torchvision import transforms
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_state(state):
    """
    Takes in raw Atari image, returns 84x84 resized/scaled grayscale image
    state: should be 210 x 160 x 3 shaped np.array
    output: 1x84x84 image
    """
    cropped = Image.fromarray(state)\
        .crop((0, 34, 160, 160 + 34))
    composite = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((96, 96))
    ])
    image = composite(cropped)
    small_img = np.uint8(image)
    return np.expand_dims(small_img, 0)


def copy_model_params(model1, model2):
    """
    copies model parameters from model1 to model2. Both must be same model class
    :param model1:
    :param model2:
    :return: model2
    """
    return model2.load_state_dict(model1.state_dict())