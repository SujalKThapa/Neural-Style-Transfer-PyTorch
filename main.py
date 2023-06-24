import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.models as models
import torchvision.transforms as transforms

import copy

class ContentLoss(nn.Module):
    def __int__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def foward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature)

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b * c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


def print_hi(name):
    print(f'{name}')
    # Press Ctrl+F8 to toggle the breakpoint.

def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.01)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    style_img = image_loader("./picasso.jpg")
    content_img = image_loader("./dancing.jpg")
    assert style_img.size() == content_img.size()#, \
    unloader = transforms.ToPILImage()
    plt.ion()
    plt.figure()
    imshow(style_img, title="Style Image")
    plt.figure()
    imshow(content_img, title="Content Image")
    print_hi('Initialization completed.')