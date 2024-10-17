#%% 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # Use matplotlib for image visualization
import numpy as np
from termcolor import colored

#%% 
# Functions to show an image using matplotlib
def imshow_matplotlib(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()  # Convert to NumPy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Transpose from CHW to HWC
    plt.show()  # Display the image

#%% 
if __name__ == '__main__':
    # Check if Pytorch is using the GPU:
    print(colored("should be true: ", "yellow"), torch.cuda.is_available()) # should be true
    print(colored("should be 1: ","yellow"),torch.cuda.device_count()) # should be 1
    print(colored("should be 0: ","yellow"),torch.cuda.current_device()) # should be 0
    print(colored("GPU name: ","yellow"),torch.cuda.get_device_name(0)) # should return the GPU name

    # Dataset Implementation using CIFAR10
    # Load and normalize 
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
#%% 
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('food', 'rice', 'noodles', 'water')
#%% 
    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Move images to GPU for any processing (if necessary)
    images = images.to('cuda')

    # Show images using matplotlib (move to CPU before displaying)
    imshow_matplotlib(torchvision.utils.make_grid(images.cpu()))  # Move back to CPU for imshow

    # Print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# %% 
