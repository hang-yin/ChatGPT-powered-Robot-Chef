from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

image = Image.open("./example2.jpg")
# convert image to PyTorch tensor using torchvision.transforms
image = transforms.ToTensor()(image) # image shape [3, 401, 604]

# add batch dimension and shift color channels
patches = image.data.unfold(0,3,3) # image shape [1, 401, 604, 3]

# break image into patches of size 32x32
patch_size = 32
# horizontal patches
patches = patches.unfold(1,patch_size,patch_size) # image shape: [1, 12, 604, 3, 32]
# vertical patches
patches = patches.unfold(2,patch_size,patch_size) # image shape: [1, 12, 18, 3, 32, 32]

window = 3
stride = 1

# window slides from top to bottom
for Y in range(0, patches.shape[1]-window+1, stride):
    # window slides from left to right
    for X in range(0, patches.shape[2]-window+1, stride):
        # initialize an empty big_patch array
        big_patch = torch.zeros(patch_size*window, patch_size*window, 3)
        # this gets the current batch of patches that will make big_batch
        patch_batch = patches[0, Y:Y+window, X:X+window]
        # loop through each patch in current batch
        for y in range(patch_batch.shape[1]):
            for x in range(patch_batch.shape[0]):
                # add patch to big_patch
                big_patch[
                    y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size, :
                ] = patch_batch[y, x].permute(1, 2, 0)
        # display current big_patch
        plt.imshow(big_patch)
        plt.show()
