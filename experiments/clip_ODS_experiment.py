from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import matplotlib.patches

image = Image.open("./images/table1.png")
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

# initialize CLIP model
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

window = 3
stride = 1
scores = torch.zeros(patches.shape[1], patches.shape[2])
runs = torch.ones(patches.shape[1], patches.shape[2])

# prompt = "A photo of a fried egg"
prompt = "an eggplant"

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
        # we preprocess the image and class label with the CLIP processor
        inputs = processor(
            images=big_patch,  # big patch image sent to CLIP
            return_tensors="pt",  # tell CLIP to return pytorch tensor
            text=prompt,  # class label sent to CLIP
            padding=True
        ).to(device) # move to device if possible

        # calculate and retrieve similarity score
        score = model(**inputs).logits_per_image.item()
        # sum up similarity scores from current and previous big patches
        # that were calculated for patches within the current window
        scores[Y:Y+window, X:X+window] += score
        # calculate the number of runs on each patch within the current window
        runs[Y:Y+window, X:X+window] += 1

# calculate average similarity score for each patch
scores /= runs

# clip the scores
scores = np.clip(scores-scores.mean(), 0, np.inf)

# normalize scores
scores = (
    scores - scores.min()) / (scores.max() - scores.min()
)

# transform the patches tensor
adj_patches = patches.squeeze(0).permute(3, 4, 2, 0, 1)
adj_patches.shape

# multiply patches by scores
adj_patches = adj_patches * scores

# rotate patches to visualize
adj_patches = adj_patches.permute(3, 4, 2, 0, 1)


Y = adj_patches.shape[0]
X = adj_patches.shape[1]

fig, ax = plt.subplots(Y, X, figsize=(X*.5, Y*.5))
for y in range(Y):
    for x in range(X):
        ax[y, x].imshow(adj_patches[y, x].permute(1, 2, 0))
        ax[y, x].axis("off")
        ax[y, x].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

"""

# scores higher than 0.5 are positive
detection = scores > 0.8

# calculate coordinates of bounding boxes
y_min, y_max = (
    np.nonzero(detection)[:,0].min().item(),
    np.nonzero(detection)[:,0].max().item()+1
)
x_min, x_max = (
    np.nonzero(detection)[:,1].min().item(),
    np.nonzero(detection)[:,1].max().item()+1
)
y_min *= patch_size
y_max *= patch_size
x_min *= patch_size
x_max *= patch_size
height = y_max - y_min
width = x_max - x_min

# move color channel to final dim
image = np.moveaxis(image.data.numpy(), 0, -1)

fig, ax = plt.subplots(figsize=(Y*0.5, X*0.5))

ax.imshow(image)

# Create a Rectangle patch
rect = matplotlib.patches.Rectangle(
    (x_min, y_min), width, height,
    linewidth=3, edgecolor='#FAFF00', facecolor='none'
)

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
"""