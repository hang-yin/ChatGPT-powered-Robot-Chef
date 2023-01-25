from PIL import Image
from torchvision import transforms

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
