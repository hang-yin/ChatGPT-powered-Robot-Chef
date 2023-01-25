from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# image = Image.open("./example2.jpg")
image = Image.open("./images/pressure_cooker.jpg")

# inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
# inputs = processor(text=["a photo of a fry pan", "a photo of beans", "a photo of eggs", "a photo of a hand", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

inputs = processor(text=["a photo of a pressure cooker", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
print("Similarity to pressure cooker: ", logits_per_image[0][0])
print("Similarity to dog: ", logits_per_image[0][1])
# print(logits_per_image)
print("Label probability for a pressure cooker: ", logits_per_image.softmax(dim=1)[0][0])
print("Label probability for a dog: ", logits_per_image.softmax(dim=1)[0][1])
# probificantly extends CLIP to learns = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)