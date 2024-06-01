import os
from tqdm import tqdm
from PIL import Image
from PIL.Image import Resampling


def resizePIL(image_url):
    with Image.open(image_url) as im:
        # Provide the target width and height of the image
        (width, height) = (513, 512)
        im_resized = im.resize((width, height), resample=Resampling.LANCZOS)
        image = im_resized.rotate(270)
        return image


path_folder = os.path.join(os.path.dirname(__file__), "data/train_real_hands")
path_out = os.path.join(os.path.dirname(__file__), "data/resized_train_real_hands")

for folder in tqdm(os.listdir(path_folder)):
    for image in os.listdir(os.path.join(path_folder, folder)):
        input_image = os.path.join(path_folder, folder, image)
        resized_image = resizePIL(input_image)
        resized_image.save(os.path.join(path_out, folder, image))
