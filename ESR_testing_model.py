import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

IMAGE_PATH = "resized_image.jpg"  # Path to input image
SAVED_MODEL_PATH = "C:/Users/sabha/.cache/kagglehub/models/kaggle/esrgan-tf2/tensorFlow2/esrgan-tf2/1"  # Path to TensorFlow SavedModel directory

def preprocess_image(image_path):
    """Loads image from path and preprocesses to make it model ready."""
    hr_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)  # Ensure 3 channels
    # If PNG, remove the alpha channel. The model only supports images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4  # Ensure size is divisible by 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)  # Add batch dimension

def save_image(image, filename):
    """Saves image as a .jpg file."""
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    print(f"Saved as {filename}.jpg")

def plot_image(image, title=""):
    """Plots images from image tensors."""
    image = np.asarray(image)
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)

# Load and preprocess the input image
hr_image = preprocess_image(IMAGE_PATH)
# Plotting original resolution image
#plot_image(tf.squeeze(hr_image), title="Original Image")
#save_image(tf.squeeze(hr_image), filename="Original Image")

# Load the saved model
model = tf.saved_model.load(SAVED_MODEL_PATH)

# Run the model on the image
start = time.time()
fake_image = model(hr_image)  # Assuming the model takes in and outputs an image tensor
fake_image = tf.squeeze(fake_image)  # Remove batch dimension
print(f"Time Taken: {time.time() - start:.2f} seconds")

# Plotting super resolution image
folder1 = 'esr_imgs'
if not os.path.exists(folder1):
    os.makedirs(folder1)

# Plot and save the super-resolution image
plot_image(fake_image, title="Super Resolution Image")
filename1 = os.path.join(folder1, 'Super_Resolution_img.png')
plt.savefig(filename1)
plt.close()
