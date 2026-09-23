import matplotlib.pyplot as plt
from matplotlib.image import imread

import numpy as np

import math

from sklearn.decomposition import PCA, IncrementalPCA

# Returns an array of images in the directory, in black and white.
def open_images(file_path):
    import os

    images = []
    for filename in os.listdir(file_path):
        if filename.endswith((".png", ".jpg", ".jpeg", "webp", "bmp")):
            path = os.path.join(file_path, filename)
            img = plt.imread(path)


            image_sum = img.sum(axis=2)
            image_bw = image_sum/image_sum.max()
            images.append(image_bw)


    return images


def consruct_image(image_bw, percent_components=95):
    pca = PCA()
    pca.fit(image_bw)

    # Getting the cumulative variance

    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100

    # How many PCs explain 95% of the variance?
    k = np.argmax(var_cumu>percent_components)

    ipca = IncrementalPCA(n_components=k)
    image_recon = ipca.inverse_transform(ipca.fit_transform(image_bw))

    return image_recon


def plot_image(image_recon):
    # Plotting the reconstructed image
    plt.figure(figsize=[12,8])
    plt.imshow(image_recon,cmap = plt.cm.gray)

def plot_range_variance(image_bw, component_range):

    plt.figure(figsize=[15,15])
    for i, component in enumerate(component_range):
        plt.subplot(math.ceil(len(component_range) / 3),3,i+1)
        plt.imshow(consruct_image(image_bw, component),cmap = plt.cm.gray)
        plt.title("Variance: "+str(component))

    plt.plot()