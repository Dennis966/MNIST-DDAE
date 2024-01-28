import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

"""
Note: It is important that we are clear about the data type of our data before pickling it.
"""

## Download & instantiate the "train_set" and "test_set" Dataset objects.
train_set = datasets.MNIST(root="./dataset/",
                           train=True,
                           download=True)

test_set = datasets.MNIST(root="./dataset/",
                          train=False,
                          download=True)

## Dataset size: As defined in the "__len__" method.
print("-------------- Attributes of the Dataset object -------------- ")
print("Training set size: {}".format(len(train_set)))   ## Training set size: 60000
print("Testing set size: {}".format(len(test_set)))     ## Testing set size: 10000


## Assess a sample data, As defined in the "__getitem__" method.
idx = 0
image, label = test_set[0]
print("-------------- Attributes of each data sample object -------------- ")
## Attributes of a raw image
print("type(image) = {}".format(type(image)))     ## type(image) = <class 'PIL.Image.Image'>
print("image.size = {}".format(image.size))      ## image.shape = (28, 28)

## Attributes of the label
print("label = {}".format(label))                 ## label = 7
print("type(label) = {}".format(type(label)))     ## type(label) = <class 'int'>

## Display image
fig_clean = plt.figure()
plt.imshow(image, cmap = 'gray')                  ## Single channel image, use grayscale color mapping
plt.colorbar()
plt.title("clean image: {}".format(label))
#plt.show()

## Convert to Numpy (for mathematical preprocessing)
np_image = np.array(image)
print("np_image.shape = {}".format(np_image.shape))              ## np_image.shape = (28, 28)

## noise statistics: mean 0, variance 1
mu = 0
sigma = 1
noise = np.random.normal(mu, sigma, size=image.size)

## Display image
fig_noise = plt.figure()
plt.imshow(noise, cmap = 'gray')                  ## Single channel image, use grayscale color mapping
plt.colorbar()
plt.title("noise image")
#plt.show()

## Sidenote: More on the statistics of noise: Run a Monte Carlo experiment
# Makes more sense to use histograms rather than scatter plots in this case.
###################################################################################################### plt.scatter(realized_samples)   # TypeError: scatter() missing 1 required positional argument: 'y'

MC_trials = 1000
realized_Gauss_samples = np.random.normal(mu, sigma, size=MC_trials)
fig_MC_1dim_gauss = plt.figure()

n_bins = 50
plt.hist(realized_Gauss_samples, n_bins, density=True)
plt.title("Monte Carlo I: (r.v.s sampled from standard normal)")
#plt.show()


## Back to MNIST: (add noise to clean image)
# Note: Technically, noise_factor is not SNR yet! But it corresponds to an SNR.
noise_factor = 10         # W.L.O.G
noisy_image = image + noise_factor * noise

fig_noisy = plt.figure()
plt.imshow(noisy_image, cmap = 'gray')                  ## Single channel image, use grayscale color mapping
plt.colorbar()
plt.title("noisy image")
#plt.show()


## Do the above for all clean images in both training & testing set
noise_factor = 20                                     # W.L.O.G
train_clean_dataset = list()
train_noisy_dataset = list()

test_clean_dataset = list()
test_noisy_dataset = list()
# Train
for train_idx in range(len(train_set)):
    print("Sample number {}".format(train_idx))
    raw_clean, _ = train_set[train_idx]
    np_clean = np.array(raw_clean, dtype=np.float64)
    #print("np_clean.dtype = {}".format(np_clean.dtype))
    train_clean_dataset.append(np_clean)

    noise = np.random.normal(mu, sigma, size=np_clean.shape)

    noisy = np_clean + noise_factor * noise
    #print("noisy.dtype = {}".format(noisy.dtype))

    assert np_clean.dtype == noisy.dtype
    train_noisy_dataset.append(noisy)

# Test
for test_idx in range(len(test_set)):
    print("Sample number {}".format(test_idx))
    raw_clean, _ = test_set[test_idx]
    np_clean = np.array(raw_clean, dtype=np.float64)
    #print("np_clean.dtype = {}".format(np_clean.dtype))
    test_clean_dataset.append(np_clean)

    noise = np.random.normal(mu, sigma, size=np_clean.shape)
    noisy = np_clean + noise_factor * noise
    #print("noisy.dtype = {}".format(noisy.dtype))

    assert np_clean.dtype == noisy.dtype
    test_noisy_dataset.append(noisy)

## Now we have a list of noisy photos for both train and test sets, each photo being a numpy array.
## store them in a pickle file? serialize them to binary data by using "wb" mode.
with open("train_clean.pickle", "wb") as train_clean_file:
    pickle.dump(train_clean_dataset, train_clean_file)

with open("train_noisy.pickle","wb") as train_noisy_file:
    pickle.dump(train_noisy_dataset, train_noisy_file)

with open("test_clean.pickle", "wb") as test_clean_file:
    pickle.dump(test_clean_dataset, test_clean_file)

with open("test_noisy.pickle","wb") as test_noisy_file:
    pickle.dump(test_noisy_dataset, test_noisy_file)







