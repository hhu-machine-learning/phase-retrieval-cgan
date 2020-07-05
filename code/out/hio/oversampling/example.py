import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval

np.random.seed(11)
np.random.seed(15)

pad = 1
mask = None

image = color.rgb2gray(io.imread('test.png'))

if pad:
    image = np.pad(image, 14, mode='constant')
    mask = np.pad(np.ones((28,28)), 14, mode='constant')

magnitudes = np.abs(np.fft.fft2(image))
result = fienup_phase_retrieval(magnitudes, steps=1000, mask=mask, 
                                verbose=False)


if pad:
    plt.imsave('orig_pad.png', image, cmap='gray')
    plt.imsave('result_pad.png', result, cmap='gray')
    plt.imsave('magn_pad.png', np.fft.fftshift(np.log(magnitudes)), cmap='gray')
else:
    plt.imsave('orig.png', image, cmap='gray')
    plt.imsave('result.png', result, cmap='gray')
    plt.imsave('magn.png', np.fft.fftshift(np.log(magnitudes)), cmap='gray')
