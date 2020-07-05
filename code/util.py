import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision
import numpy as np
import math
import torch
import os
from skimage.measure import compare_ssim as _ssim
from skimage.feature import register_translation
from scipy.signal import convolve2d

def plot(image):
    if len(image.shape) == 3 and image.shape[0] == 1:
        image = image[0]
    if len(image.shape) == 3:
        if type(image) is np.ndarray:
            plt.imshow(np.clip(image, 0, 1).transpose(1,2,0))
        else:
            plt.imshow(torch.clamp(image, 0, 1).permute(1,2,0))
    else:
        if type(image) is np.ndarray:
            plt.imshow(np.clip(image, 0, 1), cmap="gray")
        else:
            plt.imshow(torch.clamp(image, 0, 1), cmap="gray")
    plt.show()


def plot_grid(images, file=None, grid_size=8, figsize=None):
    """
    Expects 4d tensor with shape (B, C, H, W)
    plot will be saved if file is not none
    """
    if type(images) is np.ndarray:
        images = torch.from_numpy(images).float()
    images_concat = torchvision.utils.make_grid(images, nrow=grid_size, padding=2, pad_value=255)
    if file is not None: torchvision.utils.save_image(images_concat.clone(), file)
    if figsize is not None: plt.figure(figsize=figsize)
    plt.imshow(np.transpose(images_concat.numpy(), (1,2,0)), interpolation='nearest')
    plt.show()
    

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def phase_correlation(moving, fixed):
    assert(moving.shape == fixed.shape)

    if moving.shape[-1] == 3:
        moving_gray = rgb2gray(moving)
        fixed_gray = rgb2gray(fixed)
    elif moving.shape[-1] == 1:
        moving_gray = moving[..., 0]
        fixed_gray = fixed[..., 0]
    else:
        print("Image channel Error!")
    
    ft_moving = np.fft.fft2(moving_gray)
    ft_fixed = np.fft.fft2(fixed_gray)
    prod = ft_moving * ft_fixed.conj()
    a = (prod / (np.abs(prod) + 1e-16))
    corr = np.fft.ifft2(a)
    corr_max = np.max(corr)
    idx = np.unravel_index(np.argmax(corr), corr.shape)
    out = np.roll(moving, -1*np.array(idx), axis=(0, 1))
    return out, corr_max


def cross_correlation(moving, fixed):
    
    if moving.shape[-1] == 3:
        moving_gray = rgb2gray(moving)
        fixed_gray = rgb2gray(fixed)
    elif moving.shape[-1] == 1:
        moving_gray = moving[..., 0]
        fixed_gray = fixed[..., 0]
    else:
        print("Image channel Error!")
    
    shift, error, diffphase = register_translation(moving_gray, fixed_gray)
    out = np.roll(moving, -np.array(shift).astype(np.int), axis=(0, 1))
    return out, error


def register_croco(predicted_images, true_images, torch=True):
    pred_reg = np.empty(predicted_images.shape, dtype=predicted_images.dtype)

    for i in range(len(true_images)):
        if torch:
            true_image = true_images[i].transpose(1, 2, 0)
            predicted_image = predicted_images[i].transpose(1, 2, 0)
        else:
            true_image = true_images[i]
            predicted_image = predicted_images[i]

        shift_predict, shift_error = cross_correlation(predicted_image, true_image)
        rotshift_predict, rotshift_error = cross_correlation(np.rot90(predicted_image, k=2, axes=(0, 1)), true_image)
        
        if torch:
            pred_reg[i] = shift_predict.transpose(2, 0, 1) if shift_error <= rotshift_error else rotshift_predict.transpose(2, 0, 1)
        else:
            pred_reg[i] = shift_predict if shift_error <= rotshift_error else rotshift_predict
        
    return pred_reg



def sharp_dist(predicted_images, true_images):
    if predicted_images.shape[1]==3:
        predicted_images_gray = np.transpose(predicted_images,(0,2,3,1))
        predicted_images_gray = rgb2gray(predicted_images_gray)[:, None]
    else:
        predicted_images_gray = predicted_images
    
    if true_images.shape[1]==3:
        true_images_gray = np.transpose(true_images,(0,2,3,1))
        true_images_gray = rgb2gray(true_images_gray)[:, None]
    else:
        true_images_gray = true_images
    
    dists = []
    for true, predicted in zip(true_images_gray, predicted_images_gray):
        f = np.array([[1,-1]])
        filtered_true = convolve2d(true[0], f, mode='valid')[:-1,:]\
            +convolve2d(true[0], f.T, mode='valid')[:,:-1]
        filtered_predicted = convolve2d(predicted[0], f,mode='valid')[:-1,:]\
            +convolve2d(predicted[0], f.T, mode='valid')[:,:-1]
        dists.append(np.mean(np.abs(filtered_true-filtered_predicted)))
    return np.mean(dists), np.std(dists)


def register_phaco(predicted_images, true_images, torch=True):
    pred_reg = np.empty(predicted_images.shape, dtype=predicted_images.dtype)
    
    for i in range(len(true_images)):
        if torch:
            true_image = true_images[i].transpose(1, 2, 0)
            predicted_image = predicted_images[i].transpose(1, 2, 0)
        else:
            true_image = true_images[i]
            predicted_image = predicted_images[i]
            
        shift_predict, shift_corr = phase_correlation(predicted_image, true_image)
        rotshift_predict, rotshift_corr = phase_correlation(np.rot90(predicted_image, k=2, axes=(0, 1)), true_image)
        
        if torch:
            pred_reg[i] = shift_predict.transpose(2, 0, 1) if shift_corr >= rotshift_corr else rotshift_predict.transpose(2, 0, 1)
        else:
            pred_reg[i] = shift_predict if shift_corr_max >= rotshift_corr_max else rotshift_predict

    return pred_reg


def mse(predicted_images, true_images):
    # Expect shape of N x (1,3) x X x Y
    N = len(true_images)
    total_se = np.empty((N))
    
    for i in range(len(true_images)):    
        total_se[i] = np.mean((true_images[i] - predicted_images[i]) ** 2)
        
    return np.mean(total_se), np.std(total_se)


def ssim(predicted_images, true_images):
    # Expect shape of N x [1,3] x X x Y
    assert(predicted_images.shape == true_images.shape)
    assert(predicted_images.shape[-3] <= 3)
    N = len(true_images)
    total_ssim = np.empty((N))

    for i in range(len(true_images)):    
        total_ssim[i] = _ssim(true_images[i].transpose(1, 2, 0), predicted_images[i].transpose(1, 2, 0), multichannel=True)
        
    return np.mean(total_ssim), np.std(total_ssim)


def mae(predicted_images, true_images):
    # Expect shape of N x (1,3) x X x Y
    N = len(true_images)
    total_ae = np.empty((N))
    
    for i in range(len(true_images)):    
        total_ae[i] = np.mean(np.abs(true_images[i] - predicted_images[i]))
        
    return np.mean(total_ae), np.std(total_ae)
    

def magn_mse(predicted_images, true_images):
    # Expect shape of N x (1,3) x X x Y
    N = len(true_images)
    total_se = np.empty((N))
    
    for i in range(len(true_images)):    
        true_image = true_images[i]
        predicted_image = predicted_images[i]

        true_magn = np.abs(np.fft.fft2(true_image))
        predicted_magn = np.abs(np.fft.fft2(predicted_image))

        total_se[i] = np.mean((true_magn - predicted_magn) ** 2)
        
    return np.mean(total_se), np.std(total_se)


def benchmark(pred, true, check_all=False, check=["mse", "magn", "imcon"]):
    # expects numpy arrays
    
    pred_signal = np.real(pred)
    true_signal = np.real(true)
    
    checks = [e.lower() for e in check]
    
    pred_croco = register_croco(pred_signal, true_signal)
    pred_phaco = register_phaco(pred_signal, true_signal)
    
    markdown = ""
    
    print("Signal error:")
    if "mse" in checks or check_all:
        _mse = mse(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_mse[0], 4 + math.floor(-math.log10(_mse[0])))
        print("  MSE: {}, std: {}".format(*_mse))
    if "mae" in checks or check_all:
        _mae = mae(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_mae[0], 4 + math.floor(-math.log10(_mae[0])))
        print("  MAE: {}, std: {}".format(*_mae))
    if "ssim" in checks or check_all:
        _ssim = ssim(pred_signal, true_signal)
        markdown = markdown + " {:.{}f} |".format(_ssim[0], 4 + math.floor(-math.log10(abs(_ssim[0]))))
        print("  SSIM: {}, std: {}".format(*_ssim))
    if "sharpness" in checks or check_all:
        _sharpness = sharp_dist(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_sharpness[0], 4 + math.floor(-math.log10(_sharpness[0])))
        print("  Sharpness: {}, std: {}".format(*_sharpness))
    if "phaco" in checks or check_all:
        _fasimse = mse(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasimse[0], 4 + math.floor(-math.log10(_fasimse[0])))
        print("  PhCo-MSE: {}, std: {}".format(*_fasimse))
        _fasimae = mae(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasimae[0], 4 + math.floor(-math.log10(_fasimae[0])))
        print("  PhCo-MAE: {}, std: {}".format(*_fasimae))
        _fasissim = ssim(pred_phaco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_fasissim[0], 4 + math.floor(-math.log10(abs(_fasissim[0]))))
        print("  PhCo-SSIM: {}, std: {}".format(*_fasissim))
    if "croco" in checks or check_all:
        _crocomse = mse(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocomse[0], 4 + math.floor(-math.log10(_crocomse[0])))
        print("  CroCo-MSE: {}, std: {}".format(*_crocomse))
        _crocomae = mae(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocomae[0], 4 + math.floor(-math.log10(_crocomae[0])))
        print("  CroCo-MAE: {}, std: {}".format(*_crocomae))
        _crocossim = ssim(pred_croco, true_signal)
        markdown = markdown + " {:.{}f} |".format(_crocossim[0], 4 + math.floor(-math.log10(abs(_crocossim[0]))))
        print("  CroCo-SSIM: {}, std: {}".format(*_crocossim))
    if "magn" in checks or check_all:
        _magn = magn_mse(pred, true)
        markdown = markdown + " {:.{}f} |".format(_magn[0], 4 + math.floor(-math.log10(_magn[0])))
        print()
        print("Magnitude error:")
        print("  MSE Magnitude: {}, std: {}".format(*_magn))
    if "imcon" in checks or check_all:
        print()
        print("Image constraints:")
        print("  Imag part =", np.mean(np.imag(pred)), "- should be very close to 0")
        print("  Real part is in [{0:.2f}, {1:.2f}]".format(np.min(np.real(pred)), np.max(np.real(pred))), "- should be in [0, 1]")
    
    print()
    print("Markdown table values:")
    print(markdown)
