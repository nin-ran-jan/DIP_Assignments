from PIL import Image
import numpy as np

from scipy.signal import convolve2d
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html

from scipy.fft import fft2
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html

from scipy.fft import ifft2
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.ifft2.html

from scipy.fftpack import fftshift
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.fftshift.html

from scipy.fftpack import ifftshift
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.ifftshift.html

from scipy.signal import gaussian
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.gaussian.html


def unsharp(F, dim_box):
    w = (1/dim_box**2)*np.ones((dim_box, dim_box))
    F_convolved_w = convolve2d(F, w, 'same', boundary="symm")
    unsharp_mask = F - F_convolved_w

    unsharp_masked_image = F + unsharp_mask

    for i in range(len(unsharp_mask)):
        for j in range(len(unsharp_mask[0])):
            if unsharp_masked_image[i][j] < 0:
                unsharp_masked_image[i][j] = 0
            elif unsharp_masked_image[i][j] > 255:
                unsharp_masked_image[i][j] = 255

    highboost_filtered_image = F + 16*unsharp_mask

    for i in range(len(unsharp_mask)):
        for j in range(len(unsharp_mask[0])):
            if highboost_filtered_image[i][j] < 0:
                highboost_filtered_image[i][j] = 0
            elif highboost_filtered_image[i][j] > 255:
                highboost_filtered_image[i][j] = 255
    
    return F_convolved_w.astype(np.uint8), unsharp_masked_image.astype(np.uint8), \
        highboost_filtered_image.astype(np.uint8)

def unsharpUsingFft(f, dim_box):

    w = (1/dim_box**2)*np.ones((dim_box, dim_box))
    f_p = np.zeros((f.shape[0]+w.shape[0]-1, f.shape[1]+w.shape[1]-1))
    f_p[:f.shape[0], :f.shape[1]] = f
    w_p = np.zeros((f.shape[0]+w.shape[0]-1, f.shape[1]+w.shape[1]-1))
    w_p[:w.shape[0], :w.shape[1]] = w

    F_p = fft2(f_p.astype(np.double))
    W_p = fft2(w_p.astype(np.double))

    F_W = np.multiply(F_p, W_p)

    mask = F_p - F_W
    F_unsharp = F_p + mask
    F_highboost = F_p + (4*mask)

    f_unsharp = ifft2(F_unsharp).real
    f_highboost = ifft2(F_highboost).real

    for i in range(len(f_unsharp)):
        for j in range(len(f_unsharp[0])):
            if f_unsharp[i][j] < 0:
                f_unsharp[i][j] = 0
            elif f_unsharp[i][j] > 255:
                f_unsharp[i][j] = 255

    for i in range(len(f_highboost)):
        for j in range(len(f_highboost[0])):
            if f_highboost[i][j] < 0:
                f_highboost[i][j] = 0
            elif f_highboost[i][j] > 255:
                f_highboost[i][j] = 255

    f_unsharp = f_unsharp[:f.shape[0],:f.shape[1]]
    f_highboost = f_highboost[:f.shape[0],:f.shape[1]]

    return f_unsharp.astype(np.uint8), f_highboost.astype(np.uint8)

def addNoise(f, K):
    for i in range(6):
        for j in range(f.shape[1]):
            f[100*i,j] += K
            if f[100*i,j] > 255:
                f[100*i,j] = 255
    
    return f.astype(np.uint8)

def filterImage(f, num):
    max_pixel_output = 255
    img_dim = f.shape[0]

    F = fftshift(fft2(f.astype(np.double)))
    F_log_space = np.log(1+np.abs(F))
    F_to_range = ((F_log_space-np.amin(F_log_space))/np.amax(F_log_space))*max_pixel_output

    printImage(F_to_range.astype(np.uint8), True, "cameraman_magnitude_"+str(num)+".jpg")

    gauss_var = 6
    if num == 20:
        gauss_vert_var = 20
    elif num == 30:
        gauss_vert_var = 15
    elif num == 50:
        gauss_vert_var = 10

    gauss_array = gaussian(img_dim, gauss_var, sym=True)
    gauss_array_vert = gaussian(img_dim, gauss_vert_var, sym=True)
    filter = np.ones((img_dim,img_dim))
    for i in range(filter.shape[0]):
        for j in range(filter.shape[1]):
            filter[i,j] -= gauss_array[j]*(1-gauss_array_vert[i])
    filter = ((filter-np.amin(filter))/np.amax(filter))*max_pixel_output

    printImage(filter.astype(np.uint8), True, "cameraman_filter_"+str(num)+".jpg")

    filter = filter/max_pixel_output

    F_filtered = np.multiply(F,filter)
    F_filtered_log_space = np.log(1+np.abs(F_filtered))
    F_filtered_to_range = ((F_filtered_log_space-np.amin(F_filtered_log_space))/np.amax(F_filtered_log_space))*max_pixel_output

    printImage(F_filtered_to_range.astype(np.uint8), True, "cameraman_filtered_magnitude_"+str(num)+".jpg")

    f_filtered = ifft2(ifftshift(F_filtered)).real

    for i in range(f_filtered.shape[0]):
        for j in range(f_filtered.shape[1]):
            if f_filtered[i,j] < 0:
                f_filtered[i,j] = 0
            elif f_filtered[i,j] > max_pixel_output:
                f_filtered[i,j] = max_pixel_output

    printImage(f_filtered.astype(np.uint8), True, "cameraman_filtered_"\
        +str(num)+"_"+str(gauss_var)+"_"+str(gauss_vert_var)+".jpg")

def printImage(img_arr, save_bool, save_str):
    output_image = Image.fromarray(img_arr, 'L')
    if save_bool:
        output_image.save(save_str)
    output_image.show(title=save_str)

def q1():
    x5 = np.array(Image.open('x5.bmp'))
    printImage(x5, False, "x5.bmp")

    blur, unsharp_masked, highboost_filtered = unsharp(x5, dim_box=5)
    printImage(blur, True, "x5_blur.bmp")
    printImage(unsharp_masked, True, "x5_unsharp.bmp")
    printImage(highboost_filtered, True, "x5_highboost.bmp")


def q2():
    x5 = np.array(Image.open('x5.bmp'))
    printImage(x5, False, "x5.bmp")

    unsharp_fft, highboost_fft = unsharpUsingFft(x5, dim_box=5)
    printImage(unsharp_fft, True, "x5_unsharp_fft.bmp")
    printImage(highboost_fft, True, "x5_highboost_fft.bmp")


def q3():
    #Resizing cameraman because the input dimension given is 256x256

    cameraman_resized = np.array(Image.open('cameraman.jpg').resize((512,512)))
    printImage(cameraman_resized, True, "cameraman_resized.jpg")

    max_pixel_output = 255
    cameraman_resized = cameraman_resized.astype(np.int32)

    F = fftshift(fft2(cameraman_resized.copy().astype(np.double)))
    F_log_space = np.log(1+np.abs(F))
    F_to_range = ((F_log_space-np.amin(F_log_space))/np.amax(F_log_space))*max_pixel_output
    printImage(F_to_range.astype(np.uint8), True, "cameraman_magnitude.jpg")

    cameraman_noisy_20 = addNoise(cameraman_resized.copy(), K=20)
    printImage(cameraman_noisy_20, True, "cameraman_noisy_20.jpg")
    filterImage(cameraman_noisy_20, 20)

    cameraman_noisy_30 = addNoise(cameraman_resized.copy(), K=30)
    printImage(cameraman_noisy_30, True, "cameraman_noisy_30.jpg")
    filterImage(cameraman_noisy_30, 30)

    cameraman_noisy_50 = addNoise(cameraman_resized.copy(), K=50)
    printImage(cameraman_noisy_50, True, "cameraman_noisy_50.jpg")
    filterImage(cameraman_noisy_50, 50)

def main():
    q1()
    q2()
    q3()

if __name__ == "__main__":
    main()