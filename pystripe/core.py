import argparse
from argparse import RawDescriptionHelpFormatter
from pathlib import Path
import os, time
import numpy as np
from scipy import fftpack, ndimage
from skimage.filters import threshold_otsu
import tifffile
import imageio as iio
import pywt
import multiprocessing
import tqdm
import random
from dcimg import DCIMGFile
from pystripe import raw
from .lightsheet_correct import correct_lightsheet
import warnings
import shutil
from typing import Optional
import torch, torchvision
import ptwt
import more_itertools as mit
warnings.filterwarnings("ignore")

supported_extensions = ['.tif', '.tiff', '.raw', '.dcimg', '.png']
supported_output_extensions = ['.tif', '.tiff', '.png']
nb_retry = 10


def _get_extension(path):
    """Extract the file extension from the provided path

    Parameters
    ----------
    path : str
        path with a file extension

    Returns
    -------
    ext : str
        file extension of provided path

    """
    return Path(path).suffix


def imread(path):
    """Load a tiff or raw image

    Parameters
    ----------
    path : str
        path to tiff or raw image

    Returns
    -------
    img : ndarray
        image as a numpy array

    """
    img = None
    extension = _get_extension(path)
    if extension == '.raw':
        img = raw.raw_imread(path)
    elif extension == '.tif' or extension == '.tiff':
        img = tifffile.imread(path)
    elif extension == '.png':
        img = iio.imread(path)

    return img


def imread_dcimg(path, z):
    """Load a slice from a DCIMG file

    Parameters
    ------------
    path : str
        path to DCIMG file
    z : int
        z slice index to load

    Returns
    --------
    img : ndarray
        image as numpy array

    """
    with DCIMGFile(path) as arr:
        img = arr[z]
    return img


def check_dcimg_shape(path):
    """Returns the image shape of a DCIMG file

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    shape : tuple
        image shape

    """
    with DCIMGFile(path) as arr:
        shape = arr.shape
    return shape


def check_dcimg_start(path):
    """Returns the starting z position of a DCIMG substack.

    This function assumes a zero-padded 6 digit filename in tenths of micron.
    For example, `0015250.dicom` would indicate a substack starting at z = 1525 um.

    Parameters
    ------------
    path : str
        path to DCIMG file

    Returns
    --------
    start : int
        starting z position in tenths of micron

    """
    return int(os.path.basename(path).split('.')[0])


def imsave(path, img, compression=1, output_format:Optional[str]=None):
    """Save an array as a tiff or raw image

    The file format will be inferred from the file extension in `path`

    Parameters
    ----------
    path : str
        path to tiff or raw image
    img : ndarray
        image as a numpy array
    compression : int
        compression level for tiff writing
    output_format : Optional[str]
        Desired format extension to save the image. Default: None
        Accepted ['.tiff', '.tif', '.png']
    """
    extension = _get_extension(path)

    if output_format is None:
        # Saving any input format to tiff
        if extension == '.raw' or extension == '.png':
            # TODO: get raw writing to work
            # raw.raw_imsave(path, img)
            tifffile.imsave(os.path.splitext(path)[0]+'.tiff', img, compress=compression) # Use with versions <= 2020.9.3
            # tifffile.imsave(os.path.splitext(path)[0]+'.tiff', img, compressionargs={'level': compression}) # Use with version 2023.03.21

        elif extension == '.tif' or extension == '.tiff':
            tifffile.imsave(path, img, compress=compression) # Use with versions <= 2020.9.3
            # tifffile.imsave(path, img, compressionargs={'level': compression}) # Use with version 2023.03.21

    else:
        # Saving output images based on the output format
        if output_format not in supported_output_extensions:
            raise ValueError(f"Output format {output_format} is not valid! Supported extensions are: {supported_output_extensions}")

        filename = os.path.splitext(path)[0] + output_format
        if output_format == '.tif' or output_format == '.tiff':
            tifffile.imsave(filename, img, compress=compression) # Use with versions <= 2020.9.3
            # tifffile.imsave(path, img, compressionargs={'level': compression}) # Use with version 2023.03.21
        
        elif output_format == '.png':
            # print(img.dtype)
            iio.imwrite(filename, img, compress_level=compression) # Works fine up to version 2.15.0
            # iio.v3.imwrite(filename, img, compress_level=compression) # version 2.27.0

def wavedec(img, wavelet, level=None):
    """Decompose `img` using discrete (decimated) wavelet transform using `wavelet`

    Parameters
    ----------
    img : ndarray
        image to be decomposed into wavelet coefficients
    wavelet : str
        name of the mother wavelet
    level : int (optional)
        number of wavelet levels to use. Default is the maximum possible decimation

    Returns
    -------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level

    """
    return pywt.wavedec2(img, wavelet, mode='symmetric', level=level, axes=(-2, -1))

def wavedec_torch(imgs_torch, wavelet, level=None):
    return ptwt.wavedec2(imgs_torch, wavelet, mode='symetric', level=level, axes=(-2, -1))

def waverec(coeffs, wavelet):
    """Reconstruct an image using a multilevel 2D inverse discrete wavelet transform

    Parameters
    ----------
    coeffs : list
        the approximation coefficients followed by detail coefficient tuple for each level
    wavelet : str
        name of the mother wavelet

    Returns
    -------
    img : ndarray
        reconstructed image

    """
    return pywt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))

def waverec_troch(coeffs, wavelet):
    ptwt.waverec2(coeffs, wavelet, mode='symmetric', axes=(-2, -1))

def fft(data, axis=-1, shift=True):
    """Computes the 1D Fast Fourier Transform of an input array

    Parameters
    ----------
    data : ndarray
        input array to transform
    axis : int (optional)
        axis to perform the 1D FFT over
    shift : bool
        indicator for centering the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.rfft(data, axis=axis)
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata

def ftt_torch(data, axis=-1, shift=True):
    fdata = torch.fft.rfft(data, axis=axis)
    # fdata = fftpack.rfft(fdata, axis=0)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata

def ifft(fdata, axis=-1):
    # fdata = fftpack.irfft(fdata, axis=0)
    return fftpack.irfft(fdata, axis=axis)


def fft2(data, shift=True):
    """Computes the 2D Fast Fourier Transform of an input array

    Parameters
    ----------
    data : ndarray
        data to transform
    shift : bool
        indicator for center the DC component

    Returns
    -------
    fdata : ndarray
        transformed data

    """
    fdata = fftpack.fft2(data)
    if shift:
        fdata = fftpack.fftshift(fdata)
    return fdata


def ifft2(fdata):
    return fftpack.ifft2(fdata)


def magnitude(fdata):
    return np.sqrt(np.real(fdata) ** 2 + np.imag(fdata) ** 2)


def notch(n, sigma):
    """Generates a 1D gaussian notch filter `n` pixels long

    Parameters
    ----------
    n : int
        length of the gaussian notch filter
    sigma : float
        notch width

    Returns
    -------
    g : ndarray
        (n,) array containing the gaussian notch filter

    """
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        n = int(n)
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    x = np.arange(n)
    g = 1 - np.exp(-x ** 2 / (2 * sigma ** 2))
    return g


def gaussian_filter(shape, sigma):
    """Create a gaussian notch filter

    Parameters
    ----------
    shape : tuple
        shape of the output filter
    sigma : float
        filter bandwidth

    Returns
    -------
    g : ndarray
        the impulse response of the gaussian notch filter

    """
    g = notch(n=shape[-1], sigma=sigma)
    g_mask = np.broadcast_to(g, shape).copy()
    return g_mask


def hist_match(source, template):
    """Adjust the pixel values of a grayscale image such that its histogram matches that of a target image

    Parameters
    ----------
    source: ndarray
        Image to transform; the histogram is computed over the flattened array
    template: ndarray
        Template image; can have different dimensions to source
    Returns
    -------
    matched: ndarray
        The transformed output image

    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def max_level(min_len, wavelet):
    w = pywt.Wavelet(wavelet)
    return pywt.dwt_max_level(min_len, w.dec_len)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_torch(x):
    return 1 / (1 + torch.exp(-x))
# def sigmoid(x):
#     if x >= 0:
#         z = np.exp(-x)
#         return 1 / (1 + z)
#     else:
#         z = np.exp(x)
#         return z / (1 + z)


def foreground_fraction(img, center, crossover, smoothing):
    z = (img-center)/crossover
    f = sigmoid(z)
    return ndimage.gaussian_filter(f, sigma=smoothing)

def foreground_fraction_torch(imgs_torch, center, crossover, smoothing):
    z = (imgs_torch - center)/crossover
    f = sigmoid_torch(z)
    ks = (8, 8)  # kernal size, set to ~ NDimage defaullt
    return torchvision.transforms.functional.gaussian_blur(f, ks, smoothing)

def filter_subband(img, sigma, level, wavelet):
    img_log = np.log(1 + img)

    if level == 0:
        coeffs = wavedec(img_log, wavelet)
    else:
        coeffs = wavedec(img_log, wavelet, level)
    approx = coeffs[0]
    detail = coeffs[1:]

    width_frac = sigma / img.shape[0]
    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        s = ch.shape[0] * width_frac
        fch = fft(ch, shift=False)
        g = gaussian_filter(shape=fch.shape, sigma=s)
        fch_filt = fch * g
        ch_filt = ifft(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    img_log_filtered = waverec(coeffs_filt, wavelet)
    return np.exp(img_log_filtered)-1


def apply_flat(img, flat):
    return (img / flat).astype(img.dtype)

def apply_flat_torch(imgs_torch, flat):
    if flat.is_tensor() is False:
        flat = torch.from_numpy(flat.astype(np.float32))

    if imgs_torch.get_device() == flat.get_device():
        flat.to(device=imgs_torch.get_device())

    return (imgs_torch / flat)
        


def filter_streaks(img, sigma, level=0, wavelet='db3', crossover=10, threshold=-1, flat=None, dark=0):
    """Filter horizontal streaks using wavelet-FFT filter

    Parameters
    ----------
    img : ndarray
        input image array to filter
    sigma : float or list
        filter bandwidth(s) in pixels (larger gives more filtering)
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.

    Returns
    -------
    fimg : ndarray
        filtered image

    """
    smoothing = 1

    if threshold == -1:
        try:
            threshold = threshold_otsu(img)
        except ValueError:
            threshold = 1

    img = np.array(img, dtype=float) # np.float deprecated in version 1.20

    #
    # Need to pad image to multiple of 2
    #
    pady, padx = [_ % 2 for _ in img.shape]
    if pady == 1 or padx == 1:
        img = np.pad(img, ((0, pady), (0, padx)), mode="edge")

    # TODO: Clean up this logic with some dual-band CLI alternative
    sigma1 = sigma[0]  # foreground
    sigma2 = sigma[1]  # background
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  # Single band
                fimg = filter_subband(img, sigma1, level, wavelet)
            else:  # Dual-band
                background = np.clip(img, None, threshold)
                foreground = np.clip(img, threshold, None)
                background_filtered = filter_subband(background, sigma[1], level, wavelet)
                foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
                # Smoothed homotopy
                f = foreground_fraction(img, threshold, crossover, smoothing=1)
                fimg = foreground_filtered * f + background_filtered * (1 - f)
        else:  # Foreground filter only
            foreground = np.clip(img, threshold, None)
            foreground_filtered = filter_subband(foreground, sigma[0], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = foreground_filtered * f + img * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = np.clip(img, None, threshold)
            background_filtered = filter_subband(background, sigma[1], level, wavelet)
            # Smoothed homotopy
            f = foreground_fraction(img, threshold, crossover, smoothing=1)
            fimg = img * f + background_filtered * (1 - f)
        else:
            # sigma1 and sigma2 are both 0, so skip the destriping
            fimg = img

    # TODO: Fix code to clip back to original bit depth
    # scaled_fimg = hist_match(fimg, img)
    # np.clip(scaled_fimg, np.iinfo(img.dtype).min, np.iinfo(img.dtype).max, out=scaled_fimg)

    # Subtract the dark offset fiirst
    if dark > 0:
        fimg = fimg - dark

    # Divide by the flat
    if flat is not None:
        fimg = apply_flat(fimg, flat)

    # Convert to 16 bit image
    np.clip(fimg, 0, 2**16 - 1, out=fimg)  # Clip to 16-bit unsigned range
    fimg = fimg.astype('uint16')

    if padx > 0:
        fimg = fimg[:, :-padx]
    if pady > 0:
        fimg = fimg[:-pady]
    return fimg


def read_filter_save(output_root_dir, input_path, output_path, sigma, level=0, wavelet='db3',
                     crossover=10, threshold=-1, compression=1,
                     flat=None, dark=0, z_idx=None, rotate=False,
                     lightsheet=False,
                     artifact_length=150,
                     background_window_size=200,
                     percentile=.25,
                     lightsheet_vs_background=2.0,
                     dont_convert_16bit=False, output_format=None):

    """Convenience wrapper around filter streaks. Takes in a path to an image rather than an image array

    Note that the directory being written to must already exist before calling this function

    Parameters
    ----------
    output_root_dir : Path
        highest level output path (error log location)
    input_path : Path
        path to the image to filter
    output_path : Path
        path to write the result
    sigma : list
        bandwidth of the stripe filter
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    compression : int
        compression level for writing tiffs
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    z_idx : int
        z index of DCIMG slice. Only applicable to DCIMG files.
    rotate : bool
        rotate x and y if true
    lightsheet : bool
        if False, use wavelet method, if true use correct_lightsheet
    artifact_length : int
        # of pixels to look at in the lightsheet direction
    background_window_size : int
        Look at this size window around the pixel in x and y
    percentile : float
        Take this percentile as background with lightsheet
    lightsheet_vs_background : float
        weighting factor to use background or lightsheet background
    dont_convert_16bit : bool
        Flag for converting to 16-bit
    output_format: str
        Desired output format [.png, .tiff, .tif]. Default None
    """

    n = 3
    for i in range(n):
        try:
            if z_idx is None:
                # Path must be TIFF or RAW
                img = imread(str(input_path))
                dtype = img.dtype
                if not dont_convert_16bit:
                    dtype = np.uint16
            else:
                # Path must be to DCIMG file
                assert str(input_path).endswith('.dcimg')
                img = imread_dcimg(str(input_path), z_idx)
                dtype = np.uint16
        except:
            if i == n -1:
                file_name = os.path.join(output_root_dir, 'destripe_log.txt')
                if not os.path.exists(file_name):
                    error_file = open(file_name, 'w')
                    error_file.write('Error reading the following images.  Pystripe will interpolate their content.')
                    error_file.close()
                error_file = open(file_name, 'a+')
                error_file.write('\n{}'.format(str(input_path)))
                error_file.close()
                return
            else:
                time.sleep(0.05)
                continue
            # output_dir = os.path.dirname(output_path)
            

    if rotate:
        img = np.rot90(img)
    if not lightsheet:
        fimg = filter_streaks(img, sigma, level=level, wavelet=wavelet, crossover=crossover, threshold=threshold, flat=flat, dark=dark)
    else:
        fimg = correct_lightsheet(
            img.reshape(img.shape[0], img.shape[1], 1),
            percentile=percentile,
            lightsheet=dict(selem=(1, artifact_length, 1)),
            background=dict(
                selem=(background_window_size, background_window_size, 1),
                spacing=(25, 25, 1),
                interpolate=1,
                dtype=np.float32,
                step=(2, 2, 1)),
            lightsheet_vs_background=lightsheet_vs_background
            ).reshape(img.shape[0], img.shape[1])
        if flat is not None:
            fimg = apply_flat(fimg, flat)
    # Save image, retry if OSError for NAS
    for _ in range(nb_retry):
        try:
            imsave(str(output_path), fimg.astype(dtype), compression=compression, output_format=output_format)
        except OSError:
            print('Retrying...')
            continue
        break


def _read_filter_save(input_dict):
    """Same as `read_filter_save' but with a single input dictionary. Used for pool.imap() in batch_filter

    Parameters
    ----------
    input_dict : dict
        input dictionary with arguments for `read_filter_save`.

    """
    # input_path = input_dict['input_path']
    # output_path = input_dict['output_path']
    # sigma = input_dict['sigma']
    # level = input_dict['level']
    # wavelet = input_dict['wavelet']
    # crossover = input_dict['crossover']
    # threshold = input_dict['threshold']
    # compression = input_dict['compression']
    # flat = input_dict['flat']
    # read_filter_save(input_path, output_path, sigma, level, wavelet, crossover, threshold, compression, flat)
    read_filter_save(**input_dict)

def num_ioworkers():
    if os.cpu_count <= 16:
        return 8
    elif os.cpu_count <= 32:
        return 12
    elif os.cpu_count <= 64:
        return 16
    else:
        return int(0.25 * os.cpu_count)

def batch_to_torch16(args_batch, num_ioworkers, img_dims):
        
    if (all(_get_extension(args['input_path']) in ('.tiff', '.tif') for args in args_batch) and len(args_batch) > 1):
        assert(len(set(args['rotation'] for args in args_batch)) == 1), "Batch loading can only be done on inputs with the same `rotation`"
        num_attempts = min(3, len(args_batch))
        # dummy_idxs = random.sample(range(len(args_batch)), attempts)
        file_batch = [str(args['input_path']) for args in args_batch]
        if ioworkers is None or ioworkers < 1:
            ioworkers = num_ioworkers()

        #for dummy_idx in dummy_idxs:
        for _ in range(num_attempts):
            try:
                # if img_dims is None:
                #     dummy = args_batch[dummy_idx]
                #     img_dims_using = imread(str(dummy['input_path'])).shape
                # else:
                #     img_dims_using = img_dims

                batch_ndarry = tifffile.imread(files=file_batch, ioworkers=ioworkers)        
                                    #chunkshape=img_dims_using,
                                    #dtype=np.int16,
                                    #out_inplace=False,
                                    #ioworkers=ioworkers)
                
                if args_batch[0]['rotation']:
                    batch_ndarry = np.rot90(batch_ndarry)
                
                batch_ndarry.dtype = np.int16
                return torch.from_numpy(batch_ndarry)
            except:
                continue
        else:
            print('A tiff batch failed to load. Resorting to single-threaded reading')

    def _single_read(args): # For non-Tiffs or in the event of a corrupted image.
        n = 3
        input_path = args['input_path']
        z_idx = args['z_idx']
        for i in range(n):
            try:
                if z_idx is not None: # Presuming path is a DCIMG file!!
                    assert str(input_path).endswith('.dcimg')
                    img = imread_dcimg(str(input_path), z_idx)
                    if args['rotate']:
                        return np.rot90(img)
                    # dtype = np.uint16
                    return img
                # elif _get_extension(args['input_path']) in ('.tiff', '.tif'): # Best to use tiffile.imread here, lest img_dims is wholly innacurate it will error out later
                #     img = tifffile.imread(str(args['input_path']))
                #                          #files=str(args['input_path']),
                #                           #chunkshape=img_dims,
                #                           #dtype=np.int16,
                #                           #out_inplace=False
                #                           #)
                #     if args['rotate']:
                #         img = np.rot90(img)
                #     # if img.dtype == np.uint16:
                #     #     img.dtype = np.int16
                #     return img
                else:
                    img = imread(str(input_path))
                    if args['rotate']:
                        return np.rot90(img)
                    # dtype = img.dtype
                    # if not dont_convert_16bit:
                    #     dtype = np.uint16
                    return img
            except:
                if i == n -1:
                    file_name = os.path.join(args['output_root_dir'], 'destripe_log.txt')
                    if not os.path.exists(file_name):
                        error_file = open(file_name, 'w')
                        error_file.write('Error reading the following images.  Pystripe will interpolate their content.')
                        error_file.close()
                    error_file = open(file_name, 'a+')
                    error_file.write('\n{}'.format(str(input_path)))
                    error_file.close()
                    args['FAILED_TO_READ'] = True
                    return
                else:
                    time.sleep(0.05)
                    continue

    single_reads = [_single_read(args) for args in args_batch]
    single_reads = list(filter(lambda img: img is not None, single_reads))
    concated = np.concatenate(single_reads)
    if concated.dtype == np.uint16:
        concated.dtype = np.int16
    return torch.from_numpy(concated)
    # try:
    #     return torch.from_numpy(concated)
    # except TypeError:
        # return torch.from_numpy(concated.astype(np.float32))

def offsign16_to_32(int16_tensor):
    shift = int(2**15)
    t = int16_tensor + shift
    t.to(dtype=torch.float32)
    return t + shift

def _find_all_images(search_path, input_path, output_path, zstep=None):
    """Find all images with a supported file extension within a directory and all its subdirectories

    Parameters
    ----------
    input_path : path-like
        root directory to start image search
    zstep : int
        step-size for DCIMG stacks in tenths of micron

    Returns
    -------
    img_paths : list
        a list of Path objects for all found images

    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    search_path = Path(search_path)



    assert search_path.is_dir()
    img_paths = []
    for p in search_path.iterdir():
        if p.is_file():
            if p.suffix in supported_extensions:
                if p.suffix == '.dcimg':
                    if zstep is None:
                        raise ValueError('Unknown zstep for DCIMG slice positions')
                    shape = check_dcimg_shape(str(p))
                    start = check_dcimg_start(str(p))
                    substack = [(p, i, start + i * zstep) for i in range(shape[0])]
                    img_paths += substack
                else:
                    img_paths.append(p)
        elif p.is_dir():
            rel_path = p.relative_to(input_path)
            o = output_path.joinpath(rel_path)
            if not o.exists():
                o.mkdir(parents=True)
            img_paths.extend(_find_all_images(p, input_path, output_path, zstep))

    return img_paths


def batch_filter(input_path, output_path, workers, chunks, sigma, auto_mode, 
                 gpu_acceleration=False, gpu_chunks=int, level=0, wavelet='db3', crossover=10,
                 threshold=-1, compression=1, flat=None, dark=0, zstep=None, rotate=False,
                 lightsheet=False,
                 artifact_length=150,
                 background_window_size=200,
                 percentile=.25,
                 lightsheet_vs_background=2.0,
                 dont_convert_16bit=False,
                 output_format=None
                 ):
    """Applies `streak_filter` to all images in `input_path` and write the results to `output_path`.

    Parameters
    ----------
    input_path : Path
        root directory to search for images to filter
    output_path : Path
        root directory for writing results
    workers : int
        number of CPU workers to use
    chunks : int
        number of images for each CPU to process at a time
    sigma : list
        bandwidth of the stripe filter in pixels
    auto_mode : bool
        Destriping is reccorded automatically by this script. Set to True by if called by
        Joe's Destripe GUI.
    gpu_acceleration : bool
        Use GPU to perform accelerated destriping. Experimental feature, reimplimentation of the
        CPU alogorithm using pytorch may yield minor differences in output.
    gpu_chunks : int
        number of images for GPU to process at a time. Please note that GPU wavelet destriping uses
        on the order of 8 times the amount of memory relative to input. Caution is advised to avoid
        GPU memory limit errors by submitting only a small number of images.
    ioworkers: int
        Maximum number of threads to execute
        :py:attr:`FileSequence.imread` asynchronously.
        If *None*, default to processors multiplied
        by 5.
        Using threads can significantly improve runtime when reading
        many small files from a network share.
    level : int
        number of wavelet levels to use
    wavelet : str
        name of the mother wavelet
    crossover : float
        intensity range to switch between filtered background and unfiltered foreground. Default: 100 a.u.
    threshold : float
        intensity value to separate background from foreground. Default is Otsu
    compression : int
        compression level to use in tiff writing
    flat : ndarray
        reference image for illumination correction. Must be same shape as input images. Default is None
    dark : float
        Intensity to subtract from the images for dark offset. Default is 0.
    zstep : int
        Zstep in tenths of micron. only used for DCIMG files.
    rotate : bool
        Flag for 90 degree rotation.
    dont_convert_16bit : bool
        Flag for converting to 16-bit
    output_format: str
        Desired output format [.png, .tiff, .tif]. Default None
    """

    error_path = os.path.join(output_path, 'destripe_log.txt')
    if os.path.exists(error_path):
        os.remove(error_path)

    print('Looking for images in {}...'.format(input_path))
    img_paths = _find_all_images(input_path, input_path, output_path, zstep)
    print('Found {} compatible images'.format(len(img_paths)))

    if gpu_acceleration:
        if torch.cuda_is_available() is False:
            print('GPU Device is unavailable. Falling back on CPU methodology.')
            gpu_acceleration = False

        if lightsheet:
            print('GPU Acceleration is not available for lightsheet methodology. CPU computation will be used.')
            gpu_acceleration = False

        if any(_get_extension(fpath) not in ('.tiff', '.tif') for fpath in img_paths):
            print('GPU Acceleration is only available for TIF/TIFF formats. CPU de-striping will be used instead.')
            gpu_acceleration = False

    if workers == 0 and gpu_acceleration == False:
        workers = multiprocessing.cpu_count()


    # if auto_mode:
        # count_path = os.path.join(input_path, 'image_count.txt')
        # print('count_path: {} count: {}'.format(count_path, len(img_paths)))
        # with open(count_path, 'w') as fp:
            # fp.write(str(len(img_paths)))
            # fp.close
            
    if auto_mode:
        img_path_strs = list(str(path) for path in img_paths)
        list_path = os.path.join(output_path, 'destriped_image_list.txt')
        # print('writing image_list.  {} images'.format(len(img_path_strs)))
        with open(list_path, 'w') as fp:
            fp.write('\n'.join(img_path_strs) + '\n')
            fp.close
        # print('writing image list: {}'.format(list_path))
            
    # copy text and ini files
    for file in input_path.iterdir():
        if Path(file).suffix in ['.txt', '.ini']:
            output_file = os.path.join(output_path, os.path.split(file)[1])
            shutil.copyfile(file, output_file)
                
    if gpu_acceleration:
        print('Prepping GPU for accelerated destriping...')
    else:
        print('Setting up {} workers...'.format(workers))
    
    args = []
    for p in img_paths:
        if isinstance(p, tuple):  # DCIMG found
            p, z_idx, z = p
            rel_path = p.relative_to(input_path).parent.joinpath('{:04d}.tif'.format(z))
        else:  # TIFF or RAW found
            z_idx = None
            rel_path = p.relative_to(input_path)
        o = output_path.joinpath(rel_path)
        if not o.parent.exists():
            o.parent.mkdir(parents=True)
        arg_dict = {
            'output_root_dir': output_path,
            'input_path': p,
            'output_path': o,
            'sigma': sigma,
            'level': level,
            'wavelet': wavelet,
            'crossover': crossover,
            'threshold': threshold,
            'compression': compression,
            'flat': flat,
            'dark': dark,
            'z_idx': z_idx,
            'rotate': rotate,
            'lightsheet': lightsheet,
            'artifact_length': artifact_length,
            'background_window_size': background_window_size,
            'percentile': percentile,
            'lightsheet_vs_background': lightsheet_vs_background,
            'dont_convert_16bit' : dont_convert_16bit,
            'output_format': output_format,
            'FAILED_TO_READ': None
        }
        args.append(arg_dict)
    print('Pystripe batch processing progress:')
    bar_format = '{l_bar}{bar:60}{r_bar}{bar:-10b}' if auto_mode else None
    if gpu_acceleration:
        gpu_batch_filter(args, gpu_chunks, bar_format=bar_format)
    else:
        with multiprocessing.Pool(workers) as pool:
            list(tqdm.tqdm(
                    pool.imap(_read_filter_save, args, chunksize=chunks), 
                    total=len(args), 
                    ascii=True,
                    bar_format=bar_format))
    
    print('Done!')

    if os.path.exists(error_path):
        with open(error_path, 'r') as fp:
            first_line = fp.readline()
            images = fp.readlines()
            for image_path in images:
                interpolate(image_path, input_path, output_path)
            x = len(images)
            print('{} images could not be opened and were interpolated.  See destripe log for more details'.format(x))
            fp.close()

def destripe_torch32(imgs_torch, imgs_args):
    filter_args  = {'sigma': None,
                    'level': 0,
                    'wavelet': 'db3',
                    'crossover': 10,
                    'threshold': -1,
                    'flat': None,
                    'dark':0 }
    
    for filter_arg in filter_args:
        assert(len(set(img_args[filter_arg] for img_args in imgs_args) == 1)), "Batch de-striping can only be done on inputs with identical filter parameters (`sigma`, `level`, `wavelet`, `crossover`, `threshold`, `flat`, `dark`)"
        filter_args[filter_arg] = imgs_args[0][filter_arg]
    
    smoothing = 1

    if filter_args['threshold'] == -1:
        raise Exception('Please prep threshold prior to destripe_gpu')
    
    assert(imgs_torch.shape[-1] % 2 == 0 and imgs_torch.shape[-2] % 2 == 0), "Image dimensions must be a multiple of two. If non-standard image sizes are being used, contact Ben Kaplan ben.kaplan@lifecanvas.com for support"

    sigma1 = filter_args['sigma'][0] # foreground
    sigma2 = filter_args['sigma'][1] # background
    if sigma1 > 0:
        if sigma2 > 0:
            if sigma1 == sigma2:  #Single band
                fimgs = filter_subbands_gpu(imgs_torch, sigma1, filter_args)
            else:
                background = torch.clip(imgs_torch, None, filter_args['threshold'])
                foreground = torch.clip(imgs_torch, filter_args['threshold'], None)
                background_filtered = filter_subbands_gpu(background, sigma2, filter_args)
                foreground_filtered = filter_subbands_gpu(foreground, sigma1, filter_args)
                # Smoothed homotopy
                f = foreground_fraction_torch(imgs_torch,
                                              filter_args['threshold'],
                                              filter_args['crossover'],
                                              smoothing=smoothing)
                fimgs = foreground_filtered * f + background_filtered * (1 - f)
        else:
            foreground = torch.clip(imgs_torch, filter_args['threshold'], None)
            foreground_filtered = filter_subbands_gpu(foreground, sigma1, filter_args)
            f = foreground_fraction_torch(imgs_torch,
                                          filter_args['threshold'],
                                          filter_args['crossover'],
                                          smoothing=smoothing)
            fimgs = foreground_filtered * f + imgs_torch * (1 - f)
    else:
        if sigma2 > 0:  # Background filter only
            background = torch.clip(imgs_torch, None, filter_args['threshold'])
            background_filtered = filter_subbands_gpu(background, sigma2, filter_args)
            # Smoothed homotopy
            f = foreground_fraction_torch(imgs_torch,
                                          filter_args['threshold'],
                                          filter_args['crossover'],
                                          smoothing=smoothing)
            fimgs = imgs_torch * f + background_filtered * (1 - f)
        else:
            fimgs = imgs_torch

    # Subtract the dark offset fiirst
    if filter_args['dark'] > 0:
        fimgs = fimgs - filter_args['dark']

    # Divide by the flat
    if filter_args['flat'] is not None:
        fimgs = apply_flat_torch(fimgs, filter_args['flat'])

    
    


def filter_subbands_gpu(imgs_torch, use_sigma, filter_args):

    imgs_log = torch.log(1 + imgs_torch)

    if filter_args['level'] == 0:
        coeffs = wavedec_torch(imgs_log, filter_args['wavelet'])
    else:
        coeffs = wavedec_torch(imgs_log, filter_args['wavelet'], filter_args['level'])

    approx = coeffs[0]
    detail = coeffs[1:]

    width_frac = use_sigma / imgs_torch.shape[-2]
    coeffs_filt = [approx]
    for ch, cv, cd in detail:
        s = ch.shape[-2] * width_frac
        fch = ftt_torch(ch, shift=False)
        g = gaussian_filter(shape=fch.shape[-2:], sigma=s)
        g = torch.from_numpy(np.float32(g)).cuda()
        fch_filt = fch * g
        ch_filt = ifft_torch(fch_filt)
        coeffs_filt.append((ch_filt, cv, cd))

    imgs_log_filtered = waverec_torch(coeffs_filt, filter_args['wavelet'])
    return torch.exp(imgs_log_filtered)-1

    


def _prep_threshold(imgs_batch, args_batch):
    assert(len(set(args['threshold'] for args in args_batch)) == 1), "Batch loading can only be done on inputs with the same `threshold`. If threshold is set as -1, then a threshold for a batch will be calculated"

    if args_batch[0]['threshold'] == -1:
        mid_idx = imgs_batch.size[0]/2
        mid_img = imgs_batch[mid_idx].numpy()
        if mid_img.dtype == np.int16:
            mid_img = mid_img.astype(np.uint16, copy=True)
        try:
            threshold = threshold_otsu(mid_img)
        except ValueError:
            threshold = 1

        for args in args_batch:
            args['threshold'] = threshold


def gpu_batch_filter(args, gpu_chunks, bar_format=None):
    chunked_args = mit.chunked(args, gpu_chunks)

    with tqdm.tqdm(total=(len(chunked_args)),ascii=True, bar_format=bar_format) as pbar:    
        last_args_batch = None
        last_imgs_batch = None
        for args_batch in chunked_args:
            # if all(_get_extension(args[input_path] in ['.tiff', '.tif']) for args 
            imgs_batch = batch_to_torch16(args_batch)
            args_batch = list(filter(lambda args: args['FAILED_TO_READ'] != True, args_batch))
            _prep_threshold(imgs_batch, args_batch)

            imgs_batch = imgs_batch.to(device='cuda', non_blocking=False)
            imgs_batch = destripe_torch32(imgs_batch, args_batch)
            imgs_batch = imgs_batch.to(device='cpu', non_blocking=True)

            if last_imgs_batch is not None:
                torch_imwrite(last_imgs_batch, last_args_batch)
                pbar.update(1)

            last_imgs_batch = imgs_batch
            last_args_batch = args_batch
        else:
            torch_imwrite(last_imgs_batch, last_args_batch)
            pbar.update(1)
                    
def torch_imwrite(imgs_batch, args_batch):
    images = imgs_batch.numpy()
    if images.dtype == np.int16:
        images.dtype = np.uint16
    
    for img in images:
        out_path = str(args_batch.pop(0)['output_path'])
        tifffile.imwrite(out_path, img, compression=False)
    

def normalize_flat(flat):
    flat_float = flat.astype(np.float32)
    return flat_float / flat_float.max()


def _parse_args():
    parser = argparse.ArgumentParser(description="Pystripe\n\n"
        "If only sigma1 is specified, only foreground of the images will be filtered.\n"
        "If sigma2 is specified and sigma1 = 0, only the background of the images will be filtered.\n"
        "If sigma1 == sigma2 > 0, input images will not be split before filtering.\n"
        "If sigma1 != sigma2, foreground and backgrounds will be filtered separately.\n"
        "The crossover parameter defines the width of the transistion between the filtered foreground and background",
                                     formatter_class=RawDescriptionHelpFormatter,
                                     epilog='Developed 2018 by Justin Swaney, Kwanghun Chung Lab\n'
                                            'Massachusetts Institute of Technology\n')
    parser.add_argument("--input", "-i", help="Path to input image or path", type=str, required=True)
    parser.add_argument("--output", "-o", help="Path to output image or path (Default: x_destriped)", type=str, default='')
    parser.add_argument("--sigma1", "-s1", help="Foreground bandwidth [pixels], larger = more filtering", type=float, default=0)
    parser.add_argument("--sigma2", "-s2", help="Background bandwidth [pixels] (Default: 0, off)", type=float, default=0)
    parser.add_argument("--level", "-l", help="Number of decomposition levels (Default: max possible)", type=int, default=0)
    parser.add_argument("--wavelet", "-w", help="Name of the mother wavelet (Default: Daubechies 3 tap)", type=str, default='db3')
    parser.add_argument("--threshold", "-t", help="Global threshold value (Default: -1, Otsu)", type=float, default=-1)
    parser.add_argument("--crossover", "-x", help="Intensity range to switch between foreground and background (Default: 10)", type=float, default=10)
    parser.add_argument("--workers", "-n", help="Number of workers for batch processing (Default: # CPU cores)", type=int, default=0)
    parser.add_argument("--chunks", help="Chunk size for batch processing (Default: 1)", type=int, default=1)
    parser.add_argument("--compression", "-c", help="Compression level for written tiffs (Default: 1)", type=int, default=1)
    parser.add_argument("--flat", "-f", help="Flat reference TIFF image of illumination pattern used for correction", type=str, default=None)
    parser.add_argument("--dark", "-d", help="Intensity of dark offset in flat-field correction", type=float, default=0)
    parser.add_argument("--zstep", "-z", help="Z-step in micron. Only used for DCIMG files.", type=float, default=None)
    parser.add_argument("--rotate", "-r", help="Rotate output images 90 degrees counter-clockwise", action='store_true')
    parser.add_argument("--lightsheet", help="Use the lightsheet method", action="store_true")
    parser.add_argument("--artifact-length", help="Look for minimum in lightsheet direction over this length", default=150, type=int)
    parser.add_argument("--background-window-size", help="Size of window in x and y for background estimation", default=200, type=int)
    parser.add_argument("--percentile", help="The percentile at which to measure the background", type=float, default=.25)
    parser.add_argument("--lightsheet-vs-background", help="The background is multiplied by this weight when comparing lightsheet against background", type=float, default=2.0)
    parser.add_argument("--dont-convert-16bit", help="Is the output converted to 16-bit .tiff or not", action="store_true")
    parser.add_argument("--output_format", "-of", help="Desired format output for the images", type=str, required=False, default=None)
    args = parser.parse_args()
    return args


def interpolate(image_path, input_path, output_path):
    # print('Interpolate:\nimage_path: {}\ninput_path: {}\noutput_path: {}\n'.format(image_path, input_path, output_path))
    rel_path = Path(image_path).relative_to(input_path)
    o_dir = os.path.dirname(output_path.joinpath(rel_path))
    # print('other files in output directory:')
    # print(os.listdir(o_dir))
    image_num = int(os.path.splitext(os.path.split(image_path)[1])[0])
    closest_image = {
        'name': os.listdir(o_dir)[0],
        'distance': abs(int(os.path.splitext(os.listdir(o_dir)[0])[0]) - image_num)
        }
    # print('image_num: {}\nclosest image:'.format(image_num))
    # print(closest_image)
    for filename in os.listdir(o_dir):
        try: 
            test_num = int(os.path.splitext(filename)[0])
        except:
            continue
        if abs(test_num - image_num) < closest_image['distance']:
            closest_image['name'] = filename
            closest_image['distance'] = abs(test_num - image_num)
            # print('closest_image:')
            # print(closest_image)
    new_file_name = str(image_num) + os.path.splitext(closest_image['name'])[1]
    try:
        shutil.copyfile(os.path.join(o_dir, closest_image['name']), os.path.join(o_dir, new_file_name))
    except Exception as e:
        # print(e)
        pass


def main():
    args = _parse_args()
    sigma = [args.sigma1, args.sigma2]
    input_path = Path(args.input)

    flat = None
    if args.flat is not None:
        flat = normalize_flat(imread(args.flat))

    zstep = None
    if args.zstep is not None:
        zstep = int(args.zstep * 10)

    if args.output_format not in ['.png', '.tif', '.tiff']:
        raise ValueError("Custom output format not supported.")

    if args.dark < 0:
        raise ValueError('Only positive values for dark offset are allowed')

    if args.output_format is not None and args.output_format not in supported_output_extensions:
        raise ValueError(f"Output format {args.output_format} is currently not supported! Supported formats are: {supported_output_extensions}")

    if input_path.is_file():  # single image
        if input_path.suffix not in supported_extensions:
            print('Input file was found but is not supported. Exiting...')
            return
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(input_path.stem+'_destriped'+input_path.suffix)
        else:
            output_path = Path(args.output)
            assert output_path.suffix in supported_extensions
        output_root_dir = output_path
        
        read_filter_save(output_root_dir,
                         input_path,
                         output_path,
                         sigma=sigma,
                         level=args.level,
                         wavelet=args.wavelet,
                         crossover=args.crossover,
                         threshold=args.threshold,
                         compression=args.compression,
                         flat=flat,
                         dark=args.dark,
                         rotate=args.rotate,  # Does not work on DCIMG files
                         lightsheet=args.lightsheet,
                         artifact_length=args.artifact_length,
                         background_window_size=args.background_window_size,
                         percentile=args.percentile,
                         lightsheet_vs_background=args.lightsheet_vs_background,
                         dont_convert_16bit=args.dont_convert_16bit,
                         output_format=args.output_format
                         )

    elif input_path.is_dir():  # batch processing
        if args.output == '':
            output_path = Path(input_path.parent).joinpath(str(input_path)+'_destriped')
        else:
            output_path = Path(args.output)
            assert output_path.suffix == ''
        batch_filter(input_path,
                     output_path,
                     workers=args.workers,
                     chunks=args.chunks,
                     sigma=sigma,
                     auto_mode=False,
                     level=args.level,
                     wavelet=args.wavelet,
                     crossover=args.crossover,
                     threshold=args.threshold,
                     compression=args.compression,
                     flat=flat,
                     dark=args.dark,
                     zstep=zstep,
                     rotate=args.rotate,
                     lightsheet=args.lightsheet,
                     artifact_length=args.artifact_length,
                     background_window_size=args.background_window_size,
                     percentile=args.percentile,
                     lightsheet_vs_background=args.lightsheet_vs_background,
                     dont_convert_16bit=args.dont_convert_16bit,
                     output_format=args.output_format
                     )
    else:
        print('Cannot find input file or directory. Exiting...')


if __name__ == "__main__":
    main()
