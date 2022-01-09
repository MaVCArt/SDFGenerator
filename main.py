"""
MIT License

Copyright (c) 2022 Mattias Van Camp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy as np
from PIL import Image
from sdf_erosion import calculate_sdf


# ----------------------------------------------------------------------------------------------------------------------
def generate_sdf_data(input_data, spread=25, normalize_distance=True):
    """

    :param input_data: input data, boolean field (0/1 float ndarray)
    :type input_data: numpy.ndarray

    :param spread: the number of pixels to "spread" the distance field.
    :type spread: int

    :param normalize_distance: if True, will calculate the distance field and _then_ normalize the output.
    :type normalize_distance: bool

    :return: numpy array
    :rtype: numpy.ndarray
    """
    # -- this method returns a C "MemoryView" array, which numpy can understand and convert
    result = calculate_sdf(
        bool_field=input_data,
        radius=spread,
        normalize_distance=normalize_distance,
    )
    
    # -- convert to numpy ndarray
    result = np.asarray(result)

    # -- return the result
    return result


# ----------------------------------------------------------------------------------------------------------------------
def generate_sdf(input_image, boolean_cutoff=0.5, spread=25, normalize_distance=True):
    """
    Given a black/white input image, generate a Signed Distance field as output, with the following
    description:

    - Red / Green Channel: Direction
    - Blue Channel: Bitmask of Distance Field
    - Alpha Channel: Greyscale Signed Distance Field normalized between 0 and 1

    The input image will be pre-processed by converting it to float and dividing it by 255.
    
    The boolean_cutoff value will further be used to ensure the input image is a real bit mask,
    by setting any values in it that are below this value to 0, and any others to 1.

    If the "normalize_distance" parameter is set to True, the alpha channel output, which contains
    the standard Signed Distance Field output, will be normalized as a post-process, ensuring that
    its values are normalized between 0 and 1, as relative to the final distance values, _not_
    the spread value.

    This is particularly useful when you are dealing with big "spread" values, as the output values
    tend to be very low-contrast. This step somewhat addresses this by measuring the output's minimum
    and maximum values, and normalizing all values using those new extremes. This effectively ensures
    that the output contains the maximum contrast in the values calculated.

    If you want "true" output values, disable this setting, but for most purposes this should be left
    to the default value of True.

    :param input_image: Black / White input image (must be a PIL Image)
    :type input_image: PIL.Image

    :param boolean_cutoff: cutoff value for the input image pre-process.
    :type boolean_cutoff: float

    :param spread: the number of pixels to "spread" the distance field.
    :type spread: int

    :param normalize_distance: if True, will calculate the distance field and _then_ normalize the output.
    :type normalize_distance: bool

    :return: the generated data as an image with 4 channels
    :rtype: PIL.Image
    """
    # -- filter the input image, converting it to greyscale and ensuring we have floating point values to work with.
    data = np.asarray(input_image.convert('L')).astype(np.float64) / 255.0

    # -- convert image to boolean field; the algorithm requires absolutes in order to work properly
    data[data > boolean_cutoff] = 1.0
    data[data <= boolean_cutoff] = 0.0

    result = generate_sdf_data(data, spread, normalize_distance)

    # -- return a composed image
    return Image.fromarray(np.uint8(result * 255))
