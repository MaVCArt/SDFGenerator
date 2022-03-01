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
import random
import numpy as np
from PIL import Image
from sdf_erosion import calculate_sdf


# ----------------------------------------------------------------------------------------------------------------------
def flood_fill(shape_mask, id_mask, row, column, target_color):
    shape_mask[row, column] = 0.0
    id_mask[row, column] = target_color

    offsets = [
        [row - 1, column],
        [row, column + 1],
        [row + 1, column],
        [row, column - 1],
        [row - 1, column - 1],
        [row + 1, column + 1],
        [row - 1, column + 1],
        [row + 1, column - 1],
    ]

    neighbours = list()

    for _row, _column in offsets:
        if _row < 0:
            continue

        if _row >= shape_mask.shape[0]:
            continue

        if _column < 0:
            continue

        if _column >= shape_mask.shape[1]:
            continue

        if shape_mask[_row, _column] == 0.0:
            continue

        if id_mask[_row, _column] == target_color:
            continue

        neighbours.append((_row, _column))

    return neighbours


# ----------------------------------------------------------------------------------------------------------------------
def generate_image_id(input_image, boolean_cutoff=0.5):
    """
    Generate Image ID by flood filling all disconnected islands with a unique ID value.
    This unique ID will simply be an integer that starts at 0 but increases by 1 for every island.
    This raw data can then be turned into a full RGB image with unique colors per island,
    or it can be channel packed if the user prefers a bit mask.

    :param input_image: input data, boolean field (0/1 float ndarray)
    :type input_image: numpy.ndarray

    :param boolean_cutoff: cutoff value for the input image pre-process.
    :type boolean_cutoff: float

    :return: the generated data as an image with 4 channels
    :rtype: PIL.Image
    """
    # -- filter the input image, converting it to greyscale and ensuring we have floating point values to work with.
    shape_mask = np.asarray(input_image.convert('L')).astype(np.float64) / 255.0

    # -- convert image to boolean field; the algorithm requires absolutes in order to work properly
    shape_mask[shape_mask > boolean_cutoff] = 1.0
    shape_mask[shape_mask <= boolean_cutoff] = 0.0

    id_mask = np.zeros(shape_mask.shape, dtype=int)

    remaining_pixels = shape_mask[shape_mask == 1.0]

    target_value = random.randint(0, 255)

    nr_islands = 0

    last_remaining = len(remaining_pixels)

    row, column = 0, -1
    while len(remaining_pixels):
        column += 1

        if column >= shape_mask.shape[0]:
            column = 0
            row += 1

        if row >= shape_mask.shape[1]:
            break

        # -- scanline approach, one line at a time.
        if shape_mask[row, column] == 0.0:
            continue

        neighbours = flood_fill(shape_mask, id_mask, row, column, target_value)
        while True:
            new_neighbours = list()
            for r, c in neighbours:
                new_neighbours += flood_fill(shape_mask, id_mask, r, c, target_value)
            new_neighbours = list(set(new_neighbours))
            if not new_neighbours:
                break
            neighbours = new_neighbours
            if len(neighbours) > (shape_mask.shape[0] * shape_mask.shape[1]):
                print('Something\'s gone terribly wrong - we have more neighbours than pixels!')
                break

        target_value = random.randint(0, 255)
        nr_islands += 1

        remaining_pixels = shape_mask[shape_mask == 1.0]

        if len(remaining_pixels) == last_remaining:
            print('Infinite loop encountered!')
            break

        last_remaining = len(remaining_pixels)

    print('%s islands!' % nr_islands)

    return Image.fromarray(np.asarray(id_mask * 255))


# ----------------------------------------------------------------------------------------------------------------------
def generate_sdf_data(input_data, spread=25, normalize_distance=True):
    """
    Generate raw SDF Data as a numpy array.

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
