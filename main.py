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
from . import utils
from sdf_generator import calculate_sdf, get_boundaries


# ----------------------------------------------------------------------------------------------------------------------
def generate_progression_gradient(sdf, sdf_id, id_lut):
    pass


# ----------------------------------------------------------------------------------------------------------------------
def generate_id_progression_gradient(sdf, sdf_id, color_id, coord_map):
    """
    Generate a progression gradient for a single ID color. This will return a greyscale image with a gradient that
    progresses along the SDF, tangential to its direction, which is stored in the red and green channels.
    The direction is always clockwise. The starting point will attempt to be chosen as middle top.
    """
    # -- find all coordinates where the RGB value is equal to the given color ID
    coord_pairs = np.asarray(np.where((sdf_id == color_id).all(axis=-1)[..., None]))

    coords = list()

    for i in range(len(coord_pairs[0])):
        column, row = coord_pairs[0][i], coord_pairs[1][i]
        coords.append([column, row])

    result = np.zeros((sdf.shape[0], sdf.shape[1], 2), dtype=np.float64)

    # -- todo: find top-middle pixel at SDF channel 4 value 0.5 (roughly)
    current_coord = random.choice(coords)

    current_value = 1.0

    origin = utils.get_origin_for_coord(coord_map, current_coord)

    mask = coord_map[:, :, :2] == origin
    result[mask] = current_value

    debug_trace = np.zeros((sdf.shape[0], sdf.shape[1], 3), dtype=np.float64)
    debug_trace[current_coord[0], current_coord[1]] = 1.0

    previous_direction = None

    while True:
        # -- perpendicular clockwise vector in 2D is reversed axes, with the x-axis mirrored.
        direction = np.asarray(
            [
                sdf[current_coord[0], current_coord[1], 1],
                -sdf[current_coord[0], current_coord[1], 0]
            ]
        )

        direction += [-0.5, 0.5]
        direction *= 2.0

        next_coord = utils.get_closest_neighbour_in_direction(
            coord_map,
            current_coord,
            direction
        )

        if next_coord is None:
            break

        if sdf[next_coord[0], next_coord[1], 2] == 0.0:
            continue

        if (current_coord == next_coord).all():
            raise ValueError('Next coord cannot be the same as current coord!')

        current_coord = next_coord
        current_value += 1.0

        previous_direction = direction

        origin = utils.get_origin_for_coord(coord_map, current_coord)
        mask = coord_map[:, :, :2] == origin

        result[mask] = current_value
        debug_trace[current_coord[0], current_coord[1]] = 1.0

        print(current_value)

        if current_value > 500.0:
            break

    utils.np_arr_to_img(debug_trace).show()

    return result, current_value


# ----------------------------------------------------------------------------------------------------------------------
def generate_image_id(input_image, use_id_array=False, id_array=None, boolean_cutoff=0.5):
    """
    Generate Image ID by flood filling all disconnected islands with a unique ID value.
    This unique ID will simply be an integer that starts at 0 but increases by 1 for every island.
    This raw data can then be turned into a full RGB image with unique colors per island,
    or it can be channel packed if the user prefers a bit mask.

    :param input_image: RGB input image (must be a PIL Image)
    :type input_image: PIL.Image

    :param id_array: iterable of IDs to use for the IDs used in the output.
    :type id_array: list | np.array

    :param use_id_array:
    :type use_id_array: bool

    :param boolean_cutoff: cutoff value for the input image pre-process.
    :type boolean_cutoff: float

    :return: the generated data as an image with 3 channels. Colors will be assigned randomly or using the provided
        ID array.
    :rtype: PIL.Image
    """
    # -- filter the input image, converting it to greyscale and ensuring we have floating point values to work with.
    shape_mask = np.asarray(input_image.convert('L')).astype(np.float64) / 255.0

    # -- convert image to boolean field; the algorithm requires absolutes in order to work properly
    shape_mask[shape_mask > boolean_cutoff] = 1.0
    shape_mask[shape_mask <= boolean_cutoff] = 0.0

    id_mask = np.zeros((shape_mask.shape[0], shape_mask.shape[1], 3), dtype=int)

    remaining_pixels = shape_mask[shape_mask == 1.0]

    target_value = np.asarray([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    target_value_index = 0
    if use_id_array:
        target_value = id_array[target_value_index]

    ids = list()
    ids.append(target_value)

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

        # -- flood fill the current island
        utils.flood_fill(shape_mask, id_mask, row, column, target_value)

        target_value = np.asarray([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        if use_id_array:
            target_value_index += 1
            if target_value_index >= len(id_array):
                raise ValueError('More islands in image than IDs provided!')
            target_value = id_array[target_value_index]

        ids.append(target_value)

        nr_islands += 1

        remaining_pixels = shape_mask[shape_mask == 1.0]

        if len(remaining_pixels) == last_remaining:
            print('Infinite loop encountered!')
            break

        last_remaining = len(remaining_pixels)

    # -- compose a LUT for the generated IDs
    id_lut = np.asarray([ids])

    return id_mask, id_lut


# ----------------------------------------------------------------------------------------------------------------------
def get_id_islands(img_id, id_lut):
    result = list()

    # -- we only need to iterate over the LUT's first row, as we know it's a 1-row image.
    for column in range(id_lut.shape[1]):
        where = np.where((img_id == id_lut[0][column]).all(axis=-1)[..., None])
        result.append(where)

    return np.asarray(result)


# ----------------------------------------------------------------------------------------------------------------------
def calculate_sdf_color_id(img_id, coord_map, sdf):
    result = np.zeros((sdf.shape[0], sdf.shape[1], 3), dtype=np.int)

    for row in range(img_id.shape[0]):
        for column in range(img_id.shape[1]):
            # -- no need to do anything if there is no SDF information here.
            if sdf[column, row, 2] == 0:
                continue

            a, b = coord_map[column, row][:2]
            result[column, row] = img_id[a, b]

    return result


# ----------------------------------------------------------------------------------------------------------------------
def calculate_boundaries(input_image, boolean_cutoff=0.5):
    data = utils.img_to_bool_field(input_image, bool_cutoff=boolean_cutoff)
    data = get_boundaries(data)
    return np.asarray(data)


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

    :return: (SDF, coord map)
    :rtype: tuple
    """
    data = utils.img_to_bool_field(input_image, bool_cutoff=boolean_cutoff)

    # -- this method returns a C "MemoryView" array, which numpy can understand and convert
    sdf, coord_map = calculate_sdf(
        bool_field=data,
        radius=spread,
        normalize_distance=normalize_distance,
    )

    # -- convert to numpy ndarray
    sdf = np.asarray(sdf)
    coord_map = np.asarray(coord_map)

    return sdf, coord_map
