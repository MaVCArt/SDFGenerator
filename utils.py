"""
This is a collection of utilities designed to help with the various functions of the SDF Generator.
"""
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
def flood_fill_pixel_square(shape_mask, id_mask, row, column, target_color):
    """
    Takes a single pixel, and sets its color, as well as its neighbours', to the given target color.
    This behaviour is masked by the shape mask, which makes this act like a flood fill when called recursively.

    :param shape_mask: numpy array containing an image that serves as our mask for the flood fill to be bound by.
    :type shape_mask: np.Array

    :param id_mask: current ID mask intermediate result
    :type id_mask: np.Array

    :param row: the row index to flood fill
    :type row: int

    :param column: the column index to flood fill.
    :type column: int

    :param target_color: the color (a numpy array) to flood fill the center pixel with.
    :type target_color: np.Array

    :return: list of neighbours we can use for a subsequent recursive loop
    :rtype: list
    """
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
        # -- clip behaviour to the bounds of the mask shape
        if _row < 0:
            continue
        if _row >= shape_mask.shape[0]:
            continue
        if _column < 0:
            continue
        if _column >= shape_mask.shape[1]:
            continue

        # -- if the neighbouring pixel is masked, do not fill it. This limits the behaviour to the mask boundaries.
        if shape_mask[_row, _column] == 0.0:
            continue

        neighbours.append((_row, _column))

    return neighbours


# ----------------------------------------------------------------------------------------------------------------------
def color_mask(array, color):
    """
    From a given array of tuples (3 elements), generate a black-white mask (booleans) for all elements in the input
    array that matched all three elements of the given color.

    This functions like a "color ID select".
    """
    r_mask = np.asarray(array[..., 0] == color[0])
    g_mask = np.asarray(array[..., 1] == color[1])
    b_mask = np.asarray(array[..., 2] == color[2])
    return r_mask & g_mask & b_mask


# ----------------------------------------------------------------------------------------------------------------------
def find_image_islands(image_id, lut):
    """
    From a given image, return its islands as distinct numpy arrays that can serve as masks for that island only.
    This aids in finding islands below a certain size, which can help in reducing the noise in a given bitmap.

    :param image_id: RGB input image (must be a PIL Image), image ID
    :type image_id: PIL.Image

    :param lut: RGB input image (must be a PIL Image), LUT
    :type lut: PIL.Image

    :return: List of np.Array instances that contain a mask texture for each of their islands.
    :rtype: list
    """
    masks = list()

    image_id = np.array(image_id).astype(np.float64)
    lut = np.asarray(lut).astype(np.float64)

    # -- for each color in the LUT, we generate a black and white mask for that color and return it
    for column in range(lut.shape[1]):
        # -- get our mask
        color = lut[0, column]

        # -- ignore black pixels
        if color[0] == 0 and color[1] == 0 and color[2] == 0:
            continue

        mask = color_mask(image_id, color)

        # -- now that we have our mask, generate an array from it
        masked = np.zeros([image_id.shape[0], image_id.shape[1]], dtype=int)
        masked[mask] = 1.0

        # -- and return it
        masks.append(masked)

    return masks
