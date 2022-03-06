import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------------------------------------------------
def img_to_bool_field(img, bool_cutoff=0.5):
    """
    Convert a PIL Image instance to a bit mask with only black or white values.

    :param img: Black / White input image (must be a PIL Image)
    :type img: PIL.Image

    :param bool_cutoff: cutoff value for the input image pre-process.
    :type bool_cutoff: float

    :return: numpy array as a boolean field (float type)
    :rtype: np.array
    """
    # -- filter the input image, converting it to greyscale and ensuring we have floating point values to work with.
    data = np.asarray(img.convert('L')).astype(np.float64) / 255.0

    # -- convert image to boolean field; the algorithm requires absolutes in order to work properly
    data[data > bool_cutoff] = 1.0
    data[data <= bool_cutoff] = 0.0

    return data


# ----------------------------------------------------------------------------------------------------------------------
def flood_fill(shape_mask, id_mask, row, column, target_color):
    """
    Flood fill the given id mask starting at the given row and column.

    :param shape_mask: boolean shape mask setting boundaries for all flood fill islands.
    :type shape_mask: np.array

    :param id_mask: base id mask array, will be manipulated directly.
    :type id_mask: np.array

    :param row: the row to start the flood fill from
    :type row: int

    :param column: the column to start the flood fill from
    :type column: int

    :param target_color: the color to fill with
    :type target_color: np.array

    :return: None
    """
    neighbours = flood_fill_single_pixel(shape_mask, id_mask, row, column, target_color)

    while True:
        new_neighbours = list()
        for r, c in neighbours:
            new_neighbours += flood_fill_single_pixel(shape_mask, id_mask, r, c, target_color)

        new_neighbours = list(set(new_neighbours))
        if not new_neighbours:
            break

        neighbours = new_neighbours
        if len(neighbours) > (shape_mask.shape[0] * shape_mask.shape[1]):
            print('Something\'s gone terribly wrong - we have more neighbours than pixels!')
            break


# ----------------------------------------------------------------------------------------------------------------------
def flood_fill_single_pixel(shape_mask, id_mask, row, column, target_color):
    """
    Flood fill method that performs a flood fill on the ID mask starting from the row, column, and fills from there
    using the ID mask as a boundary. This will essentially flood fill the ID mask from the starting pixel using the
    "target-color" argument, bound by the shape mask, assuming the shape mask is a bit mask (black or white).

    This method only fills one pixel and its immediate neighbours, and returns any remaining neighbours that remain
    in need of filling.

    :param shape_mask: boolean shape mask setting boundaries for all flood fill islands.
    :type shape_mask: np.array

    :param id_mask: base id mask array, will be manipulated directly.
    :type id_mask: np.array

    :param row: the row to start the flood fill from
    :type row: int

    :param column: the column to start the flood fill from
    :type column: int

    :param target_color: the color to fill with
    :type target_color: np.array

    :return: list of remaining neighbours that have not been flood filled.
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

        neighbours.append((_row, _column))

    return neighbours


# ----------------------------------------------------------------------------------------------------------------------
def np_arr_to_img(arr):
    return Image.fromarray(np.uint8(arr * 255))
