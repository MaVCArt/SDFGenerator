# cython: profile=True
# cython: language=C++

import cython
import numpy as np
from cython.parallel import prange

DTYPE = np.float64


# ----------------------------------------------------------------------------------------------------------------------
cpdef object shift(object arr, int num, object fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


# ----------------------------------------------------------------------------------------------------------------------
cpdef object get_row_boundaries(object row):
    rising_diff = row - shift(row, 1, fill_value=0)
    decreasing_diff = row - shift(row, -1, fill_value=0)
    return np.maximum(rising_diff, decreasing_diff)


# ----------------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef object get_boundaries(object data):
    cdef object rows = data.copy().astype(float)

    # -- row boundaries
    for i in range(len(data)):
        rows[i] = get_row_boundaries(data[i])

    cdef object columns = data.copy().T
    cdef object column_data = data.T

    for i in range(len(column_data)):
        columns[i] = get_row_boundaries(column_data[i])
    columns = columns.T

    return np.maximum(rows, columns)


# ----------------------------------------------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cpdef tuple calculate_sdf(
        object bool_field,
        object id_map,
        int radius=25,
        int mode=2, # -- 0: external | 1: internal | 2: dual
        bint normalize_distance=True,
):
    cdef int width = bool_field.shape[0]
    cdef int height = bool_field.shape[1]

    # -- get boundary field, this gives us the pixels we want to erode.
    cdef object boundary_field = get_boundaries(bool_field).astype(float)

    if id_map.shape[:2] != bool_field.shape:
        raise ValueError('id map shape must match bool field shape: %s / %s' % (id_map.shape[:2], bool_field.shape))

    # -- get all pixels to erode
    cdef double[:, :] bool_field_arr = bool_field.astype(float)

    # -- get all boundary coords
    cdef int[:, :] coords = np.asarray(np.where(boundary_field == 1.0)).T.astype(int)

    # -- declare a three-dimensional array, distance in the alpha channel, direction in
    # -- red and green, black/white bit mask in blue, denoting pixels that were touched
    # -- and pixels that were not

    # -- this output data, when converted to a numpy array, can be directly converted to an
    # -- RGBA image, with distance info in the alpha channel for maximum bit depth.
    # -- this is declared as a 3D double, because it creates a cython memory slice,
    # -- which acts as a pre-allocated array in memory, such that we can operate on it
    # -- from multiple threads at once.
    cdef double[:, :, :] result = np.full(
        (bool_field.shape[0], bool_field.shape[1], 4),
        0.0,
        dtype=DTYPE
    )

    # -- copy id map to an easily accessible array which can be accessed from a thread
    cdef int[:, :, :] id_map_result = id_map.astype(int)

    # -- copy id map to an easily accessible array which can be accessed from a thread
    cdef int[:, :, :] id_map_original = id_map.astype(int)

    # -- float converts to double implicitly in C
    cdef double max_size = <float> radius

    # -- this prevents cython from adding zero division safeties in the C code below
    if max_size == 0.0:
        raise ZeroDivisionError

    # -- pre-declaring cython variables as we cannot do so in a prange
    cdef double vector_length, new_value = 0.0
    cdef int coord_range = len(coords)

    # -- allocating iterator variables
    cdef int i, x, y, column, row
    cdef int min_x, max_x, min_y, max_y = 0

    # -- this is what makes this so fast; the "prange" method means it runs in parallel on as many threads
    # -- as cython can use.
    # -- "nogil" makes this even faster, as it ensures all code in this statement is run in pure C.
    # -- the gotcha however is that no python object interactions can happen inside the loop.
    for i in prange(coord_range, nogil=True, schedule='static'):
        column, row = coords[i][0], coords[i][1]

        min_x, max_x = max(0, column - radius), min(width, column + radius)
        min_y, max_y = max(0, row - radius), min(height, row + radius)

        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                # -- don't attempt to change pixels with a value of 1.0, as there is no higher value.
                if result[x, y][3] >= 1.0:
                    continue

                # -- erosion is equal to the original value minus the distance of the current
                # -- pixel to the original one. This gives us euclidean distance, rather than
                # -- squared distance.
                vector_length = (((x - column) ** 2 + (y - row) ** 2) ** 0.5)

                # -- zero division protection, this is perfectly valid but should be avoided.
                if vector_length <= 0.0:
                    continue

                # -- normalize value between 0 and 1, so the output is automatically
                # -- remapped.
                new_value = (max_size - vector_length) / max_size

                # -- the distance value is stored in the alpha channel, e.g. channel 4
                if new_value < result[x, y][3]:
                    continue

                # -- normalize the vector before storing it
                result[x, y][0] = (x - column) / vector_length
                result[x, y][1] = (y - row) / vector_length

                # -- the SDF mask is stored in the blue channel
                result[x, y][2] = 1.0

                # -- the distance value is stored in the alpha channel
                result[x, y][3] = new_value

                # -- track which ID we matched as closest
                # -- we assign individual channels because otherwise we engage the GIL or python, which slows
                # -- things down A LOT
                id_map_result[x, y, 0] = id_map_original[column, row, 0]
                id_map_result[x, y, 1] = id_map_original[column, row, 1]
                id_map_result[x, y, 2] = id_map_original[column, row, 2]

    # -- bucket values for min/max remapping of the alpha channel
    # -- these are safe, as we know values do not exceed the 0-1 range at this stage
    cdef float min_value = 10.0
    cdef float max_value = -10.0

    # -- remap alpha channel so that we have one long smooth gradient from 0 to 1 from inside to outside.
    for x in range(width):
        for y in range(height):
            # -- early out for untouched pixels
            # -- this is a cheat to check for NaN
            if result[x, y][3] != result[x, y][3]:
                continue

            if mode == 0:
                if bool_field_arr[x, y] == 1.0:
                    result[x, y][3] = 0.0
                    result[x, y][4] = 0.0

            if mode == 1:
                if bool_field_arr[x, y] == 0.0:
                    result[x, y][3] = 0.0
                    result[x, y][4] = 0.0

            if mode == 2:
                result[x, y][0] = result[x, y][0] * 0.5 + 0.5
                result[x, y][1] = result[x, y][1] * 0.5 + 0.5

                # -- remap based on bool field if necessary
                if bool_field_arr[x, y] == 1.0:
                    result[x, y][3] = 1.0 + (1.0 - result[x, y][3])

                # -- divide by 2 in every case
                result[x, y][3] /= 2.0

            # -- protect nan comparison; this returns True if that value is NaN
            if result[x, y][3] != result[x, y][3]:
                continue

            if result[x, y][3] < min_value:
                min_value = result[x, y][3]

            if result[x, y][3] > max_value:
                max_value = result[x, y][3]

    # -- if the user does not want to normalize the distance field, return here.
    if not normalize_distance:
        return result, id_map_result

    if max_value == min_value:
        raise ZeroDivisionError('if max and min are identical, this results in a zero division!')

    if max_value != max_value or min_value != min_value:
        raise ValueError('min / max value cannot be NaN!')

    # -- remap the alpha channel between 0 and 1 so that we increase the contrast a bit,
    # -- which helps when mapping huge ranges.
    for x in prange(width, nogil=True, schedule='static'):
        for y in range(height):
            result[x, y][3] = (result[x, y][3] - min_value) / (max_value - min_value)

    return result, id_map_result
