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
def generate_sdf(input_image, boolean_cutoff=0.5, spread=25, normalize_distance=True):
    # -- filter the input image, converting it to greyscale and ensuring we have floating point values to work with.
    data = np.asarray(input_image.convert('L')).astype(np.float64) / 255.0

    # -- convert image to boolean field; the algorithm requires absolutes in order to work properly
    data[data > boolean_cutoff] = 1.0
    data[data <= boolean_cutoff] = 0.0

    # -- this method returns a C "MemoryView" array, which numpy can understand and convert
    result = calculate_sdf(
        bool_field=data,
        radius=spread,
        normalize_distance=normalize_distance,
    )
    result = np.asarray(result)

    # -- return a composed image
    return Image.fromarray(np.uint8(result * 255))
