# SDF Generator

Simple SDF Generator using Python and Cython / numpy

Note: you will have to compile your own .pyd file if you wish to use this module. 
Setup.py is supplied, and this cython should compile on python 2.7.8 and 3.5.0, on which it was tested.

If you wish to use it for another version,
you will have to compile it using setup.py for your own python version. Please consult the official Cython website
on how to do this.

This uses Cython to accelerate the SDF Calculation, and outputs three PIL Images:

The first, an RGBA image:

- red / green channel: directional field indicating the UV space direction of the distance field
- blue channel: bit mask indicating as white all pixels that were altered in the output data
- alpha channel: traditional SDF (optionally normalized)

The second, an RGB image, with a generated color ID for the SDF, where each island in the original image
is assigned a random color. This information is then used during the SDF calculation to assign newly calculated
pixels the ID of the nearest white pixel they matched.

The final, an RGB image, is a LUT of the color ID. Using these two images combined with the first, some very interesting
shader effects can be achieved.

# Requirements

```
- python 2.7.8+
- numpy
- cython
```


# Usage

## Working with PIL Images

SDFGenerator can be given a PIL Image, and it will assume it is given a black / white image in the
0 - 255 integer range, though it will perform a conversion internally anyway to be sure.

The output of this function is another PIL image that can be saved out directly to retrieve your SDF.

```python
import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')
output = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=100, normalize_distance=True)
output.save('example_assets/output.png')
```

## Working with numpy arrays

If you're working with big data, or you can't or don't want to use PIL for your own reasons, SDFGenerator
can still be helpful to you, as under the hood it actually just operates on numpy arrays (in fact this
is one of the reasons it's so fast). the `generate_sdf_data` function can be given a `numpy.ndarray` of
type `float`, assuming this array contains only `1` and `0` as values. (Boolean arrays probably won't work.)

This function also does not convert, check or filter the input data, so beware when using this directly.

```python
import numpy as np
import SDFGenerator

data = np.ndarray((512, 512), dtype=np.float64)

# -- insert some token data
data[:255] = 1.0
data[255:] = 0.0

# -- get our SDF
output = SDFGenerator.generate_sdf_data(data, spread=25, normalize_distance=True)

```

<table>
<tr>
    <td>Input data</td>
    <td>Raw output data (RGBA)</td>
    <td>Directional Data (RG)</td>
    <td>Data bit mask (B)</td>
    <td>Traditional SDF (Alpha)</td>
</tr>
<tr>
	<td>
		<img src="https://github.com/mavcart/sdfgenerator/blob/main/example_assets/input.png?raw=true" height="100"/>
	</td>
	<td>
		<img src="https://github.com/mavcart/sdfgenerator/blob/main/example_assets/output.png?raw=true" height="100"/>
	</td>
    <td>
		<img src="https://github.com/mavcart/sdfgenerator/blob/main/example_assets/output_rgb.png?raw=true" height="100"/>
	</td>
    <td>
		<img src="https://github.com/mavcart/sdfgenerator/blob/main/example_assets/output_blue.png?raw=true" height="100"/>
	</td>
    <td>
		<img src="https://github.com/mavcart/sdfgenerator/blob/main/example_assets/output_alpha.png?raw=true" height="100"/>
	</td>
</tr>
</table>
