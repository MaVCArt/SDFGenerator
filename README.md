<div>
    <img src="https://github.com/MaVCArt/SDFGenerator/blob/main/example/demo/standard_sdf_demo.gif?raw=true" height="256"/>
    <img src="https://github.com/MaVCArt/SDFGenerator/blob/main/example/demo/id_sdf_demo.gif?raw=true" height="256"/>
</div>

# Requirements

```
- python 2.7.8+
- numpy
- cython
```

# SDF Generator

Simple SDF Generator using Python and Cython / numpy

Note: you will have to compile your own .pyd file if you wish to use this module. 
Setup.py is supplied, and this cython should compile on python 2.7.8+.

The code is currently mainly being tested on Python 3.10, so backwards compatibility is not guaranteed. However, the
modules used are intentionally "vanilla" to maintain maximum compatibility with external use.

This uses Cython to accelerate the SDF Calculation and is capable of generating various types of SDF and SDF-adjacent
images.

# Usage

## Simple SDF

The example below, taken from `run.py` under "example/simple_sdf", shows how to generate a simple SDF and extract
the images you might want.

By default, the output image is generated using the following spec;

- Red and Green Channels: directional SDF. Encodes not the distance, but the direction to the nearest mask pixel.
- Blue Channel: Alpha. Encodes all pixels within the range of the mask pixels as pure black/white.
- Alpha: Traditional SDF. This is encoded in the alpha to provide maximum bit range when imported into game engines.

```python
import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('input.png')

resize_output = (256, 256)

output = SDFGenerator.generate_sdf(
    img,
    boolean_cutoff=0.5,
    spread=50,
    normalize_distance=False,
)

output.resize(resize_output).save('output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').resize(resize_output).save('output_alpha.png')
output.convert('RGB').resize(resize_output).save('output_rgb.png')
output.getchannel('B').resize(resize_output).save('output_blue.png')
```

### Simple SDF Usafe Example

<img src="https://github.com/MaVCArt/SDFGenerator/blob/main/example/demo/standard_sdf_demo.gif?raw=true" height="256"/>

## SDF With Image ID

The example below, taken from `run.py` under "example/sdf_with_image_id", shows how to generate a simple SDF, paired 
with a color ID map and a LUT (Lookup Texture) containing all colours in the ID map.

The SDF texture is encoded using the same standard and spec as the previous example, however the Image ID and LUT bear
some additional explanation.

The Image ID is generated internally, based on a random Colour ID that gets assigned to each "island" in the mask
texture provided. "Islands" here are defined as free-floating contiguous pixel "sections" that can be flood filled.

This "colour ID" image provides the user with the ability to combine this with the LUT and SDF textures to achieve some
otherwise hard-to-do effects in shaders.

```python
import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('input.png')

resize_output = (256, 256)

output, color_id, lut = SDFGenerator.generate_sdf_with_image_id(
    img,
    boolean_cutoff=0.5,
    spread=50,
    normalize_distance=True
)

color_id.resize(resize_output).save('output_id.png')
lut.save('output_id_lut.png')

output.resize(resize_output).save('output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').resize(resize_output).save('output_alpha.png')
output.convert('RGB').resize(resize_output).save('output_rgb.png')
output.getchannel('B').resize(resize_output).save('output_blue.png')
```

### Image ID SDF Usafe Example

<img src="https://github.com/MaVCArt/SDFGenerator/blob/main/example/demo/id_sdf_demo.gif?raw=true" height="256"/>
