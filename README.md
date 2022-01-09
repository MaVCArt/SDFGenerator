# SDF Generator

Simple SDF Generator using Python and Cython / numpy

Note: the .pyd file delivered with this project is compiled for Python 2.7.8. If you wish to use it for another version,
you will have to compile it using setup.py for your own python version. Please consult the official Cython website
on how to do this.

This uses Cython to accelerate the SDF Calculation, and outputs a PIL Image object with four channels:

- red / green channel: directional field indicating the UV space direction of the distance field
- blue channel: bit mask indicating as white all pixels that were altered in the output data
- alpha channel: traditional SDF (optionally normalized)

# Requirements

```
- python 2.7.8+
- numpy
- cython
```


# Usage

```python
import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')
output = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=100, normalize_distance=True)
output.save('example_assets/output.png')
```

<table>
<tr>
    <td>Input data</td>
    <td>Raw output data</td>
    <td>Directional Data (RG)</td>
    <td>Data bit mask</td>
    <td>Traditional SDF</td>
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
