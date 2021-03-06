import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')

resize_output = (256, 256)

output, color_id, lut = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=50, normalize_distance=True)

color_id.resize(resize_output).save('example_assets/output_id.png')
lut.save('example_assets/output_id_lut.png')

output.resize(resize_output).save('example_assets/output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').resize(resize_output).save('example_assets/output_alpha.png')
output.convert('RGB').resize(resize_output).save('example_assets/output_rgb.png')
output.getchannel('B').resize(resize_output).save('example_assets/output_blue.png')
