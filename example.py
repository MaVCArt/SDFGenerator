import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')

resize_output = (256, 256)

island_id, lut = SDFGenerator.generate_image_id(img, boolean_cutoff=0.5)
island_id.save('example_assets/output_id.png')
lut.save('example_assets/output_id_lut.png')

output = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=20, normalize_distance=True)

output.resize(resize_output).save('example_assets/output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').resize(resize_output).save('example_assets/output_alpha.png')
output.convert('RGB').resize(resize_output).save('example_assets/output_rgb.png')
output.getchannel('B').resize(resize_output).save('example_assets/output_blue.png')
