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
