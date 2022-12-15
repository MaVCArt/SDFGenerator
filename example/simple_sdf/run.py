import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('input.png')

resize_output = (256, 256)

output = SDFGenerator.generate_sdf(
    img,
    boolean_cutoff=0.5,
    spread=50,
    normalize_distance=True
)

output.resize(resize_output).save('output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').resize(resize_output).save('output_alpha.png')
output.convert('RGB').resize(resize_output).save('output_rgb.png')
output.getchannel('B').resize(resize_output).save('output_blue.png')
