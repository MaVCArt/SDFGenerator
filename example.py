import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')
output = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=20, normalize_distance=True)
output.save('example_assets/output.png')

# -- save the output data as individual images to illustrate functionality
output.getchannel('A').save('example_assets/output_alpha.png')
output.convert('RGB').save('example_assets/output_rgb.png')
output.getchannel('B').save('example_assets/output_blue.png')
