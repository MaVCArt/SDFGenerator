import SDFGenerator
from PIL import Image

# -- black/white input image
img = Image.open('example_assets/input.png')

resize_output = (256, 256)

# -- because we happen to know the number of islands we will have, we pre-assign these.
id_lookup = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [128, 128, 0],
    [128, 0, 128],
    [0, 128, 128],
    [50, 50, 0],
]

# -- generate a color ID, which also gives us back a LUT we can use to tie IDs to their corresponding islands.
img_id, id_lut = SDFGenerator.generate_image_id(img, use_id_array=True, id_array=id_lookup, boolean_cutoff=0.5)

# _img_id = SDFGenerator.utils.np_arr_to_img(img_id / 255.0)
# _img_id.resize(resize_output).save('example_assets/input_id.png')
#
# _id_lut = SDFGenerator.utils.np_arr_to_img(id_lut / 255.0)
# _id_lut.save('example_assets/input_id_lut.png')

id_islands = SDFGenerator.get_id_islands(img_id, id_lut)

output, coord_map = SDFGenerator.generate_sdf(img, boolean_cutoff=0.5, spread=50, normalize_distance=True)

sdf_id = SDFGenerator.calculate_sdf_color_id(img_id, coord_map, output)
SDFGenerator.utils.np_arr_to_img(sdf_id / 255.0).resize(resize_output).save('example_assets/output_id.png')

# # -- dividing this by the image resolution normalizes the coord map in UV space
# coord_map = SDFGenerator.utils.np_arr_to_img(coord_map / 1024.0)
# coord_map.resize(resize_output).save('example_assets/output_coordmap.png')
#
# output = SDFGenerator.utils.np_arr_to_img(output)
# output.resize(resize_output).save('example_assets/output.png')
#
# # -- save the output data as individual images to illustrate functionality
# output.getchannel('A').resize(resize_output).save('example_assets/output_alpha.png')
# output.convert('RGB').resize(resize_output).save('example_assets/output_rgb.png')
# output.getchannel('B').resize(resize_output).save('example_assets/output_blue.png')

id_progress, max_value = SDFGenerator.generate_id_progression_gradient(output, sdf_id, id_lookup[0], coord_map)
SDFGenerator.utils.np_arr_to_img(id_progress / max_value).show()
