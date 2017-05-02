import numpy as np
import argparse
from k_means import k_means
from PIL import Image
from PIL import ImageFilter
from collections import namedtuple

def get_rgb_pixel_vectors(image):
    r, g, b = image.split()
    r_values = np.matrix(r)
    size = r_values.size
    r_values = np.reshape(r_values, size)
    g_values = np.reshape(np.matrix(g), size)
    b_values = np.reshape(np.matrix(b), size)
    return np.vstack((r_values, g_values, b_values))

def image_from_color_points(color_points, image_dimensions):
    shape = image_dimensions[::-1] # since array coordinates are reversed from image coordinates
    rgb_array = np.zeros(shape + (3,), 'uint8')
    rgb_array[..., 0] = color_points[0, :].reshape(shape)
    rgb_array[..., 1] = color_points[1, :].reshape(shape)
    rgb_array[..., 2] = color_points[2, :].reshape(shape)
    return Image.fromarray(rgb_array)

def pixelize(num_colors, input_file, output_file):
    # get the RGB color points of the pixels in the input_file
    img = Image.open(input_file).convert('RGB')
    img = img.resize((256,256))
    img.show()
    color_points = get_rgb_pixel_vectors(img)
    # find primary colors
    primary_colors = find_primary_colors(color_points, num_colors)
    generate_palette(primary_colors).show()
    recolored_points = recolor(color_points, primary_colors, 80)
    pixellized_image = image_from_color_points(recolored_points, img.size)
    pixellized_image.show()
    # show edges
    img_mod = img.filter(ImageFilter.FIND_EDGES)
    #img_mod.show()

def find_primary_colors(color_points, num_colors):
    return k_means(num_colors, color_points)

def recolor(color_points, primary_colors, step_size):
    # normalize the primary colors
    primary_colors_list = [color / np.linalg.norm(color) for color in primary_colors]
    primary_color_points = np.hstack(primary_colors_list)
    print(primary_color_points.shape)
    # project each pixel's color onto the primary color directions
    projections = []
    for color in primary_colors_list:
        projections.append(color.T * color_points)
    projections = np.vstack(projections)
    # get the maximal primary color of each pixel
    recolor_colors = np.argmax(projections, axis=0)
    # this line is used to make the recolor a flat array
    recolor_colors = np.array(recolor_colors).reshape(recolor_colors.size)
    print(recolor_colors.shape)
    # get the discrete (e.g. multiples of step-size) magnitude of colors
    recolor_magnitudes = np.amax(projections, axis=0)
    recolor_magnitudes = step_size * np.floor(recolor_magnitudes / step_size)
    print(recolor_magnitudes.shape)
    # generate color array
    recolored_points = primary_color_points[:, recolor_colors]
    print(recolored_points.shape)
    shaded_recolored_points = np.multiply(recolored_points, recolor_magnitudes)
    return shaded_recolored_points

def generate_palette(primary_colors):
    # 32 x 32 blocks of color
    blocks = []
    for color in primary_colors:
        block = np.hstack([color] * 32 * 32)
        blocks.append(block)

    color_points = np.hstack(blocks)
    print(color_points.shape)
    return image_from_color_points(color_points, (32, 32 * len(primary_colors)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pixelizes an image')
    parser.add_argument('num_colors', metavar='<number of colors>', type=int,
                        help='The number of primary colors to use in the pixelart representation.', default=10)
    parser.add_argument('input_file_path', metavar='<input file>', type=str,
                        help='The path to the image file to be pixellized.')
    parser.add_argument('output_file_path', metavar='<output file>', type=str,
                        help='The location that the result should be written to.')
    args = parser.parse_args()
    pixelize(args.num_colors, args.input_file_path, args.output_file_path)
