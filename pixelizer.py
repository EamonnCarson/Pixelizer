import numpy as np
import argparse
from k_means import k_means
from PIL import Image
from PIL import ImageFilter
from collections import namedtuple

Group = namedtuple('Group', ['centroid', 'points'])

def _get_rgb_pixel_vectors(image):
    r, g, b = image.split()
    r_values = np.matrix(r)
    size = r_values.size
    r_values = np.reshape(r_values, size)
    g_values = np.reshape(np.matrix(g), size)
    b_values = np.reshape(np.matrix(b), size)
    return np.vstack((r_values, g_values, b_values))

def pixelize(num_colors, input_file, output_file):
    img = Image.open(input_file).convert('RGB')
    img_mod = img.filter(ImageFilter.FIND_EDGES)
    color_points = _get_rgb_pixel_vectors(img)
    primary_colors = k_means(num_colors, color_points)
    print(primary_colors)
    img_mod.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pixelizes an image')
    parser.add_argument('num_colors', metavar='<number of colors>', type=int, help='The number of primary colors to use in the pixelart representation.', default=10)
    parser.add_argument('input_file_path', metavar='<input file>', type=str, help='The path to the image file to be pixellized.')
    parser.add_argument('output_file_path', metavar='<output file>', type=str, help='The location that the result should be written to.')
    args = parser.parse_args()
    pixelize(args.num_colors, args.input_file_path, args.output_file_path)
