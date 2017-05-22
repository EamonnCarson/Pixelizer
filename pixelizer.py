import numpy as np
import argparse
from k_means import k_means
from PIL import Image
from PIL import ImageFilter

def get_hsv_pixel_vectors(image):
    image = image.convert('HSV')
    h, s, v = image.split()
    h_values = np.array(h)
    size = h_values.size
    h_values = np.reshape(h_values, size)
    s_values = np.reshape(np.array(s), size)
    v_values = np.reshape(np.array(v), size)
    return np.vstack((h_values, s_values, v_values))

def image_from_pixel_array(mode, pixel_vectors, image_dimensions):
    shape = image_dimensions[::-1] # since array coordinates are reversed from image coordinates
    color_array = np.zeros(shape + (3,), 'uint8')
    color_array[..., 0] = pixel_vectors[0, :].reshape(shape)
    color_array[..., 1] = pixel_vectors[1, :].reshape(shape)
    color_array[..., 2] = pixel_vectors[2, :].reshape(shape)
    return Image.fromarray(color_array, mode)

def image_from_pixel_vectors(mode, pixel_vectors, image_dimensions):
    shape = image_dimensions[::-1] # since array coordinates are reversed from image coordinates
    color_array = np.zeros(shape + (3,), 'uint8')
    color_array[..., 0] = pixel_vectors[0].reshape(shape)
    color_array[..., 1] = pixel_vectors[1].reshape(shape)
    color_array[..., 2] = pixel_vectors[2].reshape(shape)
    return Image.fromarray(color_array, mode)

def median_hue_categorization(image, num_categories):
    """
    Return a set of categories based on hue.
    The see of categories will have cardinality less than num_hues.
    Each category approximates a percentile range with width 100% / num_hues.
    :param image: PILLOW Image
        Source image to categorize.
    :param num_categories: Integer
        maximum number of categories.
    :return: List of Integer Vectors
        A list of flat arrays.
        Each flat array contains the indices of pixels in a category.
    """
    # extract the hue vector from the image
    image = image.convert('HSV')
    h_raw, _, _ = image.split()
    h = np.array(h_raw)
    percentile_categorization(h, num_categories);

def render_categories(image, categories, hue_palette=None):
    if not hue_palette:
        # generate a diverse palette
        hue_palette = np.floor(np.linspace(0, 255, len(categories), endpoint=False))
    size = image.size[0] * image.size[1]
    result_hue = np.zeros(size)
    for (category, pixels) in enumerate(categories):
        result_hue[pixels] = hue_palette[category]
    result_sv = 128 * np.ones(size)
    return image_from_pixel_vectors('HSV', (result_hue, result_sv, result_sv), image.size)

def pixel_categorization(num_colors, input, output):
    img = Image.open(input)
    img.show('Original Image')
    categories = median_hue_categorization(img, num_colors)
    categories_rendered = render_categories(img, categories)
    categories_rendered.show('Categorized')

def find_luminosity_extrema(image, cutoff_ratio=8):
    """
    Given an image, finds the extremes of luminosity
    (e.g. what is 'black' and 'white' in this picture)
    :param image: 
        The image to be analyzed
    :param cutoff_ratio:
        The amount of extreme values to be considered
        when calculating b and w. For example, a 
        cutoff_ratio of 8 would consider only the
        brightest and darkest 1/8ths of the image
        when calculating b and w.
    :return: 
        a tuple (b, w) where
            b is the lowest luminosity color
            w is the highest luminosity color
    """
    image = image.convert('HSV')
    hsv = get_hsv_pixel_vectors(image)
    v = hsv[2, :]
    brightness_categories = percentile_categorization(v, cutoff_ratio)
    b_category = brightness_categories[0]
    b = np.rint(np.average(hsv[:, b_category], axis=1))
    w_category = brightness_categories[-1]
    w = np.rint(np.average(hsv[:, w_category], axis=1))
    return (b, w)

def percentile_categorization(data, num_categories):
    """
    Categorizes elements of the np.array data by sorting and
    then categorizing by percentiles. Guaranteed that if two
    colors are the same then they will be in the same categ-
    ory. Best to use only for data where elements are conti-
    guous.
    :param data: 
        A list of integers
    :param num_categories:
        The number of categories to create
    :return: 
        A list of np.arrays, where each np.array holds all
        indices of data that belong to a category.
    """
    data = data.ravel()
    frequencies = np.bincount(data)
    # calculate range index-width
    range_width = np.floor(data.size / num_categories)
    # get category boundaries (a list of exclusive upper bounds of
    # the hue range of a category)
    category_boundaries = []
    width = 0
    for (index, frequency) in enumerate(frequencies):
        if width > range_width:
            category_boundaries.append(index)
            width = 0
        width += frequency
    # return categories
    categories = []
    prev_boundary = -1
    for boundary in category_boundaries:
        in_bounds = np.logical_and(prev_boundary < data, data <= boundary)
        categories.append(np.argwhere(in_bounds))
        prev_boundary = boundary
    categories.append(np.argwhere(prev_boundary < data))
    return categories;

# Debug Methods

def display_colors(colors, mode='HSV', blocksize=64):
    blocks = []
    base = np.ones((blocksize, blocksize))
    for color in colors:
        block = np.hstack([color] * blocksize * blocksize)
        blocks.append(block)
    pixel_array = np.hstack(blocks)
    output = image_from_pixel_array(mode, pixel_array, (blocksize, len(colors) * blocksize))
    output.show()

def _test_lumin_extrema(input):
    image = Image.open(input)
    image.show()
    b, w = find_luminosity_extrema(image)
    display_colors((b, w))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pixelizes an image')
    parser.add_argument('num_colors', metavar='<number of colors>', type=int,
                        help='The number of primary colors to use in the pixelart representation.', default=10)
    parser.add_argument('input_file_path', metavar='<input file>', type=str,
                        help='The path to the image file to be pixellized.')
    parser.add_argument('output_file_path', metavar='<output file>', type=str,
                        help='The location that the result should be written to.')
    args = parser.parse_args()
    # pixelize(args.num_colors, args.input_file_path, args.output_file_path)
    # pixel2(args.num_colors, args.input_file_path, args.output_file_path)
    # pixel_categorization(args.num_colors, args.input_file_path, args.output_file_path)
    _test_lumin_extrema(args.input_file_path)
