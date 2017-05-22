import numpy as np
import argparse
from k_means import k_means
from PIL import Image
from PIL import ImageFilter

def get_hsv_pixel_vectors(image):
    image = image.convert('HSV')
    h, s, v = image.split()
    h_values = np.matrix(h)
    size = h_values.size
    h_values = np.reshape(h_values, size)
    s_values = np.reshape(np.matrix(s), size)
    v_values = np.reshape(np.matrix(v), size)
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

def pixelize(num_colors, input_file, output_file):
    # get the RGB color points of the pixels in the input_file
    img = Image.open(input_file)
    img.show('Original Image')
    img = img.resize((512,512))
    pixel_vectors = get_hsv_pixel_vectors(img)
    # find primary colors
    pixel_HS_vectors = pixel_vectors[:2, :] # we only care about hue and saturation
    primary_colors = find_primary_colors(pixel_HS_vectors, num_colors)
    generate_palette_image(primary_colors).show('Image Palette')
    # Recolor the points based on closest primary color
    recolored_pixel_vectors = recolor(pixel_vectors, primary_colors, 32)
    pixelized_image = image_from_pixel_array('HSV', recolored_pixel_vectors, img.size)
    # pixelized_image.show('HSV pixelized image')
    output_location = 'output/{:s}.png'.format(output_file)
    output_image = pixelized_image.convert('RGB').resize((512, 512), Image.NEAREST)
    output_image.show('output')
    output_image.save(output_location, 'PNG')
    # show edges
    # img_mod = img.filter(ImageFilter.FIND_EDGES)
    # img_mod.show()


def find_primary_colors(color_points, num_colors):
    return k_means(num_colors, color_points)

def recolor(pixel_vectors, primary_colors, step_size):
    # primary colors
    primary_color_vectors = np.hstack(primary_colors)
    # normalize the primary colors
    normalized_primary_colors = [color / np.linalg.norm(color) for color in primary_colors]
    # get pixel color bands
    pixel_HS_vectors = pixel_vectors[:2, :]
    pixel_V_vectors  = pixel_vectors[2, :]

    # project each pixel's color onto the primary color directions
    projections = []
    for color in normalized_primary_colors:
        projections.append(color.T * pixel_HS_vectors)
    projections = np.vstack(projections)
    # get the maximal primary color of each pixel
    recolor_colors = np.argmax(projections, axis=0)
    # this line is used to make it a flat array
    recolor_colors = np.array(recolor_colors).reshape(recolor_colors.size)
    # index into primary_color_vectors to get recolored_
    recolored_HS_vectors = primary_color_vectors[:, recolor_colors]
    # multiply by maximum HSV value, round to nearest integer

    # discretize the values with step_size
    discretized_pixel_values = np.rint(pixel_V_vectors / step_size) * step_size
    recolored_V_vectors = np.clip(discretized_pixel_values, 0, 256 - 1)

    # Recombine HS and V to get recolored pixel vectors and output
    recolored_pixel_vectors = np.vstack([recolored_HS_vectors,
                                         recolored_V_vectors]);
    return recolored_pixel_vectors

def generate_palette_image(primary_colors):
    # 32 x 32 blocks of color
    num_shades = 4
    blocks = []
    for color in primary_colors:
        for value in range(num_shades):
            hsv_vector = np.vstack([color, value * 255 // num_shades])
            block = np.hstack([hsv_vector] * 32 * 32)
            blocks.append(block)
    pixel_vectors = np.hstack(blocks)
    return image_from_pixel_array('HSV', pixel_vectors, (32 * num_shades, 32 * len(primary_colors)))

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
    h = np.array(h_raw).ravel()
    # create frequency table
    frequencies = np.bincount(h)
    # calculate range index-width
    range_width = np.floor(h.size / num_categories)
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
        in_bounds = np.logical_and(prev_boundary < h, h <= boundary).ravel()
        categories.append(np.argwhere(in_bounds))
        prev_boundary = boundary
    categories.append(np.argwhere(prev_boundary < h).ravel())
    return categories;

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

def pixel_builtin(num_colors, input, output):
    img = Image.open(input)
    img.show('Original Image')
    quantized = img.resize((512, 512)).quantize(num_colors, method=0)
    quantized.show(title='Median')
    quantized = img.quantize(num_colors, method=1)
    #quantized.show(title='Max Coverage')
    quantized = img.quantize(num_colors, method=2)
    #quantized.show(title='Octree')

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
    pixel_categorization(args.num_colors, args.input_file_path, args.output_file_path)

