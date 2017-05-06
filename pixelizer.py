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

def image_from_pixel_vectors(mode, pixel_vectors, image_dimensions):
    shape = image_dimensions[::-1] # since array coordinates are reversed from image coordinates
    color_array = np.zeros(shape + (3,), 'uint8')
    color_array[..., 0] = pixel_vectors[0, :].reshape(shape)
    color_array[..., 1] = pixel_vectors[1, :].reshape(shape)
    color_array[..., 2] = pixel_vectors[2, :].reshape(shape)
    return Image.fromarray(color_array, mode)

def pixelize(num_colors, input_file, output_file):
    # get the RGB color points of the pixels in the input_file
    img = Image.open(input_file)
    img.show('Original Image')
    img = img.resize((256,256))
    pixel_vectors = get_hsv_pixel_vectors(img)
    # find primary colors
    pixel_HS_vectors = pixel_vectors[:2, :] # we only care about hue and saturation
    primary_colors = find_primary_colors(pixel_HS_vectors, num_colors)
    generate_palette_image(primary_colors).show('Image Palette')
    # Recolor the points based on closest primary color
    recolored_pixel_vectors = recolor(pixel_vectors, primary_colors, 32)
    pixelized_image = image_from_pixel_vectors('HSV', recolored_pixel_vectors, img.size)
    pixelized_image.show('HSV pixelized image')
    # show edges
    img_mod = img.filter(ImageFilter.FIND_EDGES)
    #img_mod.show()

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
            hsv_vector = np.vstack([color, value * 256 // num_shades])
            block = np.hstack([hsv_vector] * 32 * 32)
            blocks.append(block)
    pixel_vectors = np.hstack(blocks)
    return image_from_pixel_vectors('HSV', pixel_vectors, (32 * num_shades, 32 * len(primary_colors)))

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
