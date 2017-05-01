import numpy as np
import argparse
from PIL import Image
from PIL import ImageFilter
from collections import namedtuple

Group = namedtuple('Group', ['centroid', 'points'])

def get_rgb_pixel_vectors(image):
    r, g, b = image.split()
    r_values = np.matrix(r)
    g_values = np.matrix(g)
    b_values = np.matrix(b)
    return np.vstack((r_values, g_values, b_values))

def group_by_centroid(points, centroids):
    """ groups the points (columns of points param) by their closest
        centroid
    """
    centroids = np.hsplit(centroids, centroids.shape[1])
    # calculate distances from points to centroids
    distances = []
    for centroid in centroids: 
        distance_to_centroid = np.linalg.norm(points - centroid, axis=0) #0 is columns
        distances.append(distance_to_centroid)
    distances = np.vstack(distances)
    # calculate index of nearest centroid
    nearest_centroid = np.argmin(distances, axis=0)
    # group the points by centroid index
    groups = []
    for (centroid_index, centroid) in enumerate(centroids):
        group_points = np.compress(nearest_centroid == centroid_index, points, axis=1) 
        if group_points.size == 0:
            # in the rare case that no points are closest to our centroid, just
            # print out an error message so we can debug
            print("Oops! we just lost a centroid it seems. Sorry!")
        else:
            groups.append(group_points)
    return groups

def calculate_centroids(groups):
    centroids = []
    for group in groups:
        centroid = np.average(group, axis=1)
        centroids.append(centroid)
    return np.hstack(centroids)

def k_means(k, points, max_updates=100):
    # select k random positions vectors from points to be our initial centroids
    centroids = np.hstack([np.average(group, axis=1) for group in np.hsplit(points, k)])
    old_centroids = None
    iteration = 0
    while iteration < max_updates and not np.array_equal(centroids, old_centroids):
        old_centroids = centroids
        groups = group_by_centroid(points, centroids)
        centroids = calculate_centroids(groups)
        iteration += 1
    return centroids

def main(input_file, output_file):
    img = Image.open(input_file).convert('RGB')
    img_mod = img.filter(ImageFilter.FIND_EDGES)
    get_rgb_pixel_vectors(img)
    img_mod.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pixellizes an image')
    parser.add_argument('input_file_path', metavar='<input_file>', type=str, help='The path to the image file to be pixellized.')
    parser.add_argument('output_file_path', metavar='<output_file>', type=str, help='The location that the result should be written to.')
    arguments = parser.parse_args()
    main(arguments.input_file_path, arguments.output_file_path)
