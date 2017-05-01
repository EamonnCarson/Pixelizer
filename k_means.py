import numpy as np

def get_columns_as_list(matrix):
    return np.hsplit(matrix, matrix.shape[1])

def group_by_centroid(points, centroids):
    """ groups the points (columns of points param) by their closest
        centroid
    """
    centroids = get_columns_as_list(centroids)
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
    """ calculates the center of each group.
    """
    centroids = []
    for group in groups:
        centroid = np.average(group, axis=1)
        centroids.append(centroid)
    return np.hstack(centroids)

def k_means(k, points, max_updates=100):
    """ Takes in parameters:
            k = the number of clusters
            points = the set of points to cluster
            max_updates = the maximum iteration count
        returns:
            a list of centroids of the clusters
    """
    # select our k initial centroids
    groups = np.array_split(points, k, axis=1)
    centroids = np.hstack([np.average(group, axis=1) for group in groups])
    old_centroids = None
    iteration = 0
    while iteration < max_updates and not np.array_equal(centroids, old_centroids):
        old_centroids = centroids
        groups = group_by_centroid(points, centroids)
        centroids = calculate_centroids(groups)
        iteration += 1
    return get_columns_as_list(centroids)
