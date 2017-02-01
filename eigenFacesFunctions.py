import os
import scipy.misc as sp
import numpy.linalg as la
import numpy as np
import matplotlib.pyplot as pt
from orthogonality import orthogonalComponent

def loadImages(path, as_vector=True):
    """
    Input:
        - path: path to directory containing img*.png
        - A flag 'as_vector' indicating if you want each image to be stored
          as a 2-D array of pixels, or as a 1-D vector of pixels.
    Output:
        When each image is stored as a 1-D vector of pixels:
        - 2-D Numpy array with each row being a vector of pixels. 
        When each image is stored as a 2-D array of pixels:
        - 3-D Numpy array with entries being 2-D arrays of pixels.
    """
    i=0
    images = []
    for file in os.listdir(path):
        if file.endswith('.png'): # Its an image
            image_name = os.path.join(path,"img%02d.png" % i)
            im_array = sp.imread(image_name, True)
            if as_vector: im_array = im_array.flatten()
            images.append(im_array)
            i += 1
    return np.array(images).astype(np.float64)
    
def approximate(originalface_vec, e_faces):
    """
    Approximation of the original face vector using the given vectors from the
    eigenfaces subspace.
    """
    weights = np.dot(originalface_vec, e_faces) # Each row corresponds to the weights of an image.
    approximation = np.dot(weights, e_faces.T)
    return approximation
    
def computeRatioOfFaceness(originalface_vec, approximation, e_faces):
    """
    Faceness is a measure of how much is left after projecting, used for image classification
    to determine if a given image is a face.
    
    Is defined as the norm of the projection vector over the norm of the original face vector.
    Projection being constructed using the sigular vectors.
    """
    distance = np.sum(originalface_vec.dot(e_faces)**2)
    ratio = distance/la.norm(originalface_vec)**2 
    return ratio
    
def distanceToEigenspace(x, subspace, as_rows=True):
    if as_rows == False: subspace = subspace.T
    return la.norm(orthogonalComponent(x, subspace))**2
    
        