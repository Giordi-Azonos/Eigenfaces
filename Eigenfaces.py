"""
Eigenfaces

Eigenfaces is an excercise from the book: "Coding the Matrix (Linear Algebra through Applications to Computer Science)" By Philip N. Klein - Brown University
Author of Script: Giordi Azonos

"""

#eigenFacesFucntions.py is the script that contains all the necessary functions to work on the project.
import eigenFacesFunctions as myfunc
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt

height = 189
width = 166

def displayFace(pixels_array,as_vector=True, Title='Face'):
    """
    Displays a grayscale matplotlib plot of the pixels 1-D vector, 
    with the given 'Tile'. The pixels vector can also be given as a
    pixels array by flagging the 'as_vector' variable as False. 
    """
    if as_vector == True: 
        face = pixels_array.reshape((height,width))
    else: 
        face = pixels_array
    new_fig = pt.figure()
    new_fig.suptitle(Title, fontsize=14, fontweight='bold')
    new_plot = new_fig.add_subplot(111)
    new_plot.imshow(face, cmap='gray')

def reconstructFaceStepbyStep(originalface_vec, weights_vec, e_faces):
    """
    Receives: 
        .A vector of the face you are trying to re-construct named 'originalface_vec'
        .A weights vector that corresponds to the image you want to re-construct
        .An e_faces matrix that corresponds to the singular values subspace
        
    Displays the re-construction of the given face singular vector by singular vector,
    that is, it first uses just one singular vector, prints the ratio of the 
    reconstruction vs the original face, and plots the re-constructed image vs 
    the original image. And so on for the 20 singular vectors.
    """
    for n in range(1,21):
        # approximation of image using n eigenvectors
        approximation = np.dot(weights_vec[0:n], e_faces[:, 0:n].T)
        # Compute the distance from the approximation to the subspace of eigenfaces
        distance = np.sum(approximation.dot(e_faces)**2)
        ratio = distance/la.norm(originalface_vec)**2  
        print ('    Ratio using '+str(n)+' singular vectors = ' +str(ratio))
        
        if(n==1 or n%5 == 0):
            #I dont want to plot all faces, just a few examples.
            fig = pt.figure(figsize=(20,10))
            fig.suptitle('Original vs Approximation', fontsize=14, fontweight='bold')

            plot1 = fig.add_subplot(121)
            plot1.set_title('Mean Adjusted Original Face')
            plot1.imshow(originalface_vec.reshape((height,width)), cmap='gray')

            plot2 = fig.add_subplot(122)
            plot2.set_title('Reconstructed with '+str(n)+' singular vectors')
            plot2.imshow(approximation.reshape((height,width)), cmap='gray')

# Path where the face images are stored.
faces_path = os.path.join( os.getcwd(), 'faces' )
unclassif_faces_path = os.path.join( os.getcwd(), 'unclassified')

# 1. Load Images to Python. Images are loaded as 1-D vectors of pixels unless 
# stated otherwise in the 'as_vector' variable.
# The resulting array has each row as a vector of pixels for each image.
face_images = myfunc.loadImages(faces_path, as_vector=True)
maybe_faces = myfunc.loadImages(unclassif_faces_path, as_vector=True)

# "Average" face
mu = np.mean(face_images, 0)
displayFace(mu, Title='Mean Face')

# Mean adjusted Data.
ma_data = face_images - mu
ma_maybe_faces = maybe_faces - mu
displayFace(ma_data[0], Title='Mean Adjusted Face 1')

# Perform the SVD decomposition over the mean adjusted array of data.
# The columns of U form an orthonormal basis for the eigenspace of the covariance
# matrix. Each col of U is a singular vector corresponding to an eigenface.
U, S, V = la.svd(ma_data.transpose(), full_matrices=False)
e_faces = U
all_weights = np.dot(ma_data, e_faces) # Each row corresponds to the weights of an image.

# Plot the First Three eigenfaces
# First three eigen faces are the ones that explain most part of the variance
displayFace(U[:,0], Title='Eigen Face 1')
displayFace(U[:,1], Title='Eigen Face 2')
displayFace(U[:,2], Title='Eigen Face 3')

# The columns of U are in decreasing importance, with the first column being the first
# eigen faces with the most importance, that is, the one that explains the most variance
# in the space of faces, and so on. Hence we can work with just the first 10 eigen faces.
n=10
n_e_faces = U[:,0:n]

# Reconstruct eigenvalue-by-eigenvalue face at img_ix
img_ix=0
reconstructFaceStepbyStep(ma_data[img_ix], all_weights[img_ix], e_faces)

#Reconstruction of Faces
# Attempt to reconstruct a face using the first 10 most significant eigen vectors.
# This is important for image compression. You can just save the most important
# eigen vectors, and then reconstruct the image from this eigen vectors, however
# the SVD decomposition is an expensive algorithm.
img_ix = 0
average_distance = 0 
display_images = True
for image in ma_data:
    almost_face = myfunc.approximate(image, n_e_faces) 
    
    dist = myfunc.distanceToEigenspace(image, n_e_faces.T)/1000000
    average_distance += dist
    print('"Distance" of image'+str(img_ix+1)+' to '+str(n)+' e_faces = '+str(format(round(dist), ',f')))
    
    if (display_images and img_ix % 5 == 0): 
        # img%5==0 because I dont want to plot all faces, just a few examples.
        fig = pt.figure(figsize=(20,10))
        fig.suptitle('Face'+str(img_ix+1)+' Original vs Approximation', fontsize=14, fontweight='bold')
            
        plot1 = fig.add_subplot(121)
        plot1.set_title('Original Face')
        plot1.imshow((image+mu).reshape((height,width)), cmap='gray')
            
        plot2 = fig.add_subplot(122)
        plot2.set_title('Reconstructed with '+str(n)+' singular vectors')
        plot2.imshow((almost_face+mu).reshape((height,width)), cmap='gray')
    img_ix += 1
average_distance /= ma_data.shape[0]
print ('Average Distance ='+str(format(round(average_distance), ',f')))

# Classification of Maybe Faces:
# Now we will work with some images that may be faces, or may not be faces, and we will try to classify them
# as faces or not faces.
n=10
# After analizing the maybe faces data set, I reached the conclusion that 
# a distance of the image to the eigen faces subspace greater than 40,000,000
# implies that the image is not a face. From our data set, every image that 
# is a face has a distance to the eigen faces subspace less than 40,000,000
threshold = 40
display_images = True
img_ix = 0
for image in ma_maybe_faces:
    if display_images: displayFace(image+mu, Title='Unclassified Image '+str(img_ix+1))
        
    dist = myfunc.distanceToEigenspace(image, n_e_faces.T)/1000000
    print('"Distance" of image'+str(img_ix+1)+' to '+str(n)+' e_faces = '+str(format(round(dist), ',f')))
    
    if dist > threshold: 
        print ('=> Image '+str(img_ix+1)+' is not a face.')  
    else:
        print ('=> Image '+str(img_ix+1)+' is a face.')
    
    img_ix += 1

pt.show()