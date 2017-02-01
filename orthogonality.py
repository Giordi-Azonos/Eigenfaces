"""
File: orthogonality.py
Autor: Giordi Azonos Beverido
-----------------------------
Contains functions regarding orthogonality of vectors
and matrices. 

Orthogonality is such an important topic in linear algebra
that it desrves its own file. Having a good knowledge about
orthogonality lets you compute test functions to see
if a vector has independent rows, compute the basis and rank
of a matrix, build orthogonal projections, QR factorization,
gramSchmidt and so on.
"""

import numpy as np
import numpy.linalg as la

def correctType(vector):
	"""
	Is used to cast all variables to numpy arrays, so the functions
	in this module can work with numpy arrays.

	If a given value is a list, it casts the list to numpy array, else
	it raises an error stating that the value must be either a list or a numpy array.
	"""
	if not isinstance(vector, np.ndarray):
		if isinstance(vector,list): vector = np.array(vector) # if its a list, convert it to numpy array
		else: raise ValueError("value '" +str(vector)+"' must be either a list or a numpy array")
	return vector

def areOrthogonal(vec_list):
	"""
	Returns true if a given list of vectors consits
	of mutually orthogonal vectors.
	"""
	if len(vec_list) == 0: return True
	for c in range(len(vec_list)):
		v0 = correctType(vec_list[c])
		for i in range(c+1,len(vec_list)):
			v = correctType(vec_list[i])
			if (v0.dot(v) < 1E-08): return False # Returns false if one element of the list is not orthogonal to other element
	return True

def isMatrixOrthogonal(matrix):
	"""
	Returns True if the columns of a matrix are mutually orthogonal,
	and each one has length one.
	"""
	assert len(matrix) != 0
	matrix = correctType(matrix)
	n_cols = matrix.shape[1]
	for c in range(n_cols):
		col0 = matrix[:,c]
		if abs(1-la.norm(col0)) > 1E-08: return False
		for inner_c in range(c+1,n_cols): 
			next_col = matrix[:,inner_c]
			if col0.dot(next_col) > 1E-08 : return False
	return True

def parallelComponent(b, v, eps = 1E-10):
	"""
	Receives two numpy vectors, and returns the 
	projection of b, over the span of v, that is, 
	its parallel component. b = alpha * v
	"""
	b = correctType(b)
	v = correctType(v)
	alpha = (1.0*b.dot(v)/v.dot(v)) if v.dot(v) >= eps else 0.0
	return alpha * v

def orthogonalComponent(b, vlist):
	"""
	Project b orthogonal to vlist.

	Input:
		- b: a Vec
		- vlist: a list of mutually orthogonal numpy vecs

	Output: the projection of b orthogonal to the Vecs in vlist
	"""

	b = correctType(b)
	#if not areOrthogonal(vlist): raise ValueError('Elements of vlist are not mutually orthogonal')
	for v in vlist:
		b = b - parallelComponent(b, v)
	return b

def projectionMatrix(v):
	"""
	Receives a numpy vector, and returns a projection matrix,
	as numpy array, such that given a vector x
	M * x is the projection of x along v.
	"""
	v = correctType(v)
	return np.outer(v,v)

def augOrthogonalComponent(b, vlist, eps = 1E-10):
	b = correctType(b)
	# if not areOrthogonal(vlist): raise ValueError('Elements of vlist are not mutually orthogonal')
	alphadict = {len(vlist):1}
	for i,v in enumerate(vlist):
		vdotv = 1.0*v.dot(v)
		sigma = (b.dot(v))/(vdotv) if vdotv > eps else 0.0
		alphadict[i] = sigma
		b = b - sigma*v
	return (b, alphadict)

def orthogonalize(vlist):
	'''
	Orthogonalizes vlist preserving order.
	The ith vector in the output list is the projection of vlist[i] orthogonal to
	the space spanned by all the previous vectors in the output list.

	Input: a list of Vecs

	Output: a list of mutually orthogonal Vecs spanning the same space as the input Vecs
	'''
	vstarlist = []
	for v in vlist:
		vstarlist.append(orthogonalComponent(v, vstarlist))
	return np.array(vstarlist)

def gramSchmidt(matrix):
	"""
	Receives a matrix, and returns another matrix with its columns being mutually orthogonal,
	and having lenght 1. In other words, returns an orthogonal matrix.
	"""
	n_cols = matrix.shape[1]
	ortho_mat = correctType(matrix)
	for k in range(n_cols):
		acol = ortho_mat[:, k]
		col = acol
		for j in range(k):
			prev_col = ortho_mat[:,j]
			col = col - prev_col.dot(col)*prev_col
		col = col/la.norm(col)
		ortho_mat[:,k] = col
	return ortho_mat

def QR(matrix):
	"""
	Receives a matrix, and returns two matrices:
		.Q -> Orthogonal Matrix (Matrix having mutually orthonormal columns)
		.R -> Upper Triangular Matrix
	such that, matrix = Q dot R
	"""
	n_cols = matrix.shape[1]
	matrix = correctType(matrix)
	Q = np.zeros(A.shape)
	R = np.zeros((A.shape[0], A.shape[0]))
	for k in range(n_cols):
		q = matrix[:, k] # Get a column of the given matrix
		for j in range(k):
			prev_col = Q[:,j] # Previous col of Q
			sigma_coeff = prev_col.dot(q) # Sigma Coefficient that Orthogonalizes col against the previous col of Q
			R[j,k] = sigma_coeff # Store it in R
			q = q - sigma_coeff*prev_col # Orthogonalize the column
		col_norm = la.norm(q) #Compute the norm of the orthogonalized column, in order to orthonormalize
		Q[:,k] = q/col_norm # Store in Q the orthonormalized column.
		R[k,k] = col_norm # Store in the diagonal of R the norm
	return (Q, R)


def testOrthogonality(matrix):
	print ("Q:")
	print(Q)

	print ("QT Q:")
	QtQ = np.dot(Q.T, Q)
	QtQ[np.abs(QtQ) < 1E-10] = 0
	print (QtQ)

def isZeroVector(list, tol = 1E-8):
	"""
	Tests if a given numpy array, or list is aproximateley zero.
	"""
	vec = correctType(list)
	return la.norm(vec) < tol

def OrthogonalBasis(rows_list):
	"""
	Returns a numpy array consisting of orthogonal vectors that form
	a basis for the given rows_list.
	That is, builds an orthogonal basis for the given vectors and returns
	it as a numpy array. 
	"""
	orthogonal_basis = orthogonalize(rows_list)
	return np.array([v_star for v_star in orthogonal_basis if not isZeroVector(v_star)])

def Rank(matrix):
	"""
	Returns the rank of the given matrix.
	"""
	return len(Basis(matrix)) # Compute a basis for the matrix, and return its length

def Basis(rows_list):
	"""
	Returns the vectors on rows_list that form a basis for that rows_list.
	"""
	orthogonal_basis = orthogonalize(rows_list)
	subset_basis = []
	for i in range(len(rows_list)):
		v = rows_list[i]
		v_star = orthogonal_basis[i]
		if not isZeroVector(v_star): subset_basis.append(v)
	return np.array(subset_basis)

def areLinearlyIndependent(rows_list):
	"""
	Tests if a given list of vectors are linearly independent between them.
	"""
	return Rank(rows_list) == len(rows_list)

def augOrthogonalize(vec_list):
	"""
	. input: a list [v1,...,vn] of vectors
	. output: the pair ([w1,...,wn],[u1,...,un]) of lists of vectors such that
		. [w1,...,wn] are mutually orthogonal vectors whose span equals Span {v1,...,vn} 
		. for i = 1,...,n:
			|         |   |         | |         |
			|v1 ... vn| = |w1 ... wn| |u1 ... un| 
			|         |   |         | |         |
	"""
	vstarlist = []
	sigma_vecs = []
	#D = set(range(len(vlist)))
	for v in vec_list:
		(vstar, sigmadict)= augOrthogonalComponent(v, vstarlist)
		vstarlist.append(vstar)
		sigma_vecs.append(sigmadict)
	return vstarlist, sigma_vecs






