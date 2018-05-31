#!/usr/bin/python

#  Fitting complex backward and forward models
#

import numpy as np
import scipy.linalg as linalg
import scipy.stats as stats
import scipy.signal as signal
from scipy import fftpack

def ridge_fit_SVD(XtX,XtY,lambdas):
    # Fast implementation of regularized least square fitting of the complex model's coefficients.
    # The idea is to perform a computationally heavy operation of computing eigenvalues and eigenvectors only once and then simply compute the coefficients
    # for variety of regularization parameters (multiple fast operations).
    #
    # Input:
    # XtX - covariance matrix of design matrix X. Shape: [N x N]
    # XtY - covariance matrix of design matrix X and vector Y. Shape: [N x M]
    # lambdas - list of regularization parameters to be considered of length R
    #
    # Output:
    # coeff - array of models coefficients for each regularization parameter. Shape: [R x N x M]
    #
    # Note:
    # For forward model, the output matrix is rectangular [timelags x channels] but for backward model it is a row vector of length [timelags*channels].
    # To obtain the rectangular shape, each of those row vectors need to be reshaped accordingly. 
    #

    # Compute eigenvaluesa and eigenvectors of covariance matrix XtX
    S,V = linalg.eigh(XtX, overwrite_a=True, turbo=True)
    
    # Sort the eigenvalues
    s_ind = np.argsort(S)[::-1]
    S = S[s_ind]
    V = V[:,s_ind]
    
    # Pick eigenvalues close to zero, remove them and corresponding eigenvectors and compute the average
    tol = np.finfo(float).eps
    r = sum(S > tol)
    S = S[0:r]
    V = V[:,0:r]
    nl = np.mean(S)
    
    # Compute z
    z = np.dot(V.T,XtY)
    
    # Initialize empty list to store coefficient for different regularization parameters
    coeff = []
        
    # Compute coefficients for different regularization parameters
    for l in lambdas:
        coeff.append(np.dot(V ,(z/(S[:,np.newaxis] + nl*l))))
    
    return np.array(coeff)

def fast_hilbert(X, axis = 0):
    # Fast implementation of Hilbert transform. The trick is to find the next fast length of vector for fourier transform (fftpack.helper.next_fast_len(...)).
    # Next the matrix of zeros of the next fast length is preallocated and filled with the values from the original matrix.
    #
    # Input:
    # X - input matrix
    # axis - axis along which the hilbert transform should be computed
    #
    # Output:
    # X - analytic signal of matrix X (the same shape, but dtype changes to np.complex)
    # 
    fast_shape = np.array([fftpack.helper.next_fast_len(X.shape[0]), X.shape[1]])
    X_padded = np.zeros(fast_shape)
    X_padded[:X.shape[0], :] = X
    X = signal.hilbert(X_padded, axis=axis)[:X.shape[0], :]
    return X

def train(eeg, Y, tlag, cmplx=True, forward=False, lambdas=[0]):
    # Custom traning function. Takes training eeg, fundamental waveform (Y) datasets zscores and fits the backward or forward model.
    #
    # Input:
    # eeg - eeg data. Numpy array with shape [T x N], where T - number of samples, N - number of recording channels.
    # Y - speech signal features (envelope, fundamental waveform etc.). Numpy array with shape [T x 1], where T - number of samples (the same as in EEG).
    # tlag - timelag range to consider in samples. Two element list. [-100, 400] means one does want to consider timelags of -100 ms and 400 ms for 1kHz sampling rate.
    # complex - boolean. True if complex model is considered and coeff will have complex values. Otherwise False and coeff will be real-only.
    # forward model - boolean. True if forward model shall be built. False if backward.
    # lambdas - range of regularization parameters to be considered. If None lambdas = [0] means no regularization.
    # 
    # Output:
    # coeff - list of model coefficients for each considered regularization parameter.
    
    # eeg and Y need to have the same number of samples.
    assert (eeg.shape[0] == Y.shape[0])
    assert (Y.shape[1] == 1)
    assert (len(tlag) == 2)
    
    lag_width = tlag[1] - tlag[0]
    
    # If forward model is to be considered swap the names of eeg and Y variables, as now the goal is to map FW to EEG
    if forward == True:
        eeg, Y = Y, eeg
        tlag = np.array(tlag)[::-1]
    else:
        tlag = np.array(tlag)*-10
    
    # Align Y, so that it is misaligned with eeg by tlag_width samples
    Y = Y[tlag[0]:tlag[1], :]
    
    # Apply hilbert transform to EEG data (if backward model) or fundamental waveform (if forward model)
    if cmplx == True:
        eeg = fast_hilbert(eeg, axis=0)
    
    # Preallocate memory for the design matrix X
    X = np.zeros((Y.shape[0], int(lag_width*eeg.shape[1])), dtype=eeg.dtype)
    
    # Fill in the design matrix X
    for t in range(X.shape[0]):
        X[t,:] = eeg[t:(t + lag_width),:].reshape(lag_width*eeg.shape[1])
    
    # If complex concatenate the real and imaginary parts columne-wise
    if cmplx == True:
        X = np.hstack((X.real, X.imag))
    
    # Standardize X and Y matrices
    X = stats.zscore(X, axis=0)
    Y = stats.zscore(Y, axis=0)
    
    # Compute covariance matrices XtX and XtY
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)
    
    # Fit the model using chosen set of regularization parameters
    coeff = ridge_fit_SVD(XtX, XtY, lambdas)
    
    # Flip the coefficients of the forward model to reflect the timelags w.r.t. to fundamental waveform 
    # Note: In order to reconstruct EEG from fundamental waveform with this model it needs to be flipped column-wise once again!!!
    if forward == True:
        coeff = coeff[:,::-1,:]
    
    return coeff