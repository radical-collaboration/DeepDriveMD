"""
TD2
This module contains only one function, TD2, which performs temporal decorrelation of 2nd order.
"""

import numpy as np
from sys import stdout
from numpy import argsort, cov, diag, dot, sqrt, std, float64, mean,shape, ndarray, matrix, absolute, arange, log, tanh, pi,abs
from numpy.linalg import inv, eigh, eig
from numpy import sqrt
import warnings


def TD2(Y, m=None, U=None, lag=None, verbose=True):
    
    """
    TD2 - Temporal Decorrelation of 2nd order of real signals 
    
    Parameters:
    
        Y -- an m x T spatially whitened matrix (m: dimensioanlity of subspace, T snapshots). May be a numpy 
                array or matrix where 
                
                                m: dimensionality of the subspace we are interested in. Defaults to None, in
                which case m=n.
                T: number of snapshots of MD trajectory                 
      
        U -- whitening matrix obtained after doing the PCA analysis on m components
              of real data
        
        lag -- lag time in the form of an integer denoting the time steps
        
        verbose -- print info on progress. Default is True.
    
    Returns:
    
        V -- An n x m matrix V (NumPy matrix type) is a separating matrix such that
        V = Btd2 x U (U is obtained from SD2 of data matrix and Btd2 is obtained from 
        time-delayed covariance of matrix Y)
        
        Z -- Z = B2td2 * Y is spatially whitened and temporally decorrelated
        (2nd order) source extracted from the m x T spatially whitened matrix Y.         
         
        Dstd2: has eigen values sorted by increasing variance
                PCstd2: holds the index for m most significant principal components by decreasing variance
        
                R = Dstd2[PCstd2]
                R    -- Eigen values of the time-delayed covariance matrix of Y 
        Btd2 -- Eigen vectors of the time-delayed covariance matrix of Y
        
        Quick notes (more at the end of this file):
        
        o TD2 is performed on a spatially decorrelated real valued signal.
          This code makes use of AMUSE algorithm which can be found using the link:
          http://docs.markovmodel.org/lecture_tica.html
          
        o Tic is the time lagged covariance matrix computed with time lag = lag
          To ensure symmetricity of the covariance matrix a mathematical computation is 
          made: Tic = 0.5*[Tic + Tic.T]
          
        o Eigen value decomposition of this time delayed symmetrized covariance matrix 
          is performed to obtain eigen vector matrix  
        """
    # GB: we do some checking of the input arguments and copy data to new
    # variables to avoid messing with the original input. We also require double
    # precision (float64) and a numpy matrix type for Y.
    warnings.simplefilter("ignore", np.ComplexWarning)
    assert isinstance(Y, ndarray),\
        "Y (input data matrix) is of the wrong type (%s)" % type(Y)
    origtype = Y.dtype # remember to return matrix B of the same type
    Y = matrix(Y.astype(float64))
    assert Y.ndim == 2, "Y has %d dimensions, should be 2" % Y.ndim
    assert (verbose == True) or (verbose == False), \
        "verbose parameter should be either True or False"
    
    [n,T] = Y.shape # GB: n is number of input signals, T is number of samples
    
    if m==None:
        m=n     # Number of sources defaults to # of sensors
    assert m<=n,\
        "TD2 -> Do not ask more sources (%d) than sensors (%d )here!!!" % (m,n)
    Y = Y.T
    Y1 = Y[0: T-lag:]
    Y2 = Y[lag:T:]
    if verbose:
        print >> stdout, "2nd order Temporal Decorrelation -> Looking for %d sources" % m
        print >> stdout, "2nd order Temporal Decorrelation -> Removing the mean value"
    
    #compute time-delayed covariance matrix
    Tic = (Y1.T * Y2)/float(T-lag)
    Tic = ((Tic + Tic.T))/2
    
    if verbose:
        print >> stdout, "2nd order Temporal Decorrelation -> Whitening the data"

    [Dtd2,Utd2] = eig((Tic)) # An eigen basis for the sample covariance matrix
    ktd2 = abs(Dtd2).argsort()
    Dstd2 = Dtd2[ktd2] # sorting by increasing variance
    PCstd2 = arange(n-1, n-m-1, -1)      
    Btd2 = Utd2[:,ktd2[PCstd2]].T 
    R = Dstd2[PCstd2]
    V = dot(Btd2,U)
    # --- Scaling  ------------------------------------------------------
    scales = sqrt(abs(R)) # The scales of the principal components .
    B2td2 = dot(diag(1./scales) , Btd2)  
    
    Z = dot( B2td2 , Y.T ) # B2td2 is a whitening matrix in temporal domain and Z is 
    # spatially whitened and temporally decorrelated
    return (Z,R,Btd2,V)
    
    """
    NOTE: At this stage, Z is obtained by projecting the trajectory obtained from Y
    onto the dominant eigen vectors
    Z is now a matrix of spatially whitened and temporally uncorrelated components in the 2nd order
    """
