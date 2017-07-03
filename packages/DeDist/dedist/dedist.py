''' Functions to estimate decoding distributions

Author: Sander Keemink, swkeemink@scimail.eu
'''

import numpy as np
from scipy.stats import mvn
from multiprocessing import Pool

def multi_fun(inputs):
    ''' Function to apply mvnun in paralell

    Parameters
    ----------
    inputs : list
        [0], array, lower bounds
        [1], array, upper bounds
        [2], array, means
        [3], array, covariance matrix

    Returns
    -------
    array
        probabilities for each stimulus

    '''
    low = inputs[0]
    upp = inputs[1]
    mean = inputs[2]
    cov = inputs[3]
    p,e = mvn.mvnun(low,upp,mean,cov)
    return p

def get_means(fun, theta, par, x, x_):
    ''' find means for multivariate normal describing error landscape

    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    full_return : binary
        if False, only returns decoding distribution. If true, also returns
        sampled errors, calculated mean, and covariance

    Returns
    -------
    array
        for each stimulus in x_, what the mean error will be
    '''
    # find real population response
    f = fun(x,theta,par)

    # find possible function values
    Fs = fun(x,x_.reshape(x_.shape+(1,)),par)

    # first, find the means
    means = np.sum( (f-Fs)**2,axis=1 )

    return means

def get_cov(fun,theta,par,sigma,x,x_):
    ''' find covariance matrix for multivariate normal describing error
    landscape

    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    sigma : float
        sigma^2 is the variance of the gaussian noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    full_return : binary
        if False, only returns decoding distribution. If true, also returns
        sampled errors, calculated mean, and covariance

    Returns
    -------
    array
        for each stimulus in x_, what the mean error will be
    '''
    # find dimensionality of multivar Gaussian
    ns = len(x_)

    # find real population response
    f = fun(x,theta,par)

    # find possible function values
    Fs = fun(x,x_.reshape(x_.shape+(1,)),par)

    # find the covariances
    cov = np.zeros((ns,ns))
    cov = 4*sigma**2*np.sum(Fs*Fs[:,None],axis=2)

    return cov

def sample_E(fun,theta,par,sigma,x,x_,n,full_return=False):
    ''' Samples n errors from a multivariate gaussian distribution.

    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    sigma : float
        sigma^2 is the variance of the gaussian noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    n : int
        number of realizations to sample
    full_return : binary
        if False, only returns decoding distribution. If true, also returns
        sampled errors, calculated mean, and covariance

    Returns
    -------
    array
        for each stimulus in x_, what how often this stimulus has the
        smallest error
    if full_return:
    array
        The sampled error profiles
    array
        The means for the multivariate normal
    array
        The covariance matrix for the multivariate normal
    '''
    # find dimensionality of multivar Gaussian
    ns = len(x_)

    # find real population response
    f = fun(x,theta,par)

    # find possible function values
    Fs = fun(x,x_.reshape(x_.shape+(1,)),par)

    # first, find the means
    means = np.sum( (f-Fs)**2,axis=1 )

    # now the covariances
    cov = np.zeros((ns,ns))
    cov = 4*sigma**2*np.sum(Fs*Fs[:,None],axis=2)

    # now do a set of realizations
    print 'Sampling from distribution'
    Errors = np.random.multivariate_normal(means,cov,size=n)
    sol_th = x_[Errors.argmin(axis=1)]
    print 'Done'
    # return values
    if full_return:
        return sol_th,Errors,means,cov
    else:
        return sol_th

def est_p(fun,theta,par,sigma,x,x_,full_return=False,lowmem=False,verbose=True):
    ''' For each stimulus in fun, estimates the probability that it gives the
    smallest error. It does this by find the multivariate normal for the error
    at each x_, with the error at each other x_' subtracted.

    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    sigma : float
        sigma^2 is the variance of the gaussian noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    full_return : binary,optional
        if False, only returns decoding distribution. If true, also returns
        the calculated means and covariance for each stimulus in x_.
        Default False
    lowmem : bool, optional
        Whether to use lower memory mode (useful if calculting big
        covariance matrices). Will not be able to use full_return!
        Default False
    verbose : bool, optional
        Whether to print progress or not

    Returns
    -------
    array
        for each stimulus in x_, the probability that this has the smallest
        error
    if full_return:
    array
        The full set of means. Dimensions as (len(x_)-1,len(x_)), such that
        means[:,i] describes the full set of means for the error differences
        with stimulus i.
    array
        The full set of covariances. Dimensions as (len(x_)-1,len(x_)-1,
        len(x_)). Thus covs[:,:,i] describes the relevant covariance matrix
        for stimulus i.


    '''
    # find dimensionality of multivar Gaussian
    ns = len(x_)

    # set integration bounds
    low = -np.ones(len(x_)-1)*1e50
    upp = np.zeros(len(x_)-1)

    # find real population response
    f = fun(x,theta,par)

    # make multidimensional version of x_ so less need for for loops
    # a + b.reshape(b.shape+(1,)) gives all possible combinations between
    # a and b
    x_mult = x_.reshape(x_.shape+(1,))

    # first, find all required function differences
    diffs = (fun(x,x_mult,par)[:,None]-fun(x,x_mult,par))
    diffs_sq = (fun(x,x_mult,par)[:,None]**2-fun(x,x_mult,par)**2)

    # then, find the means
    if verbose: print 'finding means'
    means = np.zeros((ns-1,ns)) # sum((f-f')**2)
    # loop over all to be generated means
    for i in range(ns):
        if verbose: print '\r'+str(i),
        # loop over all stimuli, except when i=j
        means[:i,i] = np.sum( diffs_sq[i,:i] - 2*f*diffs[i,:i],axis=1 )
        means[i:,i] = np.sum( diffs_sq[i,i+1:] - 2*f*diffs[i,i+1:],axis=1 )
    if verbose: print ''

    # If low memory, only calculate one covariance matrix at a time
    if lowmem:
        if verbose:
            print 'Low memory mode.'
            print 'Finding p[x] of ' + str(ns) + ':'
        for i in range(ns):
            if verbose: print '\r'+str(i),
            # find current covariance
            cov = np.zeros((ns-1,ns-1))
            cov[:i,:i] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                  *diffs[i,:i],axis=2)
            cov[:i,i:] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                      *diffs[i,i+1:],axis=2)
            cov[i:,:i] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                      *diffs[i,:i],axis=2)
            cov[i:,i:] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                  *diffs[i,i+1:],axis=2)

            # find p
            p[i],e = mvn.mvnun(low,upp,means[:,i],cov)

        return p

    # now for the tough one, the covariances
    if verbose:
        print 'finding covariances, ',
        print 'doing set x of ' + str(ns) + ':'
    # loop over coveriances to find
    covs = np.zeros((ns-1,ns-1,ns))
    for i in range(ns):
        if verbose: print '\r'+str(i),
        covs[:i,:i,i] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                  *diffs[i,:i],axis=2)
        covs[:i,i:,i] = 4*sigma**2*np.sum(diffs[i,:i][:,None]
                                  *diffs[i,i+1:],axis=2)
        covs[i:,:i,i] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                  *diffs[i,:i],axis=2)
        covs[i:,i:,i] = 4*sigma**2*np.sum(diffs[i,i+1:][:,None]
                                  *diffs[i,i+1:],axis=2)
    if verbose: print ''

    # calculate the cumulative distribution for each of the calculated covs
    if verbose: print 'Calculating cumulative distributions'

    # calculate probabilities
    pool = Pool(None) # to use less than max processes, change 'None' to number
    inputs = [[low,upp,means[:,i],covs[:,:,i]] for i in range(ns)]
    p = pool.map(multi_fun,inputs)
    pool.close()

    if full_return:
        return p, means, covs
    else:
        return p

def est_p_cor(fun,theta,par,cov,x,x_,full_return=False,lowmem=False,verbose=True):
    ''' For each stimulus in fun, estimates the probability that it gives the
    smallest error. It does this by find the multivariate normal for the error
    at each x_, with the error at each other x_' subtracted.

    This function does the same as est_p(), but for correlated noise. If you
    have uncorrelated noise, use est_p() as it is faster.

    Parameters
    ----------
    fun : function
        Function to be used. Assumed to be of form
        fun(x,x_,par)
        where x and x_ are described below, and par are the basic
        model parameters
    theta : array/float
        the real stimulus value
    par : array
        model parameters
    cov : array
        The covariance matrix for the noise
    x : array
        preferred values of neurons
    x_ : array
        actual values to be tried to decode
    full_return : binary,optional
        if False, only returns decoding distribution. If true, also returns
        the calculated means and covariance for each stimulus in x_.
        Default False
    lowmem : bool, optional
        Whether to use lower memory mode (useful if calculting big
        covariance matrices). Will not be able to use full_return!
        Default False
    verbose : bool, optional
        Whether to print progress or not

    Returns
    -------
    array
        for each stimulus in x_, the probability that this has the smallest
        error
    if full_return:
    array
        The full set of means. Dimensions as (len(x_)-1,len(x_)), such that
        means[:,i] describes the full set of means for the error differences
        with stimulus i.
    array
        The full set of covariances. Dimensions as (len(x_)-1,len(x_)-1,
        len(x_)). Thus covs[:,:,i] describes the relevant covariance matrix
        for stimulus i.


    '''
    # find dimensionality of multivar Gaussian
    ns = len(x_)

    # get inverse covariance
    cov_i = np.linalg.inv(cov)

    # set integration bounds
    low = -np.ones(len(x_)-1)*1e50
    upp = np.zeros(len(x_)-1)

    # find real population response
    f = fun(x,theta,par)

    # make multidimensional version of x_ so less need for for loops
    # a + b.reshape(b.shape+(1,)) gives all possible combinations between
    # a and b
    x_mult = x_.reshape(x_.shape+(1,))

    # first, find all required function differences
    diffs_true = f - fun(x,x_mult,par)
    Lmeans = np.array([np.dot(np.dot(diffs_true[a, :], cov_i),
                              diffs_true[a, :]) for a in range(ns)])
    diffs = (fun(x,x_mult,par)[:,None]-fun(x,x_mult,par))

    # then, find the means
    means = np.zeros((ns-1, ns))
    if verbose: print 'finding means'
    for m in range(ns):
        if verbose: print '\r'+str(m),
        means[:m, m] = Lmeans[m] - Lmeans[:m]
        means[m:, m] = Lmeans[m] - Lmeans[m+1:]
    if verbose: print ''

    # now for the covariances
    if verbose:
        print 'finding covariances, ',
        print 'doing set x of ' + str(ns) + ':'
    # loop over coveriances to find
    covs = np.zeros((ns-1, ns-1, ns))
    for m in range(ns):
        if verbose: print '\r'+str(m),
        for a in range(m):
            for b in range(m):
                covs[a, b, m] = np.dot(np.dot(diffs[m, a, :], cov_i),
                                       diffs[m, b, :])
            for b in range(m+1, ns):
                covs[a, b-1, m] = np.dot(np.dot(diffs[m, a, :], cov_i),
                                         diffs[m, b, :])
        for a in range(m+1, ns):
            for b in range(m):
                covs[a-1, b, m] = np.dot(np.dot(diffs[m, a, :], cov_i),
                                         diffs[m, b, :])
            for b in range(m+1, ns):
                covs[a-1, b-1, m] = np.dot(np.dot(diffs[m, a, :], cov_i),
                                           diffs[m, b, :])

    if verbose: print ''

    # calculate the cumulative distribution for each of the calculated covs
    if verbose: print 'Calculating cumulative distributions'

    # calculate probabilities
    pool = Pool(None)  # to use less than max processes, change 'None' to number
    inputs = [[low, upp, means[:, i], 4*covs[:, :, i]] for i in range(ns)]
    p = pool.map(multi_fun, inputs)
    pool.close()

    if full_return:
        return p, means, covs
    else:
        return p

def calc_crb(dfun,sigma,par,x,x_,db=0,b=0):
    ''' Estimates the optimally possible decoding distribution from the
    cramer-rao bound, assuming a neurons response is r_i = f_i + n_i,
    where n_i drawn from a normal dist with mean 0 and variance sigma.

    Only works for 1D systems.

    Parameters
    ----------
    dfun : function
        The derivative of the normal function. Should be of furm
        dfun(x,x_,par)
    sigma : float
        The variance in the noise
    par : array
        Parameters of the model
    x : array
        The prefered stimulus values
    x_ : array
        The stimulus values at which to evaluate the Fisher information
    db : array, optional
        The derivative of the bias, if any
    b : array, optional
        The bias, if any

    Returns
    -------
    array
        The cramer-rao bound at each stimulus value in x_

    '''
    # find population derivatives
    df = dfun(x,x_[:,None],par)

    # find the Fisher information at each stimulus value
    I = np.sum( df**2 , axis=1 )/sigma**2

    # find the CBR
    crb = (1+db)**2/I


    return crb
