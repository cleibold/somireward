import numpy as N
from numpy import copy, sort, amax, arange, exp, sqrt, abs, floor, searchsorted

import itertools
import math
import random
import js_data_tools as tool
import scipy.stats as st
import matplotlib.pyplot as pl




def GammaInc_Q( a, x):
    a1 = a-1
    a2 = a-2
    def f0( t ):
        return t**a1*math.exp(-t)
 
    def df0(t):
        return (a1-t)*t**a2*math.exp(-t)
 
    y = a1
    while f0(y)*(x-y) >2.0e-8 and y < x: y += .3
    if y > x: y = x
 
    h = 3.0e-4
    n = int(y/h)
    h = y/n
    hh = 0.5*h
    gamax = h * sum( f0(t)+hh*df0(t) for t in ( h*j for j in xrange(n-1, -1, -1)))
 
    return gamax/gamma_spounge(a)
 
c = None
def gamma_spounge( z):
    global c
    a = 12
 
    if c is None:
       k1_factrl = 1.0
       c = []
       c.append(math.sqrt(2.0*math.pi))
       for k in range(1,a):
          c.append( math.exp(a-k) * (a-k)**(k-0.5) / k1_factrl )
          k1_factrl *= -k
 
    accm = c[0]
    for k in range(1,a):
        accm += c[k] / (z+k)
    accm *= math.exp( -(z+a)) * (z+a)**(z+0.5)
    return accm/z;
 
def chi2UniformDistance( dataSet ):
    expected = sum(dataSet)*1.0/len(dataSet)
    cntrd = (d-expected for d in dataSet)
    return sum(x*x for x in cntrd)/expected
 
def chi2Probability(dof, distance):
    return 1.0 - GammaInc_Q( 0.5*dof, 0.5*distance)
 
def chi2IsUniform(dataSet, significance):
    dof = len(dataSet)-1
    dist = chi2UniformDistance(dataSet)
    return chi2Probability( dof, dist ) > significance
 

def chisquare(data):
    dof=len(data)-1
    distance =chi2UniformDistance(data)
    print "dof: %d distance: %.4f" % (dof, distance),
    prob = chi2Probability( dof, distance)
    print "probability: %.4f"%prob,
    print "uniform? ", "Yes"if chi2IsUniform(data,0.05) else "No"
 

def rayleigh_test(angles):
    '''
        Performs Rayleigh Test for non-uniformity of circular data.
        Compares against Null hypothesis of uniform distribution around circle
        Assume one mode and data sampled from Von Mises.
        Use other tests for different assumptions.
        Uses implementation close to CircStats Matlab toolbox, maths from [Biostatistical Analysis, Zar].
    '''

    no = len(angles)

    # Compute Rayleigh's R
    R = no*angle_population_R(angles)

    # Compute Rayleight's z
    z = R**2. / no

    # Compute pvalue (Zar, Eq 27.4)
    pvalue = N.exp(N.sqrt(1. + 4*no + 4*(no**2. - R**2)) - 1. - 2.*no)

    return pvalue

def angle_population_vector(angles):
    '''
        Compute the complex population mean vector from a set of angles
        Mean over Axis 0
    '''

    return N.mean(N.exp(1j*angles), axis=0)


def angle_population_vector_weighted(angles, weights):
    '''
        Compute the weighted mean of the population vector of a set of angles
    '''
    return N.nansum(N.exp(1j*angles)*weights, axis=0)/N.nansum(weights)

def angle_population_R(angles=None, angle_population_vec=None, weights=None):
    '''
        Compute R, the length of the angle population complex vector.
        Used to compute Standard deviation and diverse tests.
        If weights is provided, computes a weighted population mean vector instead.
    '''

    if angle_population_vec is None:
        if weights is None:
            angle_population_vec = angle_population_vector(angles)
        else:
            angle_population_vec = angle_population_vector_weighted(angles, weights)

    return N.abs(angle_population_vec)
def kuiper_two(data1, data2):
    """Compute the Kuiper statistic to compare two samples.
    Parameters
    ----------
    data1 : array-like
        The first set of data values.
    data2 : array-like
        The second set of data values.
    
    Returns
    -------
    D : float
        The raw test statistic.
    fpp : float
        The probability of obtaining two samples this different from
        the same distribution.
    Notes
    -----
    Warning: the fpp is quite approximate, especially for small samples.
    """
    data1, data2 = sort(data1), sort(data2)

    if len(data2)<len(data1):
        data1, data2 = data2, data1

    cdfv1 = searchsorted(data2, data1)/float(len(data2)) # this could be more efficient
    cdfv2 = searchsorted(data1, data2)/float(len(data1)) # this could be more efficient
    D = (amax(cdfv1-arange(len(data1))/float(len(data1))) + 
            amax(cdfv2-arange(len(data2))/float(len(data2))))

    Ne = len(data1)*len(data2)/float(len(data1)+len(data2))
    return D

def moore_test(ang1,ang2):
    ''' Paired non-parametric test whether angles of two samples are drawn from the same population.
        EXPERIMENTAL AND NOT VERIFIED YET    
    '''
    
    # Compute pairs of rectangular coordinates.
    x,y=[],[]
    for n in range(len(ang1)):
        x.append(N.cos(ang1[n])-N.cos(ang2[n]))
        y.append(N.sin(ang1[n])-N.sin(ang2[n]))
        
    # Compute vecotr magnitudes ond directions.
    m,d_c,d_s=[],[],[]
    
    for n in range(len(x)):
        m_local=(N.sqrt(x[n]**2+y[n]**2))
        m.append(m_local)
        d_c.append(x[n]/m_local)
        d_s.append(y[n]/m_local)
    
    i=N.asarray(m)
    indices=i.argsort()
    indices=indices+1
    #return indices,m,d_c,d_s
    rc,rs=[],[]
    for n in indices:
        rc.append(n*d_c[n-1])
        rs.append(n*d_s[n-1])
    rc=N.sum(rc)/len(ang1)
    rs=N.sum(rs)/len(ang1)
    R=N.sqrt((rc**2+rs**2)/len(ang1))
    
    return R

    

    
    
def two_sample_angle_permutation_test(data1,data2,iterations=10000,plotData=False):
    ''' A permutation for test for the means of phase angles of two dependent groups (i.e. phase of spike-oscillation coupling)
        following loosely the stragegy of Hartwich, Pollak & Klausberger, J Neurosci 2009
    
        the circular mean of each group is determined and compared to differences of
        randomly permuted values.
    
    '''
    data1=data1[~N.isnan(data1)]
    data2=data2[~N.isnan(data2)]

   
    total=[]
    total.extend(data1)
    total.extend(data2)

    
    
    c1=st.circmean(data1,low=-N.pi,high=N.pi)
    c2=st.circmean(data2,low=-N.pi,high=N.pi)
    
    c_diff=N.abs(N.pi-N.abs(N.abs(c2-c1)-N.pi))
    
    
    # Random permutations between groups.    
    c_diff_list=[]
    
    for n in range(iterations):
        rand=N.random.permutation(total)
        
        rand1=rand[:len(data1)]
        rand2=rand[len(data1):]#-len(data1)
            
        c1_rand=st.circmean(rand1,low=-N.pi,high=N.pi)
        c2_rand=st.circmean(rand2,low=-N.pi,high=N.pi)
        #mrl1_rand=tool.mrl(c1_random) 
        #mrl2_rand=tool.mrl(c2_random)
        
        c_diff_list.append(N.abs(N.pi-N.abs(N.abs(c2_rand-c1_rand)-N.pi)))
        #mrl_diff_list.append(mrl2_rand-mrl1_rand)
    
    # Calculation of p-value according to https://onlinecourses.science.psu.edu/stat464/node/35.    
    p1=sum(i>c_diff for i in c_diff_list)/float(iterations)
    
    print p1
   
    if plotData==True:
        pl.figure()
        _=pl.hist(c_diff_list,bins=50)
        pl.axvline(x=c_diff)
    #return c_diff_list,c_diff
    