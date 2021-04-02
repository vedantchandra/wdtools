# 3-D correction from Tremblay et al. 2013

import numpy as np

def teff3d(teff, logg):

    A = np.zeros(8)
    A[0]=1.0947335e-03
    A[1]=-1.8716231e-01
    A[2]=1.9350009e-02
    A[3]=6.4821613e-01
    A[4]=-2.2863187e-01
    A[5]=5.8699232e-01
    A[6]=-1.0729871e-01
    A[7]=1.1009070e-01

    Teff0=(teff-10000.0)/1000.00
    logg0=(logg-8.00000)/1.00000
    Shift=A[0]+(A[1]+A[6]*Teff0+A[7]*logg0)*np.exp(-(A[2]+A[4]*Teff0+A[5]*logg0)**2*((Teff0-A[3])**2))
    return Shift*1000.00+0.00000

def logg3d(teff, logg):
    A = np.zeros(13)
    A[1]=7.5209868E-04
    A[2]=-9.2086619E-01
    A[3]=3.1253746E-01
    A[4]=-1.0348176E+01
    A[5]=6.5854716E-01
    A[6]=4.2849862E-01
    A[7]=-8.8982873E-02
    A[8]=1.0199718E+01
    A[9]=4.9277883E-02
    A[10]=-8.6543477E-01
    A[11]=3.6232756E-03
    A[12]=-5.8729354E-02
    Teff0=(teff-10000.0)/1000.00
    logg0=(logg-8.00000)/1.00000
    Shift=(A[1]+A[5]*np.exp(-A[6]*((Teff0-A[7])**2)))+A[2]*\
        np.exp(-A[3]*((Teff0-(A[4]+A[8]*np.exp(-(A[9]+A[11]*\
        Teff0+A[12]*logg0)**2*((Teff0-A[10])**2))))**2))
    
    return Shift

def corr3d(teff, logg):
    dteff = teff3d(teff, logg)
    dlogg = logg3d(teff, logg)
    return teff + dteff, logg + dlogg