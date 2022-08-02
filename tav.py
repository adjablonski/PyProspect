import numpy as np

def calctav(alpha,nr):
    rd = np.pi/180
    n2 = nr**2
    npp = n2+1
    nm = n2-1
    a  = (nr+1)*(nr+1)/2
    k  = -(n2-1)*(n2-1)/4
    sa = np.sin(alpha*rd)
    
    b1 = np.int(alpha!=90)*np.sqrt((sa**2-npp/2)*(sa**2-npp/2)+k)
    b1 = np.nan_to_num(b1)
    b2  = sa**2-npp/2
    b   = b1-b2
    b3  = b**3
    a3  = a**3
    ts  = (k**2/(6*b3)+k/b-b/2)-(k**2/(6*a3)+k/a-a/2)
    
    tp1 = -2*n2*(b-a)/(npp**2)
    tp2 = -2*n2*npp*np.log(b/a)/(nm**2)
    tp3 = n2*(1/b-1/a)/2
    tp4 = 16*n2**2*(n2**2+1)*np.log((2*npp*b-nm**2)/(2*npp*a-nm**2))/(npp**3*nm**2)
    tp5 = 16*n2**3*(1/(2*npp*b-nm**2)-1/(2*npp*a-nm**2))/(npp**3)
    tp  = tp1+tp2+tp3+tp4+tp5
    tav = (ts+tp)/(2*sa**2)
    
    return tav