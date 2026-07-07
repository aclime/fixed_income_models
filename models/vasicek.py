import numpy as np

#Initial params
#r=0.03      #Short-term interest rate
#theta=0.05  #Long-term interest rate
#kappa=0.50	#Mean-reversion
#sigma =0.02	#Volatility
#T=30        #maturity upper bound
#T_line=np.arange(1,T+1)

def common_terms(kappa,T_line,theta,sigma):
    """compute A(t) and B(t)"""
    B = (1-(1-kappa)**T_line)/kappa
    B_minus1=np.insert(B[:-1], 0, 0.0)
    A_terms = np.log( 0.5 * np.exp(-(theta * kappa + sigma) * B_minus1) + 
                    0.5 * np.exp(-(theta * kappa - sigma) * B_minus1) )
    A_terms[0]=0.0
    A=np.cumsum(A_terms)
    return B,A

def continuous_yc(A,B,r,T_line):
    price_cont=np.exp(A-B*r)
    yield_cont=-np.log(price_cont)/T_line
    return yield_cont

def discrete_yc(A,B,r,sigma,kappa,theta,T_line):
    ru=r+kappa*(theta-r)+sigma
    rd=r+kappa*(theta-r)-sigma
    pu=np.exp(A[:-1]-B[:-1]*ru)
    pd=np.exp(A[:-1]-B[:-1]*rd)
    price_discr=np.exp(-r)*(0.5*pu+0.5*pd)
    yield_discr=-np.log(price_discr)/T_line[1:]
    return yield_discr

def ru(r,kappa,theta,sigma):
    return r+kappa*(theta-r)+sigma

def rd(r,kappa,theta,sigma):
    return r+kappa*(theta-r)-sigma
