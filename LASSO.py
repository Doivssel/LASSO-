import time
import numpy as np
import numpy.random as rdm
import pandas as pd
import matplotlib.pyplot as plt


#Standardise data if necesseray
def normalize(x):
    sd=(x-np.mean(x))/(np.std(x,axis=0))
    return(sd)

#MSE if necessary
def MSE(Y,Y_esti):
    N=np.size(Y)
    Squared=[(Y[i]-Y_esti[i])**2 for i in range(N)]
    return(np.sum(Squared)/N)

#Objective function OLS f(B)=(y-XB)^T(y-XB)/2N
def obj_OLS(y,X,B):
    N=np.size(y)
    err=y-np.dot(X,B)
    return(np.dot(err.T,err)/(2*N))

#Objective function LASSO l(B)=f(B)+lambda*Norm_1(B)
def obj_LASSO(y,X,B,lam):
    return(obj_OLS(y,X,B)+lam*np.sum(np.abs(B)))

#Gradient function OLS Gf(B)=X^T(XB-y)/N
def grad_OLS(y,X,B):
    N=np.size(y)
    return(np.dot(X.T,np.dot(X,B)-y)/N)

#Condition for Armijo to change
def cond_armijo(y,X,Bk,step):
    return(obj_OLS(y,X,Bk)+np.dot(grad_OLS(y,X,Bk).T,step*grad_OLS(y,X,Bk)))


#Armijo rule for unconstrained gradient
def armijo(y,X,Bk,a=0.5,s=0.8):
    step=0.01
    while(True):
        Bkp=Bk-step*grad_OLS(y,X,Bk)
        if(obj_OLS(y,X,Bkp)<=obj_OLS(y,X,Bk)+a*step*np.dot(grad_OLS(y,X,Bk).T,step*grad_OLS(y,X,Bk))):
            return(step)
        else:
            step=s*step


#Unconstrained gradient descent
def gradient_descent(y,X,beta,beta_t,iters):
    beta_iter=[]
    for i in range(iters):
        grad=grad_OLS(y,X,beta)
        s=armijo(y,X,beta)
        beta=beta-s*grad
        beta_iter.append(np.sum((beta-beta_t)**2))
    return(beta,beta_iter)

#Soft tresholding operator
def soft_tresh(x,tau):
    return(np.sign(x)*np.maximum(np.abs(x)-tau,np.zeros(len(x))))


#Proximate gradient descent Armijo
def proximate_descent_A(y,X,beta,iters,lamb):
    evol=[]
    err=[]
    for i in range(iters):
        grad=grad_OLS(y,X,beta)
        s=armijo(y,X,beta)
        beta=soft_tresh(beta-s*grad,s*lamb)
        evol.append(beta)
        err.append(obj_LASSO(y,X,beta,lamb))
    return(beta,evol,err)

#Fast proximate gradient descent Armijo
def fast_proximate_descent_A(y,X,beta,iters,lamb):
    evol=[]
    err=[]
    theta=np.copy(beta)
    for i in range(iters):
        grad=grad_OLS(y,X,beta)
        s=armijo(y,X,beta)
        betam=beta
        beta=soft_tresh(theta-s*grad,s*lamb)
        theta=beta+(i/(i+3))*(beta-betam)
        evol.append(beta)
        err.append(obj_LASSO(y,X,beta,lamb))
    return(beta,evol,err)

#Constant continuosly lipshitz, we take the stepsize s in (0,1/L) on prend directement 1/2L
def constant(X):
    N=np.shape(X)[0]
    eig=np.linalg.eig(np.dot(X.T,X)/N)[0]
    return(1/(20*max(eig)))

#Proximate gradient descent 1/2L
def proximate_descent_C(y,X,beta,iters,lamb):
    evol=[]
    err=[]
    s=constant(X)
    M=[]
    for i in range(iters):
        grad=grad_OLS(y,X,beta)
        beta=soft_tresh(beta-s*grad,s*lamb)
        evol.append(beta)
        err.append(obj_LASSO(y,X,beta,lamb))
        M.append(MSE(y,np.dot(X,beta)))
    return(beta,evol,err,M)


#Fast proximate gradient descent 1/2L
def fast_proximate_descent_C(y,X,beta,iters,lamb):
    evol=[]
    err=[]
    theta=np.copy(beta)
    s=constant(X)
    for i in range(iters):
        grad=grad_OLS(y,X,beta)
        betam=beta
        beta=soft_tresh(theta-s*grad,s*lamb)
        theta=beta+(i/(i+3))*(beta-betam)
        evol.append(beta)
        err.append(obj_LASSO(y,X,beta,lamb))
    return(beta,evol,err)
