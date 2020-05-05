#!/usr/bin/env python
# coding: utf-8

import numpy as np

#### dx/dt=f(t,x), x(t0)=x0

##################### Método de Euler ###################################################

def EulerM(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        x[i+1]=x[i]+h*f(t[i],x[i])
    return t,x

################## Método del Punto Medio ######################################################

def MidPt(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        
        k1=f(t[i],x[i])
        k2=f(t[i]+h/2,x[i]+h*1/2*k1)
        x[i+1]=x[i]+h*k2
        
    return t,x 

################################# Método de Heun #############################################

def Heun(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        k1=f(t[i],x[i])
        k2=f(t[i+1],x[i]+h*k1)
        x[i+1]=x[i]+h*(1/2*k1+1/2*k2)
        
    return t,x

###################### Método de Ralson ###############################################################

def Ralson(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        k1=f(t[i],x[i])
        k2=f(t[i]+2/3*h,x[i]+h*2/3*k1)
        x[i+1]=x[i]+h*(1/4*k1+3/4*k2)
        
    return t,x

########################################## Método de Runge-Kutta (RK4) ########################################

## f es la funcion
## x0 es la condicion incial
## T es el tiempo en el cual se va a solucionar el modelo
## n es el numero de puntos del metodo numerico

def RK4(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        
        k1=f(t[i],x[i])
        k2=f(t[i]+1/2*h,x[i]+h*1/2*k1)
        k3=f(t[i]+1/2*h,x[i]+h*1/2*k2)
        k4=f(t[i+1]    ,x[i]+h*k3)
        
        x[i+1]=x[i]+h*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        
    return t,x

########################################## Método de Runge-Kutta, Regla 3/8 ####################################

def RK38(f,x0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        k1=f(t[i],x[i])
        k2=f(t[i]+1/3*h,x[i]+h*k1/3)
        k3=f(t[i]+2/3*h,x[i]+h*(-1/3*k1+k2))
        k4=f(t[i+1]    ,x[i]+h*(k1-k2+k3))
        
        x[i+1]=x[i]+h*(1/8*k1+3/8*k2+3/8*k3+1/8*k4)
    return t,x

########################################## Método de Runge-Kutta-Fehlberg ####################################

def RKF(f,x0,T,n,order=5):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        k1=f(t[i],x[i])
        k2=f(t[i]+1/4*h  ,x[i]+h*k1/4)
        k3=f(t[i]+3/8*h  ,x[i]+h*(3/32*k1+9/32*k2))
        k4=f(t[i]+12/13*h,x[i]+h*(1932/2197*k1-7200/2197*k2+7296/2197*k3))
        k5=f(t[i]+h      ,x[i]+h*(439/216*k1-8*k2+3680/513*k3-845/4104*k4))
        k6=f(t[i]+h/2    ,x[i]+h*(-8/27*k1+2*k2-3544/2565*k3+1859/4104*k4-11/40*k5))
        
        if order==5:
            x[i+1]=x[i]+h*(16/135*k1+6656/12825*k3+28561/56430*k4-9/50*k5+2/55*k6)
        else:
            x[i+1]=x[i]+h*(25/216*k1+1408/2565*k3+2197/4104*k4-1/5*k5)
    return t,x

########################################## Método de Dormand-Prince ####################################

def DP(f,x0,T,n,order=5):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    
    for i in range(n-1):
        k1=f(t[i]       ,x[i])
        k2=f(t[i]+1/5*h ,x[i]+h*k1/5)
        k3=f(t[i]+3/10*h,x[i]+h*(3/40*k1+9/40*k2))
        k4=f(t[i]+4/5*h ,x[i]+h*(44/45*k1-56/15*k2+32/9*k3))
        k5=f(t[i]+8/9*h ,x[i]+h*(19372/6561*k1-25360/2187*k2+64448/6561*k3-212/729*k4))
        k6=f(t[i]+h     ,x[i]+h*(9017/3168*k1-355/33*k2+46732/5247*k3+49/176*k4-5103/18656*k5))
        k7=f(t[i]+h     ,x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6))
        if order==5:
            x[i+1]=x[i]+h*(35/384*k1+500/1113*k3+125/192*k4-2187/6784*k5+11/84*k6)
        else:
            x[i+1]=x[i]+h*(5179/57600*k1+7571/16695*k3+393/640*k4-92097/339200*k5+187/2100*k6+1/40*k7)
    return t,x

################################################# Funciones de Error#####################################################

def Error_Abs(aprox,real):
    err=abs(aprox-real)
    return err

def Error_Rela(aprox,real):
    err=abs(aprox-real)/abs(real)
    return err


def MSE(aprox,real):
    mse=np.square(aprox-real).mean()
    return mse

##################################################################################################################