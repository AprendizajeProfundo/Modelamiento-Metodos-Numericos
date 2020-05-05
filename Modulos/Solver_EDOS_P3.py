#!/usr/bin/env python
# coding: utf-8


import numpy as np

#### dx/dt=f1(t,x,y,z), x(t0)=x0
#### dy/dt=f2(t,x,y,z), y(t0)=y0
#### dz/dt=f3(t,x,y,z), z(t0)=z0

##################### Método de Euler ###################################################

def EulerM(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        x[i+1]=x[i]+h*f1(t[i],x[i],y[i],z[i])
        y[i+1]=y[i]+h*f2(t[i],x[i],y[i],z[i])
        z[i+1]=z[i]+h*f3(t[i],x[i],y[i],z[i])
        
    return t,x,y,z

################## Método del Punto Medio #############################################

def MidPt(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        
        k1=f1(t[i],x[i],y[i],z[i])
        k2=f1(t[i]+h/2,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        x[i+1]=x[i]+h*k2
        
        k1=f2(t[i],x[i],y[i],z[i])
        k2=f2(t[i]+h/2,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        y[i+1]=y[i]+h*k2
        
        k1=f3(t[i],x[i],y[i],z[i])
        k2=f3(t[i]+h/2,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        z[i+1]=z[i]+h*k2
        
    return t,x,y,z

################################# Método de Heun #############################################

def Heun(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        
        k1=f1(t[i],x[i],y[i],z[i])
        k2=f1(t[i+1],x[i]+h*k1,y[i]+h*k1,z[i]+h*k1)
        x[i+1]=x[i]+h*(1/2*k1+1/2*k2)
        
        k1=f2(t[i],x[i],y[i],z[i])
        k2=f2(t[i+1],x[i]+h*k1,y[i]+h*k1,z[i]+h*k1)
        y[i+1]=y[i]+h*(1/2*k1+1/2*k2)
        
        k1=f3(t[i],x[i],y[i],z[i])
        k2=f3(t[i+1],x[i]+h*k1,y[i]+h*k1,z[i]+h*k1)
        z[i+1]=z[i]+h*(1/2*k1+1/2*k2)
        
    return t,x,y,z

########################### Método de Ralson ###############################################################

def Ralson(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        
        k1=f1(t[i],x[i],y[i],z[i])
        k2=f1(t[i]+2/3*h,x[i]+h*2/3*k1,y[i]+h*2/3*k1,z[i]+h*2/3*k1)
        x[i+1]=x[i]+h*(1/4*k1+3/4*k2)
        
        k1=f2(t[i],x[i],y[i])
        k2=f2(t[i]+2/3*h,x[i]+h*2/3*k1,y[i]+h*2/3*k1,z[i]+h*2/3*k1)
        y[i+1]=y[i]+h*(1/4*k1+3/4*k2)
        
        k1=f3(t[i],x[i],y[i],z[i])
        k2=f3(t[i]+2/3*h,x[i]+h*2/3*k1,y[i]+h*2/3*k1,z[i]+h*2/3*k1)
        z[i+1]=z[i]+h*(1/4*k1+3/4*k2)
        
    return t,x,y,z

########################################## Método de Runge-Kutta (RK4) ########################################

def RK4(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        
        k1=f1(t[i],x[i],y[i],z[i])
        k2=f1(t[i]+1/2*h,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        k3=f1(t[i]+1/2*h,x[i]+h*1/2*k2,y[i]+h*1/2*k2,z[i]+h*1/2*k2)
        k4=f1(t[i+1]    ,x[i]+h*k3    ,y[i]+h*k3,    z[i]+h*k3)
        
        x[i+1]=x[i]+h*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        
        k1=f2(t[i],x[i],y[i],z[i])
        k2=f2(t[i]+1/2*h,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        k3=f2(t[i]+1/2*h,x[i]+h*1/2*k2,y[i]+h*1/2*k2,z[i]+h*1/2*k2)
        k4=f2(t[i+1]    ,x[i]+h*k3    ,y[i]+h*k3,    z[i]+h*k3)
        
        y[i+1]=y[i]+h*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        
        k1=f3(t[i],x[i],y[i],z[i])
        k2=f3(t[i]+1/2*h,x[i]+h*1/2*k1,y[i]+h*1/2*k1,z[i]+h*1/2*k1)
        k3=f3(t[i]+1/2*h,x[i]+h*1/2*k2,y[i]+h*1/2*k2,z[i]+h*1/2*k2)
        k4=f3(t[i+1]    ,x[i]+h*k3    ,y[i]+h*k3,    z[i]+h*k3)
        
        z[i+1]=z[i]+h*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
        
    return t,x,y,z

########################################## Método de Runge-Kutta, Regla 3/8 ####################################

def RK38(f1,f2,f3,x0,y0,z0,T,n):
    t=np.linspace(T[0],T[1],n)
    h=abs(t[1]-t[0])
    x=np.empty(n)
    x[0]=x0
    y=np.empty(n)
    y[0]=y0
    z=np.empty(n)
    z[0]=z0
    
    for i in range(n-1):
        
        k1=f1(t[i],x[i],y[i],z[i])
        k2=f1(t[i]+1/3*h,x[i]+h*k1/3        ,y[i]+h*k1/3        ,z[i]+h*k1/3)
        k3=f1(t[i]+2/3*h,x[i]+h*(-1/3*k1+k2),y[i]+h*(-1/3*k1+k2),z[i]+h*(-1/3*k1+k2))
        k4=f1(t[i+1]    ,x[i]+h*(k1-k2+k3)  ,y[i]+h*(k1-k2+k3)  ,z[i]+h*(k1-k2+k3))
        
        x[i+1]=x[i]+h*(1/8*k1+3/8*k2+3/8*k3+1/8*k4)
        
        k1=f2(t[i],x[i],y[i],z[i])
        k2=f2(t[i]+1/3*h,x[i]+h*k1/3        ,y[i]+h*k1/3        ,z[i]+h*k1/3)
        k3=f2(t[i]+2/3*h,x[i]+h*(-1/3*k1+k2),y[i]+h*(-1/3*k1+k2),z[i]+h*(-1/3*k1+k2))
        k4=f2(t[i+1]    ,x[i]+h*(k1-k2+k3)  ,y[i]+h*(k1-k2+k3)  ,z[i]+h*(k1-k2+k3))
        
        y[i+1]=y[i]+h*(1/8*k1+3/8*k2+3/8*k3+1/8*k4)
        
        k1=f3(t[i],x[i],y[i],z[i])
        k2=f3(t[i]+1/3*h,x[i]+h*k1/3        ,y[i]+h*k1/3        ,z[i]+h*k1/3)
        k3=f3(t[i]+2/3*h,x[i]+h*(-1/3*k1+k2),y[i]+h*(-1/3*k1+k2),z[i]+h*(-1/3*k1+k2))
        k4=f3(t[i+1]    ,x[i]+h*(k1-k2+k3)  ,y[i]+h*(k1-k2+k3)  ,z[i]+h*(k1-k2+k3))
        
        z[i+1]=z[i]+h*(1/8*k1+3/8*k2+3/8*k3+1/8*k4)
        
        
    return t,x,y,z

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

#########################################################################################################################