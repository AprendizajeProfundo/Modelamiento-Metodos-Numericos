#!/usr/bin/env python
# coding: utf-8

import numpy as np

#### dx/dt=f(t,x), x(t0)=x0

##################### Método de Euler ###################################################

def Euler(f,x,t,h):
    x+=h*f(t,x)
    t+=h
    return t,x

################## Método del Punto Medio #################################################

def MidPt(f,x,t,h):
    x+=h*f(t+h/2,x+h*f(t,x)/2)
    t+=h   
    return t,x

################################# Método de Heun ############################################

def Heun(f,x,t,h):
    k1=f(t,x)
    x+=h*(k1/2+f(t+h,x+h*k1)/2)
    t+=h
    return t,x

###################### Método de Ralson ######################################################

def Ralson(f,x,t,h):
    k1=f(t,x)
    x+=h*(1/4*k1+3/4*f(t+2/3*h,x+h*2/3*k1))
    t+=h
    return t,x

########################################## Método de Runge-Kutta (RK4) ########################################

def RK4(f,x,t,h):
    k1=f(t,x)
    k2=f(t+h/2,x+h*k1/2)
    k3=f(t+h/2,x+h*k2/2)
    k4=f(t+h  ,x+h*k3)
    x+=h*(1/6*k1+1/3*k2+1/3*k3+1/6*k4)
    t+=h
    return t,x

########################################## Método de Runge-Kutta, Regla 3/8 ####################################

def RK38(f,x,t,h):
    k1=f(t,x)
    k2=f(t+1/3*h,x+h*k1/3)
    k3=f(t+2/3*h,x+h*(-1/3*k1+k2))
    k4=f(t+h    ,x+h*(k1-k2+k3))
    x+=h*(1/8*k1+3/8*k2+3/8*k3+1/8*k4)
    t+=h
    return t,x

################################################# Funciones de Error#####################################################

def Error(aprox,real):
    err=abs(aprox-real)
    return err

def Error_Rela(aprox,real):
    err=abs(aprox-real)/abs(real)
    return err

def MSE(err,approx,real,n):
    err=(err**2+abs(aprox-real)**2)/n