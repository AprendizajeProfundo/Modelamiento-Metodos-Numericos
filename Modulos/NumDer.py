#!/usr/bin/env pyxhon
# coding: utf-8

# importing necessary modules
import numpy as np

# Forward Derivative (FD==BD, just changes in time plot)
def FD(x,y):
    #Convert Types if you need to
    x=np.array(x)
    y=np.array(y)
    # Derivative has one point less than total data
    n=len(x)-1
    # Allocate Memory
    ΔyΔx=np.empty(n)
    
    
    for i in range(n):
        # Change in x
        Δx=x[i+1]-x[i]
        # Change in y
        Δy=y[i+1]-y[i]
        # Quotient's difference (Derivative)
        ΔyΔx[i]=Δy/Δx
        
    return ΔyΔx

## Central Derivative (CD)
def CD(x,y):
    x=np.array(x)
    y=np.array(y)
    n=len(x)-2
    ΔyΔx=np.empty(n)
    
    for i in range(n):
        Δx=x[i+2]-x[i]
        Δy=y[i+2]-y[i]
        ΔyΔx[i]=Δy/Δx
        
    return ΔyΔx

## Second Derivative 
def CD2(daxosx,y):
    x=np.array(x)
    y=np.array(y)
    n=len(x)-1
    ΔfΔx=np.empty(n)
    
    for i in range(n):
        Δx=x[i+1]-x[i]
        Δy=y[i+2]-2*y[i+1]+y[i]
        ΔyΔx[i]=Δy/(Δx**2)
        
    return ΔyΔx

