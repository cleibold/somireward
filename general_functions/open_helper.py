import numpy as N                   
import os
import openEphys as op
import scipy.io as sio

def open_helper(file_selector):
       
    ext=file_selector[len(file_selector)-3:]        
            
    if ext==".js" or ext=="nd":
        dataTemp = N.fromfile(file_selector)
            
        sample=dataTemp[0]
            
        data=N.empty((len(dataTemp)))
        for n in range(len(data)):
            data[n]=dataTemp[n]
        dataTemp=0   
    elif ext=="npy":
        data=N.load(file_selector,allow_pickle=True)    
    
    elif ext=="txt":
        f=open(file_selector, "r")
        g=f.read()
        g=g.split()
        h=N.asarray(g)
        data = N.empty((len(h)))
        for n in range(len(h)):
            data[n] = float(h[n])
            
    elif ext=="dat":
        f=open(file_selector)
        data=N.fromfile(f,dtype=N.int16) # Convert data to microV
        f.close()
    
    elif ext=="ous":
        data=op.load(file_selector)
        data=data['data']    
    
        
    else:
        print "Specify action!"
        
    return data
       










