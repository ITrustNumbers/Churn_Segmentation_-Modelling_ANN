import numpy as np
import pickle as pk

#For batch tranformation
LabelEnc = pk.load(open('Transformers/LabelEnc.pkl','rb'))
HotEnc = pk.load(open('Transformers/OneHotEnc.pkl','rb'))
Scaler = pk.load(open('Transformers/Scaler.pkl','rb'))

#For single transformation
Label_Gen = {'Female':0,'Male':1}
Label_Coun = {'Germany':[1,0],'France':[0,0],'Spain':[0,1]}

def batch_transform(val):
    
    if type(val) != 'numpy.ndarray':
        val = np.array(val)
    
    val[:,2] = LabelEnc.transform(val[:,2])
    val = HotEnc.transform(val)[:,1:]
    val = Scaler.transform(val)
    
    return val

def transform(orig):

    val = orig.copy()
    val[2] = Label_Gen[val[2]]
    val =  Label_Coun[val[1]] + [val[0]] + val[2:]
    if type(val) != 'numpy.ndarray':
        val = np.array(val)
    val = Scaler.transform(val.reshape(1,-1))
    return val