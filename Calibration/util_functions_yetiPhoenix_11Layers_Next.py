import sys
import os
from sys import platform
from  os.path import isfile

from math import sqrt, exp, log, erf, floor
import numpy 
import pandas
from decimal import *
import tensorflow.keras
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.models import model_from_json
import matplotlib
import matplotlib.pyplot as pyplot
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import mpl_toolkits
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.axes3d import Axes3D

from sklearn import preprocessing
from sklearn import model_selection

import joblib
import seaborn as sns
import scipy
from IPython.display import display
import time
import tensorflow as tf
getcontext().prec = 8
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual, FloatProgress
import ipywidgets as widgets
from matplotlib import ticker, cm
from scipy.interpolate import interp1d,interp2d
import numpy as np
from scipy.stats import norm
import random


nbdate=12
nbTraj=10000

class metaparameters(object):
    def __init__(self):
        self.INPUT_DIM=4
        self.INPUT_GOAL=0
        self.NB_NEURON_PRINCIPAL =20
        self.ACTIVATION_PRINCIPALE = 'relu'
        self.ACTIVATION_PRINCIPALE_FINALE = 'linear'
        self.ADDITIONAL_LEARNING_NB_EPOCH=1000
        self.NBLAYERS =11
        self.VERBOSE_FLAG=2
        self.NB_ADDITIONAL_LEARNING =0
        self.OPTIMIZER = 'adams'
        self.PATH=''
        self.LEARNINGBASE_ORIGIN=""
        self.LEARNINGBASE_BUT=""
        self.nbDate=12
        self.nbTraj=10000
        self.X_scaler = 0
        self.X_scaled = 0
        self.X = 0
        self.pkgPath = 0
        self.smallestNbDate = 4
        self.largestNbDate = 24
        self.smallestmaturity = 1
        self.largestmaturity = 5
        self.smallestF = 70
        self.largestF = 150
        self.largestmu = 0.05
        self.smallestmu = 0.0
        self.smallestSigma = 0.15
        self.largestSigma = 0.3
        self.smallestRho = 0.4
        self.largestRho = 1
        self.smallestBonus = -2
        self.largestBonus = 2
        self.smallestYetiBarrier = 90
        self.largestYetiBarrier = 110
        self.smallestYetiCoupon = 0
        self.largestYetiCoupon = 2
        self.smallestPhoenixBarrier = 80
        self.largestPhoenixBarrier = 100
        self.smallestPhoenixCoupon = 0.5
        self.largestPhoenixCoupon = 2
        self.smallestPDIBarrier = 40
        self.largestPDIBarrier = 70
        self.smallestPDIGearing = -5
        self.largestPDIGearing = +5
        self.smallestPDIStrike = 40
        self.largestPDIStrike = 70
        self.smallestPDIType = -1
        self.largestPDIType = 1



#pricing function
def bs(f,k,v,t):
    if (f<=0) :
        print("f<=0 !")
        print(f)
    if (v<=0) :
        print("v<=0 !")
        print(v)
    sigTsquared = sqrt(Decimal(t)/365)*v
    d1 = (log(f/k)+(.5*(v**2))*t/365)/sigTsquared
    d2 = d1-sigTsquared
    Nd1 = (1+erf(d1/sqrt(2)))/2
    Nd2 = (1+erf(d2/sqrt(2)))/2
    iNd1 = (1+erf(-d1/sqrt(2)))/2
    iNd2 = (1+erf(-d2/sqrt(2)))/2
    callPrice = round(f*Nd1-k*Nd2, 4)
    return callPrice


def bsTimeValue(f,k,v,t):
    sigTsquared = sqrt(t/365.0)*v
    d1 = (log(f/k)+(.5*(v**2))*t/365.0)/sigTsquared
    d2 = d1-sigTsquared
    Nd1 = (1.0+erf(d1/sqrt(2)))/2
    Nd2 = (1.0+erf(d2/sqrt(2)))/2
    iNd1 = (1.0+erf(-d1/sqrt(2)))/2
    iNd2 = (1.0+erf(-d2/sqrt(2)))/2
    callPrice = round(f*Nd1-k*Nd2-max(0,f-k), 8)
    return callPrice

def bsPut(f,k,v,t):
    sigTsquared = sqrt(t/365.0)*v
    d1 = (log(f/k)+(.5*(v**2))*t/365)/sigTsquared
    d2 = d1-sigTsquared
    Nd1 = (1+erf(d1/sqrt(2)))/2
    Nd2 = (1+erf(d2/sqrt(2)))/2
    iNd1 = (1+erf(-d1/sqrt(2)))/2
    iNd2 = (1+erf(-d2/sqrt(2)))/2
    putPrice = round(k*iNd2-f*iNd1, 4)
    return putPrice

def displayDistribution(Y):
    sortedEcarts=numpy.sort(Y)
    datasize=Y.size
    abcisseDist=numpy.zeros(datasize)
    for i in range(datasize):
        abcisseDist[i]=(i+0.5)/datasize
    pyplot.plot(sortedEcarts,abcisseDist)
    return

def CalibrationPrincipalOrder(Ycall,cnfigName,params,save=True,generate_errors = False):
    
    inData=Input(shape=(params.INPUT_DIM,))
    layer0 = Dense(256*params.NB_NEURON_PRINCIPAL,activation=params.ACTIVATION_PRINCIPALE,name='Layer0')(inData)
    layer1 = Dense(128*params.NB_NEURON_PRINCIPAL,activation=params.ACTIVATION_PRINCIPALE,name='Layer1')(layer0)
    layer2 = Dense(64*params.NB_NEURON_PRINCIPAL,activation=params.ACTIVATION_PRINCIPALE,name='Layer2')(layer1)
    layer3 = Dense(32*params.NB_NEURON_PRINCIPAL,activation=params.ACTIVATION_PRINCIPALE,name='Layer3')(layer2)
    layer4 = Dense(16*params.NB_NEURON_PRINCIPAL, activation=params.ACTIVATION_PRINCIPALE,name='Layer4')(layer3)
    layer5 = Dense(8*params.NB_NEURON_PRINCIPAL, activation=params.ACTIVATION_PRINCIPALE,name='Layer5')(layer4)
    layer6 = Dense(4*params.NB_NEURON_PRINCIPAL, activation=params.ACTIVATION_PRINCIPALE,name='Layer6')(layer5)
    layer7 = Dense(2*params.NB_NEURON_PRINCIPAL, activation=params.ACTIVATION_PRINCIPALE,name='Layer7')(layer6)
    layer8 = Dense(params.NB_NEURON_PRINCIPAL, activation=params.ACTIVATION_PRINCIPALE,name='Layer8')(layer7)
    prixfinal = Dense(1,activation=params.ACTIVATION_PRINCIPALE_FINALE)(layer8)
    model = Model(inputs = inData, outputs = prixfinal)
    
    # grab the number of GPUs and store it in a conveience variable
    if platform == 'linux' :
        G = 4
    # make the model parallel
        model =keras.utils.multi_gpu_model(model, gpus=G)
    
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    # Data Scaling from 0 to 1, X and y originally have very different scales.
    y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    y_scaled = (y_scaler.fit_transform(Ycall.reshape(-1, 1)))
    # Preparing test and train data: 60% training, 40% testing.
    X_train, X_test, y_train, y_test = model_selection.train_test_split( \
            params.X_scaled, y_scaled, test_size=0.10, random_state=3)
    # New sequential network structure.
    
    # Training model with train data. Fixed random seed:
    checkpointer = keras.callbacks.ModelCheckpoint(filepath = "BestWeights.hdf5",verbose=1,\
                                                   save_best_only=True , monitor = 'loss')
    numpy.random.seed(3)
    model.fit(X_train, y_train, epochs=params.INITIAL_LEARNING_NB_EPOCH, batch_size=params.BATCH_SIZE_PRINCIPAL,\
              verbose=params.VERBOSE_FLAG, shuffle = True, callbacks=[checkpointer])
    
     #fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
     #   validation_split=0.0, validation_data=None, 
     #   shuffle=True, class_weight=None, sample_weight=None, 
     #   initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    
    Y_ordre1 = numpy.zeros(Ycall.size)
    model.load_weights("BestWeights.hdf5")
    if (generate_errors):
    	print("creating the residus for the boosting")
    	for i in range(Ycall.size):
        	normalizedData=params.X_scaler.transform(params.X[i].reshape(1, -1))
        	normalizedPrediction=model.predict(normalizedData)
        	outputData=y_scaler.inverse_transform(normalizedPrediction)
        	Y_ordre1[i]=outputData[0,0]-Ycall[i]
    # sauvegarde des parametres du modele
    # save weights to HDF5
    print("starting to save")
    if (save):
        model.save_weights(params.pkgPath + '/' + cnfigName +'.hdf5')
        # save model to JSON
        model_json = model.to_json()
        with open(params.pkgPath + '/' + cnfigName +'.json', 'w') as json_file:
            json_file.write(model_json)
        # Save scaler Xand Y
        filename = params.pkgPath + '/' + cnfigName +'scalerX.pkl'
        _= joblib.dump(params.X_scaler , filename , compress = 9)
        filename = params.pkgPath + '/' + cnfigName +'scalerY.pkl'
        _= joblib.dump(y_scaler , filename , compress = 9)
        filename = params.pkgPath + '/' + cnfigName +'scaledY.pkl'
        _= joblib.dump(y_scaled , filename , compress = 9)
        filename = params.pkgPath + '/' + cnfigName +'Y_ordre1.res'
        _= joblib.dump(Y_ordre1 , filename , compress = 9)
        print("Saved model to disk as " + cnfigName)
    return Y_ordre1,model,y_scaler,y_scaled

def reloadPrincipalOrder(cnfigName,pkgPath):
    json_file = open(pkgPath + '/' + cnfigName +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(pkgPath + '/' + cnfigName +'.hdf5')
    # load scaler Xand Y
    filename = pkgPath + '/' + cnfigName +'scalerX.pkl'
    X_scaler_loaded = joblib.load( filename )
    filename = pkgPath + '/' + cnfigName +'scalerY.pkl'
    y_scaler_loaded = joblib.load( filename )

    filename = pkgPath + '/' + cnfigName +'scaledY.pkl'
    y_scaled_loaded=[]
    if isfile(filename):
        y_scaled_loaded = joblib.load( filename )
    filename = pkgPath + '/' + cnfigName +'Y_ordre1.res'
    Y_ordre1_loaded = joblib.load( filename )
    return loaded_model,X_scaler_loaded,y_scaler_loaded,y_scaled_loaded,Y_ordre1_loaded


def CalibrationRestartFromPkg(pkgPath,Ycall,y_scaler,y_scaled,cnfigName,params,save=True,generate_errors = False):
    json_file = open(pkgPath + '/' + cnfigName +'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    if platform == 'linux' :
    # make the model parallel
        G = 4
        model =keras.utils.multi_gpu_model(model, gpus=G)
    
    model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
    model.load_weights(pkgPath + '/' + cnfigName +'.hdf5')
    # load scaler Xand Y
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split( \
            params.X_scaled, y_scaled, test_size=0.10, random_state=3)
    # Training model with train data. Fixed random seed:
    checkpointer = keras.callbacks.ModelCheckpoint(filepath = "BestWeights.hdf5",verbose=1,\
                                                   save_best_only=True , monitor = 'loss')
    numpy.random.seed(3)
    model.fit(X_train, y_train, epochs=params.ADDITIONAL_LEARNING_NB_EPOCH, batch_size=params.BATCH_SIZE_PRINCIPAL,\
              verbose=params.VERBOSE_FLAG, shuffle = True, callbacks=[checkpointer])
    
     #fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
     #   validation_split=0.0, validation_data=None, 
     #   shuffle=True, class_weight=None, sample_weight=None, 
     #   initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    
    Y_ordre1 = numpy.zeros(Ycall.size)
    model.load_weights("BestWeights.hdf5")
    if (generate_errors):
        print("creating the residus for the boosting")
        for i in range(Ycall.size):
            normalizedData=params.X_scaler.transform(params.X[i].reshape(1, -1))
            normalizedPrediction=model.predict(normalizedData)
            outputData=y_scaler.inverse_transform(normalizedPrediction)
            Y_ordre1[i]=outputData[0,0]-Ycall[i]
    # sauvegarde des parametres du modele
    # save weights to HDF5
    print("starting to save")
    if (save):
        model.save_weights(params.pkgPath + '/' + cnfigName +'.hdf5')
        # save model to JSON
        model_json = model.to_json()
        with open(params.pkgPath + '/' + cnfigName +'.json', 'w') as json_file:
            json_file.write(model_json)
        # Save scaler Xand Y
        filename = params.pkgPath + '/' + cnfigName +'scalerX.pkl'
        _= joblib.dump(params.X_scaler , filename , compress = 9)
        filename = params.pkgPath + '/' + cnfigName +'scalerY.pkl'
        _= joblib.dump(y_scaler , filename , compress = 9)
        filename = params.pkgPath + '/' + cnfigName +'Y_ordre1.res'
        _= joblib.dump(Y_ordre1 , filename , compress = 9)
        print("Saved model to disk as " + cnfigName)
    params.NB_ADDITIONAL_LEARNING +=1
    print('number of additional learning : ' + str(params.NB_ADDITIONAL_LEARNING))
    notefinale =Compute_Note(model,['Data/Notation1.CSV','Data/Notation2.CSV','Data/Notation3.CSV','Data/Notation4.CSV'],\
            params,y_scaler)
    print('notef inale =',notefinale)
    return Y_ordre1,model,y_scaler,y_scaled

def BulleListPar(i,NbLayers):
    return [floor(exp(2.5-2*abs(x-i)/(0.5*NbLayers)))+1 for x in range(NbLayers)]


def ReinforceOptimality(pkgNameOriginal,pkgName2Initial,NbLayers,Ycall,y_scaler,y_scaled,cnfigName,params,save=True,generate_errors = False):
    
    pkgName=pkgNameOriginal    
     # load scaler Xand Y
    X_train, X_test, y_train, y_test = model_selection.train_test_split( \
                params.X_scaled, y_scaled, test_size=0.10, random_state=3)
    i =0
    nbloop = 10
    resultats = [[0,0,0] for k in range(nbloop)]
    for k in range(nbloop):
        i=random.randint(0, NbLayers-1)
        
        pkgPath=params.PATH + "/" + pkgName   
        json_file = open(pkgPath + '/' + cnfigName +'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        pkgName2=pkgName2Initial+"-loop-"+str(k)
        deltaNbNeuronList = BulleListPar(i,NbLayers)
        pkgPath2=params.PATH + "/" + pkgName2
        os.chdir(params.PATH)
        if not(os.path.isdir(pkgPath2)):
            os.makedirs(pkgPath2,0o777 )
        model = model_from_json(loaded_model_json)              ### chargment du model initial (1)
        model.load_weights(pkgPath + '/' + cnfigName +'.hdf5')  ### chargment du model initial (2)

        noteinitiale =Compute_Note(model,['Data/Notation1.CSV','Data/Notation2.CSV','Data/Notation3.CSV','Data/Notation4.CSV'],\
                params,y_scaler)
        print('noteinitiale=',noteinitiale)
        resultats[k][0]=noteinitiale
        resultats[k][1]=i
        weights1=model.get_weights()
        shape_list = list(map(lambda k: k.shape ,weights1))
        initial_NbNeuron = shape_list[0][0]
        layer_NbNeuron = [shape_list[2*k][1] for k in range(9)]
        layer_NbNeuron.append(1)

        inData=Input(shape=(params.INPUT_DIM,))
        layer0 = Dense(layer_NbNeuron[0]+deltaNbNeuronList[0],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer0')(inData)
        layer1 = Dense(layer_NbNeuron[1]+deltaNbNeuronList[1],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer1')(layer0)
        layer2 = Dense(layer_NbNeuron[2]+deltaNbNeuronList[2],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer2')(layer1)
        layer3 = Dense(layer_NbNeuron[3]+deltaNbNeuronList[3],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer3')(layer2)
        layer4 = Dense(layer_NbNeuron[4]+deltaNbNeuronList[4],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer4')(layer3)
        layer5 = Dense(layer_NbNeuron[5]+deltaNbNeuronList[5],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer5')(layer4)
        layer6 = Dense(layer_NbNeuron[6]+deltaNbNeuronList[6],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer6')(layer5)
        layer7 = Dense(layer_NbNeuron[7]+deltaNbNeuronList[7],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer7')(layer6)
        layer8 = Dense(layer_NbNeuron[8]+deltaNbNeuronList[8],\
                       activation=params.ACTIVATION_PRINCIPALE,name='Layer8')(layer7)
        prixfinal = Dense(1,activation=params.ACTIVATION_PRINCIPALE_FINALE)(layer8)
        model2 = Model(inputs = inData, outputs = prixfinal)

        if platform == 'linux' :
        # make the model parallel
            G = 4
            model2 =keras.utils.multi_gpu_model(model2, gpus=G)

        model2.compile(loss='mse', optimizer=params.OPTIMIZER, metrics=["accuracy"])


        #layer 0
        weights1[0] =np.transpose(np.concatenate((np.transpose(weights1[0]),numpy.zeros((deltaNbNeuronList[0],initial_NbNeuron)))))
        weights1[1] =np.concatenate((weights1[1],numpy.zeros(deltaNbNeuronList[0])))
        weights1[2] =np.concatenate((weights1[2],numpy.zeros((deltaNbNeuronList[0],layer_NbNeuron[1]))))

        #layer 1-8
        for nlayer in range(1,9):
            weights1[2*nlayer] =np.transpose(np.concatenate((np.transpose(weights1[2*nlayer]),
                                        numpy.zeros((deltaNbNeuronList[nlayer],layer_NbNeuron[nlayer-1]+deltaNbNeuronList[nlayer-1])))))
            weights1[2*nlayer+1] =np.concatenate((weights1[2*nlayer+1],numpy.zeros(deltaNbNeuronList[nlayer])))
            weights1[2*nlayer+2] =np.concatenate((weights1[2*nlayer+2],numpy.zeros((deltaNbNeuronList[nlayer],layer_NbNeuron[nlayer+1]))))

        model2.set_weights(weights1)
        print("New model synthesized")
       
        # Training model with train data. Fixed random seed:
        checkpointer = keras.callbacks.ModelCheckpoint(filepath = "BestWeights.hdf5",verbose=1,\
                                                       save_best_only=True , monitor = 'loss')
        numpy.random.seed(3)
        print("starting fit")
        startCalibrationTime = time.time()
        model2.fit(X_train, y_train, epochs=params.ADDITIONAL_LEARNING_NB_EPOCH, batch_size=params.BATCH_SIZE_PRINCIPAL,\
                  verbose=params.VERBOSE_FLAG, shuffle = True, callbacks=[checkpointer])
        endCalibrationTime = time.time()
        print('temps de calibration ',endCalibrationTime - startCalibrationTime)


         #fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
         #   validation_split=0.0, validation_data=None, 
         #   shuffle=True, class_weight=None, sample_weight=None, 
         #   initial_epoch=0, steps_per_epoch=None, validation_steps=None)

        Y_ordre1 = numpy.zeros(Ycall.size)
        model2.load_weights("BestWeights.hdf5")
        if (generate_errors):
            print("creating the residus for the boosting")
            for i in range(Ycall.size):
                normalizedData=params.X_scaler.transform(params.X[i].reshape(1, -1))
                normalizedPrediction=model.predict(normalizedData)
                outputData=y_scaler.inverse_transform(normalizedPrediction)
                Y_ordre1[i]=outputData[0,0]-Ycall[i]
        # sauvegarde des parametres du modele
        # save weights to HDF5
        print("starting to save")
        if (save):
            model2.save_weights(pkgPath2 + '/' + cnfigName +'.hdf5')
            # save model to JSON
            model_json = model2.to_json()
            with open(pkgPath2 + '/' + cnfigName +'.json', 'w') as json_file:
                json_file.write(model_json)
            # Save scaler Xand Y
            filename = pkgPath2 + '/' + cnfigName +'scalerX.pkl'
            _= joblib.dump(params.X_scaler , filename , compress = 9)
            filename = pkgPath2 + '/' + cnfigName +'scalerY.pkl'
            _= joblib.dump(y_scaler , filename , compress = 9)
            filename = pkgPath2 + '/' + cnfigName +'Y_ordre1.res'
            _= joblib.dump(Y_ordre1 , filename , compress = 9)
            print("Saved model to disk as " + cnfigName)
        params.NB_ADDITIONAL_LEARNING +=1
        print('number of additional learning : ' + str(params.NB_ADDITIONAL_LEARNING))
        notefinale =Compute_Note(model2,['Data/Notation1.CSV','Data/Notation2.CSV','Data/Notation3.CSV','Data/Notation4.CSV'],\
                params,y_scaler)
        resultats[k][2]=notefinale
        print('notefinale=',notefinale)
        pkgName=pkgName2
    return Y_ordre1,model2,y_scaler,y_scaled

def reloadDataScalers():
    filename = 'DataScalerX.pkl'
    DataScalerX = joblib.load( filename )
    filename = 'DataScalerY.pkl'
    DataScalerY = joblib.load( filename )
    return [DataScalerX,DataScalerY]

def Compute_Note(model_Option,csvlist,params,y_scaler_Option):
    TotalCodedPriceNote=0
    TotalRealPriceNote=0
    TotalInstruments=0
    _,DataScalerY=reloadDataScalers()
    for file in csvlist:
        os.chdir(params.PATH)
        dataframe = pandas.read_csv(file, header=0,sep=",",decimal=".")
        datasetNotation = dataframe.values
        XNotation = datasetNotation[1:,1:params.INPUT_DIM+1]
        dataNotationsize=datasetNotation[1:,params.INPUT_DIM].size
        XNotation_scaled = (params.X_scaler.fit_transform(XNotation))
        Ycall_Notation = datasetNotation[1:,params.INPUT_GOAL]
        normedOptions=y_scaler_Option.inverse_transform(model_Option.predict(XNotation))
        normederrors = Ycall_Notation - normedOptions.reshape(1,-1)
        priceerrors=DataScalerY.inverse_transform(numpy.array([Ycall_Notation]))-DataScalerY.inverse_transform(numpy.array(normedOptions.reshape(1,-1)))
        CodedPriceNote = numpy.sum(normederrors[0]*normederrors[0])
        RealPriceNote = numpy.sum(priceerrors[0]*priceerrors[0])
        nbInst =datasetNotation.shape[0]
        TotalCodedPriceNote+=CodedPriceNote
        TotalRealPriceNote+=RealPriceNote
        TotalInstruments+=datasetNotation.shape[0]
    TotalCodedPriceNote/=TotalInstruments
    TotalRealPriceNote/=TotalInstruments
    note=sqrt(TotalRealPriceNote)
    print('Note =',note)
    return note



def displayerror(Y_error,label):
    sns.distplot(Y_error)
    pyplot.xlabel("repricing error")
    pyplot.ylabel("probability")
    pyplot.title(label+": density")
    pyplot.show()
    pyplot.xlabel("repricing error")
    pyplot.ylabel("probability")
    pyplot.title(label + ": distribution")
    displayDistribution(Y_error)
    pyplot.show()
    return

def displaycomparaison(Y_error1,Y_error2,Y_error3,label,label1,label2,\
                       label3):
    testlen=int(Y_error1.size) 
   
    x_distrib =numpy.arange(testlen)/testlen
    fig=pyplot.figure()
    fig.suptitle(label + ": distribution")
    ax=fig.add_subplot(111)
    ax.plot(sorted(Y_error1),x_distrib, color="blue",label = 'Principal Value')
    ax.plot(sorted(Y_error2),x_distrib, color="green", label='Boosted Level 1')
    ax.plot(sorted(Y_error3),x_distrib, color="red", label='Boosted Level 2')
    
    ax.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
    ax.set_xlabel("repricing error")
    ax.set_ylabel("probability")
    
    pyplot.show()
    return

def reloadAllModels(X,pkgPath):
    X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_scaled = (X_scaler.fit_transform(X))
    model_Option,X_scaler,y_scaler_Option,y_scaled_Option,Y_ordre1_Option =reloadPrincipalOrder("Option",pkgPath)  
    filename = 'DataScalerX.pkl'
    DataScalerX = joblib.load( filename )
    filename = 'DataScalerY.pkl'
    DataScalerY = joblib.load( filename )
    learned_models=[model_Option,DataScalerX,DataScalerY,X_scaler,X_scaled,y_scaler_Option,y_scaled_Option,\
        Y_ordre1_Option]
    return learned_models
    

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            for col in row:
                html.append("<td>{0}</td>".format(col))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


def predict_OptionCall(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,
              Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon,\
              PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,learnedModels,BoostOrder):   
    model_Option,DataScalerX,DataScalerY,X_scaler,X_scaled,y_scaler_Option,y_scaled_Option,Y_ordre1_Option= learnedModels 
    inputData=numpy.asarray([[s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,
    PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,0]]).reshape(1, -1)
    normalizedData=DataScalerX.transform(inputData) 
    NoPricenormalisedData=normalizedData[:,0:23]
    normalizedOption= y_scaler_Option.inverse_transform(model_Option.predict(NoPricenormalisedData))
    Option=DataScalerY.inverse_transform(normalizedOption)[0,0]
    return Option,normalizedOption

def predict_OptionCall_new(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,
              Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon,\
              PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,BoostOrder,\
              model_Option,DataScalerX,DataScalerY,y_scaler_Option):   
    
    inputData=numpy.asarray([[s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,
    PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,0]]).reshape(1, -1)
    normalizedData=DataScalerX.transform(inputData) 
    NoPricenormalisedData=normalizedData[:,0:23]
    normalizedOption= y_scaler_Option.inverse_transform(model_Option.predict(NoPricenormalisedData))
    Option=DataScalerY.inverse_transform(normalizedOption)[0,0]
    return Option,normalizedOption

def computeAll(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,learnedModels): 
    p10 ,p10a= predict_OptionCall(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,learnedModels,0)  
    table = ListTable()
    table.append(['s1','s2','s3','mu1','mu2','mu3','v1','v2','v3','c12','c13','c23','Bonus','YetiBarrier','YetiCoupon',\
              'PhoenixBarrier', 'PhoenixCoupon', 'PDIBarrier', 'PDIGearing', 'PDIStrike', 'PDIType','T','NbDate'])
    nbTraj=100000
    cov=numpy.asarray([[1,c12,c13],[c12,1,c23],[c13,c23,1]])
    p2=YetiPhoenixPriceMC(numpy.asarray([s1,s2,s3]),numpy.asarray([mu1,mu2,mu3]),numpy.asarray([v1,v2,v3]),cov,\
          Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,nbTraj)
    table.append(["%.4f" % s1,"%.4f" % s2 ,"%.4f" % s3,\
                  "%.4f" % mu1,"%.4f" % mu2 ,"%.4f" % mu3,"%.4f" % v1,"%.4f" % v2 ,"%.4f" % v3,\
                  "%.4f" % c12,"%.4f" % c13 ,"%.4f" % c23, "%.4f" % Bonus, "%.4f" % YetiBarrier,"%.4f" % YetiCoupon ,\
                  "%.4f" % PhoenixBarrier,"%.4f" % PhoenixCoupon ,\
                  "%.4f" % PDIBarrier,"%.4f" % PDIGearing,"%.4f" % PDIStrike,"%.4f" % PDIType,"%.4f" % T,"%.4f" % NbDate])
    table.append([ 'prediction Boost 0','recomputed'])
    table.append(["%.4f" % p10,"%.4f" % p2])
    return table


def computeAll_new(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,\
              model_Option,DataScalerX,DataScalerY,y_scaler_Option): 
    p10 ,p10a= predict_OptionCall_new(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,0,model_Option,DataScalerX,DataScalerY,y_scaler_Option)  
    table = ListTable()
    table.append(['s1','s2','s3','mu1','mu2','mu3','v1','v2','v3','c12','c13','c23','Bonus','YetiBarrier','YetiCoupon',\
              'PhoenixBarrier', 'PhoenixCoupon', 'PDIBarrier', 'PDIGearing', 'PDIStrike', 'PDIType','T','NbDate'])
    nbTraj=100000
    cov=numpy.asarray([[1,c12,c13],[c12,1,c23],[c13,c23,1]])
    p2=YetiPhoenixPriceMC(numpy.asarray([s1,s2,s3]),numpy.asarray([mu1,mu2,mu3]),numpy.asarray([v1,v2,v3]),cov,\
          Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,nbTraj)
    table.append(["%.4f" % s1,"%.4f" % s2 ,"%.4f" % s3,\
                  "%.4f" % mu1,"%.4f" % mu2 ,"%.4f" % mu3,"%.4f" % v1,"%.4f" % v2 ,"%.4f" % v3,\
                  "%.4f" % c12,"%.4f" % c13 ,"%.4f" % c23, "%.4f" % Bonus, "%.4f" % YetiBarrier,"%.4f" % YetiCoupon ,\
                  "%.4f" % PhoenixBarrier,"%.4f" % PhoenixCoupon ,\
                  "%.4f" % PDIBarrier,"%.4f" % PDIGearing,"%.4f" % PDIStrike,"%.4f" % PDIType,"%.4f" % T,"%.4f" % NbDate])
    table.append([ 'prediction Boost 0','recomputed'])
    table.append(["%.4f" % p10,"%.4f" % p2])
    return table


""" ca fait 12 dim pour les sigmas et la trice correl """

def Return_Distribution(sigmas,correls,s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,\
              model_Option,DataScalerX,DataScalerY,y_scaler_Option,nbsample): 
    cov = np.zeros([12,12])
    for i in range(12):
        for j in range(12):
            cov[i,j]=sigmas[i]*sigmas[j]*correls[i,j]
    means = np.array([log(s1),log(s2),log(s3),mu1,mu2,mu3,log(v1),log(v2),log(v3),np.arctanh(c12),np.arctanh(c13),np.arctanh(c23)])
    draws = np.random.multivariate_normal(means, cov, size=nbsample)
    prices=np.zeros(nbsample)
    for i in range(nbsample):
        marketvars = draws[i]
        prices[i] ,_= predict_OptionCall_new(exp(marketvars[0]),exp(marketvars[1]),exp(marketvars[2]),\
            marketvars[3],marketvars[4],marketvars[5],exp(marketvars[6]),exp(marketvars[7]),exp(marketvars[8]),\
            np.tanh(marketvars[9]),np.tanh(marketvars[10]),np.tanh(marketvars[11]),Bonus,YetiBarrier,YetiCoupon,\
            PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,0,model_Option,DataScalerX,DataScalerY,y_scaler_Option) 
    p0 ,_= predict_OptionCall_new(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,0,model_Option,DataScalerX,DataScalerY,y_scaler_Option)    
    prices.sort()
    return prices-p0
   

def Var_Compute(sigmas,correls,s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,\
              model_Option,DataScalerX,DataScalerY,y_scaler_Option,nbsample,levels): 
    prices = Return_Distribution(sigmas,correls,s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,T,NbDate,\
              model_Option,DataScalerX,DataScalerY,y_scaler_Option,nbsample)
    abcisses=np.zeros(nbsample)
    for i in range(nbsample):
        abcisses[i]=i/nbsample
    f =interp1d(abcisses,prices,kind='cubic')
    res=np.transpose(list(map(f,levels)))
    return res


def schowcase(icase, dataset,INPUT_GOAL,learnedModels):    
    model_Option,DataScalerX,DataScalerY,X_scaler,X_scaled,y_scaler_Option,y_scaled_Option,Y_ordre1_Option= learnedModels 
    inputData = numpy.array([dataset[icase, 1:25]])
    unnormalizedData=DataScalerX.inverse_transform(inputData).reshape(-1,1)
   
    s1 = unnormalizedData[ 0,0]
    s2 = unnormalizedData[ 1,0]
    s3 = unnormalizedData[ 2,0]
    mu1 = unnormalizedData[ 3,0]
    mu2 = unnormalizedData[ 4,0]
    mu3 = unnormalizedData[ 5,0]
    v1 = unnormalizedData[ 6,0]
    v2 = unnormalizedData[ 7,0]
    v3 = unnormalizedData[ 8,0]
    c12 = unnormalizedData[ 9,0]
    c13 = unnormalizedData[ 10,0]
    c23 = unnormalizedData[ 11,0]  
    Bonus = unnormalizedData[ 12,0]  
    YetiBarrier = unnormalizedData[ 13,0]
    YetiCoupon = unnormalizedData[ 14,0]
    PhoenixBarrier = unnormalizedData[ 15,0]
    PhoenixCoupon = unnormalizedData[ 16,0]
    PDIBarrier = unnormalizedData[ 17,0]
    PDIGearing = unnormalizedData[ 18,0]
    PDIStrike = unnormalizedData[ 19,0]
    PDIType = unnormalizedData[ 20,0]
    maturity = unnormalizedData[ 21,0]
    NbDate = unnormalizedData[ 22,0]

    p10,p10a = predict_OptionCall(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,maturity,NbDate,learnedModels,0)  
    
    cov=numpy.asarray([[1,c12,c13],[c12,1,c23],[c13,c23,1]])
    nbTraj=100000
    NbDate=int(NbDate)
    normcallStored = dataset[icase,INPUT_GOAL]
    callStoredrecomputed = DataScalerY.inverse_transform(numpy.array([[normcallStored]]))
    callRecomputed=YetiPhoenixPriceMC(numpy.asarray([s1,s2,s3]),numpy.asarray([mu1,mu2,mu3]),numpy.asarray([v1,v2,v3]),cov,\
                          Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon,\
                          PDIBarrier, PDIGearing, PDIStrike, PDIType,maturity,NbDate,nbTraj)
    table = ListTable()
    table.append(['s1','s2','s3','mu1','mu2','mu3','v1','v2','v3','c12','c13','c23','Bonus','YetiBarrier','YetiCoupon',\
              'PhoenixBarrier', 'PhoenixCoupon', 'PDIBarrier', 'PDIGearing', 'PDIStrike', 'PDIType','maturity','NbDate'])
    
    table.append(["%.4f" % s1,"%.4f" % s2 ,"%.4f" % s3,\
                  "%.4f" % mu1,"%.4f" % mu2 ,"%.4f" % mu3,"%.4f" % v1,"%.4f" % v2 ,"%.4f" % v3,\
                  "%.4f" % c12,"%.4f" % c13 ,"%.4f" % c23, "%.4f" % Bonus, "%.4f" % YetiBarrier,"%.4f" % YetiCoupon ,\
                  "%.4f" % PhoenixBarrier,"%.4f" % PhoenixCoupon ,\
                  "%.4f" % PDIBarrier,"%.4f" % PDIGearing,"%.4f" % PDIStrike,"%.4f" % PDIType,"%.4f" % maturity,"%.4f" % NbDate])
    table.append([ 'prediction boost 0','call recomputed','callStored recomputed','normcall stored',"normCall b0"])
    table.append(["%.4f" % p10 , "%.4f" % callRecomputed,"%.4f" % callStoredrecomputed, "%.4f" % normcallStored, "%.4f" % p10a])
    return table

def schowcase_new(icase, dataset,INPUT_GOAL,model_Option,DataScalerX,DataScalerY,y_scaler_Option):    
    inputData = numpy.array([dataset[icase, 1:25]])
    unnormalizedData=DataScalerX.inverse_transform(inputData).reshape(-1,1)
   
    s1 = unnormalizedData[ 0,0]
    s2 = unnormalizedData[ 1,0]
    s3 = unnormalizedData[ 2,0]
    mu1 = unnormalizedData[ 3,0]
    mu2 = unnormalizedData[ 4,0]
    mu3 = unnormalizedData[ 5,0]
    v1 = unnormalizedData[ 6,0]
    v2 = unnormalizedData[ 7,0]
    v3 = unnormalizedData[ 8,0]
    c12 = unnormalizedData[ 9,0]
    c13 = unnormalizedData[ 10,0]
    c23 = unnormalizedData[ 11,0]  
    Bonus = unnormalizedData[ 12,0]  
    YetiBarrier = unnormalizedData[ 13,0]
    YetiCoupon = unnormalizedData[ 14,0]
    PhoenixBarrier = unnormalizedData[ 15,0]
    PhoenixCoupon = unnormalizedData[ 16,0]
    PDIBarrier = unnormalizedData[ 17,0]
    PDIGearing = unnormalizedData[ 18,0]
    PDIStrike = unnormalizedData[ 19,0]
    PDIType = unnormalizedData[ 20,0]
    maturity = unnormalizedData[ 21,0]
    NbDate = unnormalizedData[ 22,0]

    p10 ,p10a= predict_OptionCall_new(s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,Bonus,YetiBarrier,YetiCoupon,\
              PhoenixBarrier, PhoenixCoupon, PDIBarrier, PDIGearing, PDIStrike, PDIType,maturity,NbDate,0,model_Option,DataScalerX,DataScalerY,y_scaler_Option)  
    
    cov=numpy.asarray([[1,c12,c13],[c12,1,c23],[c13,c23,1]])
    nbTraj=100000
    NbDate=int(NbDate)
    normcallStored = dataset[icase,INPUT_GOAL]
    callStoredrecomputed = DataScalerY.inverse_transform(numpy.array([[normcallStored]]))
    callRecomputed=YetiPhoenixPriceMC(numpy.asarray([s1,s2,s3]),numpy.asarray([mu1,mu2,mu3]),numpy.asarray([v1,v2,v3]),cov,\
                          Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier, PhoenixCoupon,\
                          PDIBarrier, PDIGearing, PDIStrike, PDIType,maturity,NbDate,nbTraj)
    table = ListTable()
    table.append(['s1','s2','s3','mu1','mu2','mu3','v1','v2','v3','c12','c13','c23','Bonus','YetiBarrier','YetiCoupon',\
              'PhoenixBarrier', 'PhoenixCoupon', 'PDIBarrier', 'PDIGearing', 'PDIStrike', 'PDIType','maturity','NbDate'])
    
    table.append(["%.4f" % s1,"%.4f" % s2 ,"%.4f" % s3,\
                  "%.4f" % mu1,"%.4f" % mu2 ,"%.4f" % mu3,"%.4f" % v1,"%.4f" % v2 ,"%.4f" % v3,\
                  "%.4f" % c12,"%.4f" % c13 ,"%.4f" % c23, "%.4f" % Bonus, "%.4f" % YetiBarrier,"%.4f" % YetiCoupon ,\
                  "%.4f" % PhoenixBarrier,"%.4f" % PhoenixCoupon ,\
                  "%.4f" % PDIBarrier,"%.4f" % PDIGearing,"%.4f" % PDIStrike,"%.4f" % PDIType,"%.4f" % maturity,"%.4f" % NbDate])
    table.append([ 'prediction boost 0','call recomputed','callStored recomputed','normcall stored',"normCall b0"])
    table.append(["%.4f" % p10 , "%.4f" % callRecomputed,"%.4f" % callStoredrecomputed, "%.4f" % normcallStored, "%.4f" % p10a])
    return table


def predict_Option(data,learnedModels,BoostOrder):   
             
        model_Option,DataScalerX,DataScalerY,X_scaler,X_scaled,y_scaler_Option,y_scaled_Option, Y_ordre1_Option= learnedModels 
        normalizedData=DataScalerX.transform(data) 
        NoPricenormalisedData=normalizedData[:,0:23]
        normalizedOption= y_scaler_Option.inverse_transform(model_Option.predict(NoPricenormalisedData))
        Option=DataScalerY.inverse_transform(normalizedOption)
        return np.transpose(Option)[0]

def predict_Option_new(data,model_Option,DataScalerX,DataScalerY,y_scaler_Option):   
             
        normalizedData=DataScalerX.transform(data) 
        NoPricenormalisedData=normalizedData[:,0:23]
        normalizedOption= y_scaler_Option.inverse_transform(model_Option.predict(NoPricenormalisedData))
        Option=DataScalerY.inverse_transform(normalizedOption)
        return np.transpose(Option)[0]


class PricingInterface:
    def __init__(self,updatefunc,contupdate,ShowBarsFlag,model,DataScalerX,DataScalerY,y_scaler_Option):  
        self.ShowBarsFlag = ShowBarsFlag
        self.delta1 = 0
        self.delta2 = 0
        self.delta3 = 0
        self.vega1 = 0
        self.vega2 = 0
        self.vega3 = 0
        self.smin = -1
        self.smax = 1
        self.vmin = -1
        self.vmax = 1
        self.cmin = -1
        self.cmax = 1
        self.updateFuncIni=updatefunc
        self.ContinuousUpdating = contupdate
        self.ws1=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.1f'
            )
        self.ws2=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.ws3=widgets.FloatSlider(
            value=100,
            min=90,
            max=110.0,
            step=1,
            description='S3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )    
        self.wmu1=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wmu2=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wmu3=widgets.FloatSlider(
            value=0.01,
            min=0,
            max=0.05,
            step=0.001,
            description='mu3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv1=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v1',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv2=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v2',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wv3=widgets.FloatSlider(
            value=0.2,
            min=0.15,
            max=0.3,
            step=0.001,
            description='v3',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc12=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c12',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc13=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c13',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wc23=widgets.FloatSlider(
            value=0.8,
            min=0.4,
            max=1,
            step=0.001,
            description='c23',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.3f'
            )
        self.wBonus=widgets.FloatSlider(
            value=0,
            min=-2,
            max=2,
            step=0.1,
            description='Bonus',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wYetiBarrier=widgets.FloatSlider(
            value=100,
            min=90,
            max=110,
            step=1,
            description='YetiBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wYetiCoupon=widgets.FloatSlider(
            value=1,
            min=0,
            max=2,
            step=0.1,
            description='YetiCoupon',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPhoenixBarrier=widgets.FloatSlider(
            value=90,
            min=80,
            max=100,
            step=1,
            description='PhoenixBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPhoenixCoupon=widgets.FloatSlider(
            value=1,
            min=0.5,
            max=2,
            step=0.1,
            description='PhoenixCoupon',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIBarrier=widgets.FloatSlider(
            value=60,
            min=40,
            max=70,
            step=1,
            description='PDIBarrier',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIStrike=widgets.FloatSlider(
            value=60,
            min=40,
            max=70,
            step=1,
            description='PDIStrike',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )

        self.wPDIGearing=widgets.FloatSlider(
            value=-1,
            min=-5,
            max=5,
            step=1,
            description='PDIGearing',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wPDIType=widgets.FloatSlider(
            value=-1,
            min=-3,
            max=3,
            step=1,
            description='PDIType',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wMaturity=widgets.FloatSlider(
            value=3,
            min=1,
            max=7,
            step=0.1,
            description='Maturity',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='.2f'
            )
        self.wNbDate=widgets.IntSlider(
            value=12,
            min=4,
            max=24,
            step=1,
            description='NbDate',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='d'
            )
        self.wNbBoost=widgets.IntSlider(
            value=0,
            min=0,
            max=2,
            step=1,
            description='NbBoost',
            disabled=False,
            continuous_update=self.ContinuousUpdating,
            orientation='vertical',
            readout=True,
            readout_format='d'
            )
        self.model = model
        self.DataScalerX =DataScalerX
        self.DataScalerY = DataScalerY
        self.y_scaler_Option = y_scaler_Option
        self.res1 = 0
        self.result = 0


        self.wtotal=widgets.VBox([widgets.HBox([self.ws1,self.ws2,self.ws3,self.wmu1,self.wmu2,self.wmu3,self.wv1,self.wv2,self.wv3,self.wc12,self.wc13,self.wc23]),\
                     widgets.HBox([self.wBonus,self.wYetiBarrier,self.wYetiCoupon,self.wPhoenixBarrier,self.wPhoenixCoupon,self.wPDIBarrier,\
                     self.wPDIStrike,self.wPDIGearing,self.wPDIType,self.wMaturity,self.wNbDate,self.wNbBoost])])
        

        self.out = widgets.interactive_output(self.updateFunc1,\
            {'s1':self.ws1,'s2':self.ws2,'s3':self.ws3,'mu1':self.wmu1,'mu2':self.wmu2,'mu3':self.wmu3,'v1':self.wv1,'v2':self.wv2,'v3':self.wv3,\
            'c12':self.wc12,'c13':self.wc13,'c23':self.wc23,\
            'Bonus':self.wBonus,'YetiBarrier':self.wYetiBarrier,'YetiCoupon':self.wYetiCoupon,'PhoenixBarrier':self.wPhoenixBarrier,\
            'PhoenixCoupon':self.wPhoenixCoupon,'PDIBarrier':self.wPDIBarrier,'PDIGearing':self.wPDIGearing,\
            'PDIStrike':self.wPDIStrike,'PDIType':self.wPDIType,\
            'Maturity':self.wMaturity,'NbDate':self.wNbDate,'NbBoost':self.wNbBoost})
        sstep=(self.smax-self.smin)/100
        vstep=(self.vmax-self.vmin)/100

        self.fw1 = FloatProgress(value=self.result ,min=-10, max=10.0, step=0.1, description='price:',bar_style='info',orientation='vertical')
        self.fw2 = FloatProgress(value=self.delta1 ,min=self.smin, max=self.smax, step=sstep, description='delta1:',bar_style='info',orientation='vertical')
        self.fw3 = FloatProgress(value=self.delta2 ,min=self.smin, max=self.smax, step=sstep, description='delta2:',bar_style='info',orientation='vertical')
        self.fw4 = FloatProgress(value=self.delta3 ,min=self.smin, max=self.smax, step=sstep, description='delta3:',bar_style='info',orientation='vertical')
        self.fw5 = FloatProgress(value=self.vega1 ,min=self.vmin, max=self.vmax, step=vstep, description='vega1:',bar_style='info',orientation='vertical')
        self.fw6 = FloatProgress(value=self.vega2 ,min=self.vmin, max=self.vmax, step=vstep, description='vega2:',bar_style='info',orientation='vertical')
        self.fw7 = FloatProgress(value=self.vega3 ,min=self.vmin, max=self.vmax, step=vstep, description='vega3:',bar_style='info',orientation='vertical')
        
        if self.ShowBarsFlag:
            self.Wfinal = widgets.VBox([self.wtotal, self.out, widgets.HBox([self.fw1,self.fw2,self.fw3,self.fw4,self.fw5,self.fw6,self.fw7])])
        else :
            self.Wfinal = widgets.VBox([self.wtotal, self.out])
        

    def updateFunc1(self,s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,\
                     Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier,PhoenixCoupon,PDIBarrier,\
                     PDIStrike,PDIGearing,PDIType,Maturity,NbDate,NbBoost):
        
        a,smin,smax,vmin,vmax,cmin,cmax=self.updateFuncIni(\
                     s1,s2,s3,mu1,mu2,mu3,v1,v2,v3,c12,c13,c23,\
                     Bonus,YetiBarrier,YetiCoupon,PhoenixBarrier,PhoenixCoupon,PDIBarrier,\
                     PDIStrike,PDIGearing,PDIType,Maturity,NbDate,NbBoost,self.model,self.DataScalerX,self.DataScalerY,self.y_scaler_Option)
        
        self.result = a[0]
        self.delta1 = 100*(a[1]-a[0])/(s1/100)
        self.delta2 = 100*(a[2]-a[0])/(s2/100)
        self.delta3 = 100*(a[3]-a[0])/(s3/100)
        self.vega1 = 100*(a[4]-a[0])
        self.vega2 = 100*(a[5]-a[0])
        self.vega3 = 100*(a[6]-a[0])
       
        self.smin = smin
        self.smax = smax
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax

        if self.ShowBarsFlag:
            print([self.result ,self.delta1,self.delta2,self.delta3,self.vega1,self.vega2,self.vega3])
            print(a)
            print([smin,smax,vmin,vmax,cmin,cmax])
        

        
        
   
    def display(self):
            display(self.Wfinal)

def rightScale(vect):
    a=min(vect)
    b=max(vect)
    a=floor(a)
    b=-floor(-b)
    return (a,b)
            

def generateTrajectoire(random, sVect, muVect, sigmaVect, rhoMatrix, Tmax, nbDate):
    nbUnder = sVect.size
    Trajec = numpy.zeros(shape=(nbDate, nbUnder))
    deltaT = Tmax / (nbDate - 1)
    underRange = range(nbUnder)
    muVectTotal = numpy.zeros(shape=(nbUnder))
    for iUnder in underRange:
        Trajec[0, iUnder] = sVect[iUnder]
        muVectTotal[iUnder] = (muVect[iUnder] - sigmaVect[iUnder] * sigmaVect[iUnder] / 2) * deltaT
    covMatrix = numpy.zeros(shape=(nbUnder, nbUnder))
    for i1 in underRange:
        for i2 in underRange:
            covMatrix[i1, i2] = sigmaVect[i1] * sigmaVect[i2] * rhoMatrix[i1, i2] * deltaT
    U, S, V = numpy.linalg.svd(covMatrix)
    S1 = numpy.zeros(shape=(nbUnder))
    epsilon = 0.0000001
    for i in underRange:
        S1[i] = max(sqrt(S[i]), epsilon)
    choleskyMatrix = numpy.zeros(shape=(nbUnder, nbUnder))
    for i1 in underRange:
        for i2 in underRange:
            choleskyMatrix[i1, i2] = U[i1, i2] * S1[i2]
    for idate in range(1, nbDate):
        randomNoise = numpy.random.normal(size=nbUnder)
        randomNoise = choleskyMatrix.dot(randomNoise)
        for iUnder in underRange:
            lambdA = randomNoise[iUnder] + muVectTotal[iUnder]
            Trajec[idate, iUnder] = Trajec[idate - 1, iUnder] * exp(lambdA)
    return Trajec


def generateTrajectoire1D(randomR, s, mu, sigma, Tmax, nbDate):
    Trajec = numpy.zeros(nbDate)
    deltaT = Tmax / (nbDate - 1)
    muVectTotal = 0
    Trajec[0] = s
    muVectTotal = (mu - sigma * sigma / 2) * deltaT
    for idate in range(1, nbDate):
        randomNoise = numpy.random.normal()
        randomNoise = sigma * sqrt(deltaT) * randomNoise + muVectTotal
        Trajec[idate] = Trajec[idate - 1] * exp(randomNoise)
    return Trajec


def CallVanilleMC(S, mu, sigma, T, strike, nbTraj):
    randomR = 42
    sumT = 0.0
    for iTraj in range(1, nbTraj):
        traj = generateTrajectoire1D(randomR, S, mu, sigma, T, 2)
        tirage = 0
        if (traj[1] > strike):
            tirage = traj[1] - strike
        sumT += tirage
    return sumT / nbTraj


def CallWorstMC(sVect, muVect, sigmaVect, rhoMatrix, Tmax, strike, nbTraj):
    randomR = 42
    sumT = 0.0
    nbUnder = sVect.size
    underRange = range(nbUnder)
    for iTraj in range(1, nbTraj):
        traj = generateTrajectoire(randomR, sVect, muVect, sigmaVect, rhoMatrix, Tmax, 2)
        underlying = traj[1, 0]
        for iUnder in underRange:
            underlying = min(underlying, traj[1, iUnder])
        tirage = 0
        if (underlying > strike):
            tirage = underlying - strike
        sumT += tirage
    return sumT / nbTraj

def YetiPhenixWorstPayoff(trajec,  Bonus, YetiBarrier, YetiCoupon, PhoenixBarrier,  PhoenixCoupon,\
            PDIbarrier,  PDIGearing,  PDIStrike,  PDItype):
        payoff = 0
        PDIFlag = 0
        nbunder = trajec.shape[1]
        nbdate = trajec.shape[0]      
        forwards = numpy.zeros(shape=(nbdate))
        for idate in range(1, nbdate):
            w = trajec[idate, 0]
            for iunder in range(0, nbunder):
                w = min(w, trajec[idate, iunder]) 
            forwards[idate] = w
        for idate in range(1, nbdate):
            if (forwards[idate] <= YetiBarrier):
                payoff += Bonus + YetiCoupon
            else:
                payoff += Bonus
            
            if (forwards[idate] >= PhoenixBarrier):
                payoff += PhoenixCoupon; break
            if (forwards[idate] <= PDIbarrier):
                PDIFlag = 1.0
        return payoff + PDIFlag * PDIGearing * max(0.0, PDItype * (forwards[nbdate - 1] - PDIStrike))
    

def YetiPhoenixPriceMC(sVect, muVect, sigmaVect, rhoMatrix, Bonus, YetiBarrier, YetiCoupon,PhoenixBarrier,PhoenixCoupon,\
             PDIbarrier,PDIGearing,PDIStrike,PDItype,Tmax,nbdate,nbTraj):
    randomR = 41
    sumT = 0.0
    nbUnder = sVect.size
    underRange = range(nbUnder)
    for iTraj in range(1, nbTraj):
        traj = generateTrajectoire(randomR, sVect, muVect, sigmaVect, rhoMatrix, Tmax, nbdate+1)
        tirage = YetiPhenixWorstPayoff(traj, Bonus, YetiBarrier, YetiCoupon,PhoenixBarrier, PhoenixCoupon, \
                                       PDIbarrier, PDIGearing, PDIStrike, PDItype)
        sumT += tirage
    return sumT / nbTraj 

