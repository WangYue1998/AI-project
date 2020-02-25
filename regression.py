import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

def subjectlist():
    labelfile = pd.read_csv('CIS-PD_Training_Data_IDs_Labels.csv')
    return np.unique(labelfile['subject_id'].values)

# removing static gravitational acceleration  low=0.5 hign=20, return linear acceleration
def lh_filter(seq, smp_rate, low, high):
    sp = np.fft.fft(seq)  # Compute the one-dimensional discrete Fourier Transform.
    freq = np.fft.fftfreq(seq.shape[-1], 1 / smp_rate)  # (signal size, timestep)
    sp[np.abs(freq) < low] = 0
    if high > 0:
        sp[np.abs(freq) > high] = 0
    rec_smp = np.fft.ifft(sp)
    return rec_smp.real


# read all subjects' label
def readlabelfile():
    labelfile = pd.read_csv('CIS-PD_Training_Data_IDs_Labels.csv')
    return labelfile['measurement_id'].values, \
           labelfile['on_off'].values, \
           labelfile['dyskinesia'].values, \
           labelfile['tremor'].values, \
           labelfile['subject_id'].values


# read only one subject's label
def readsubject(id):
    labelfile = pd.read_csv('CIS-PD_Training_Data_IDs_Labels.csv')
    df = labelfile.loc[labelfile['subject_id'] == id]

    return df['measurement_id'].values, \
           df['on_off'].values, \
           df['dyskinesia'].values, \
           df['tremor'].values

#read one measurementID's RAW acceleration
def readaccelemeter(measurementid):
    idfile = pd.read_csv("training_data/" + str(measurementid) + ".csv")
    lx = lh_filter(idfile['X'].values, 50, 0.5, 20)
    ly = lh_filter(idfile['Y'].values, 50, 0.5, 20)
    lz = lh_filter(idfile['Z'], 50, 0.5, 20)
    return lx, ly, lz, idfile['Timestamp'].values

def slidewindow(series,window,overlap):
    smp_rate = 0.02
    lap = int(window * overlap/smp_rate)   #25
    n = int(window/smp_rate)
    maxindex = series.size - 1
    wseries =[]
    for i in range(0,maxindex,lap):
        wseries.append(series[i:i+n])
    wseries.pop()
    return np.array(wseries)


def removestatic(x,y,z,t,threshold):
    indexlist=[]
    for i in range(len(x)):
        if max(abs(x[i])) < threshold :
            if (max(abs(y[i])) < threshold):
                if (max(abs(z[i])) < threshold):
                    indexlist.append(i)
    if len(indexlist)>0:
        new_x = np.delete(x, indexlist,0)
        new_y = np.delete(y, indexlist,0)
        new_z = np.delete(z, indexlist,0)
        new_t = np.delete(t, indexlist,0)
        return new_x, new_y,new_z,new_t
    else:
        return x,y,z,t

def generatedata(measurement_id):
    lx,ly,lz,t = readaccelemeter(measurement_id)
    x = slidewindow(lx,1,0.5)
    y = slidewindow(ly, 1, 0.5)
    z = slidewindow(lz, 1, 0.5)
    X,Y,Z,T = removestatic(x,y,z,t,0.01)
    return X,Y,Z,T

# movement intensesity for each measurementID
def movementi(x,y,z):
    milist=[]
    T = 1
    # print(x)
    for i in range(len(x)):
        mi= np.sqrt(np.square(x[i])+ np.square(y[i])+ np.square(z[i]))
        milist.append(mi)
    miarray = np.array(milist)
    AI = (1/T)*(sum(miarray))
    ma= np.square(miarray-AI)
    VI = (1/T)*(sum(ma))
    return AI,VI

def generatemi(X,Y,Z):
    ailist = []
    vilist = []
    for i in range(len(X)):
        aivalue, vivalue = movementi(X[i],Y[i],Z[i])
        ailist.append(aivalue)
        vilist.append(vivalue)

    aiarray = np.array(ailist)
    viarray = np.array(vilist)
    normai = (aiarray - np.mean(aiarray)) / np.std(aiarray)
    normvi = (viarray - np.mean(viarray)) / np.std(viarray)
    return normai, normvi

#normalized signal magnitude area  for each measurementID
def sma(x, y, z):
    T =1
    xmagnitude = np.absolute(x)
    ymagnitude = np.absolute(y)
    zmagnitude = np.absolute(z)
    smavalue = (1 / T) * (np.sum(xmagnitude) + np.sum(ymagnitude) + np.sum(zmagnitude))
    return smavalue

# SMA array for multiple measurementID
def generatesma(X,Y,Z):
    smalist = []
    for i in range(len(X)):
        smavalue = sma(X[i],Y[i],Z[i])
        smalist.append(smavalue)

    smarray = np.array(smalist)
    normsma = (smarray - np.mean(smarray)) / np.std(smarray)
    return normsma

# energy for each measurementID
def energy(x,y,z):
    T = 1
    fx = np.fft.fft(x)
    fy = np.fft.fft(y)
    fz = np.fft.fft(z)

    sx = np.square(fx)
    sy = np.square(fy)
    sz = np.square(fz)

    s = np.sum(np.sum(sx)+np.sum(sy)+np.sum(sz))
    return s/T

# energy array for multiple measurementID
def generatenergy(X,Y,Z):
    elist = []
    for i in range(len(X)):
        elist.append(energy(X[i],Y[i],Z[i]))

    earray = np.array(elist)
    norme = (earray - np.mean(earray)) / np.std(earray)
    return norme

def generate_feature_data(subjectid):
    measurement_id, on_off, dyskinesia, tremor = readsubject(subjectid)
    ai_mat = []
    vi_mat = []
    s_mat=[]
    e_mat=[]

    length = [] #all input arrays must have the same shape
    for m in measurement_id:
        X, Y, Z, T = generatedata(m)
        ai, vi = generatemi(X, Y, Z)
        sma = generatesma(X,Y,Z)
        energy= generatenergy(X,Y,Z)

        aifeature = list(ai)
        vifeature = list(vi)
        smafeature = list(sma)
        energyfeature = list(energy)

        length.append(len(aifeature))

        ai_mat.append(aifeature)
        vi_mat.append(vifeature)
        s_mat.append(smafeature)
        e_mat.append(energyfeature)


    l = min(length)
    ai_mat2=[]
    vi_mat2=[]
    s_mat2 = []
    e_mat2=[]
    for a in ai_mat:
        a = a[:l]
        ai_mat2.append(a)
    for v in vi_mat:
        v = v[:l]
        vi_mat2.append(v)
    for s in s_mat:
        s = s[:l]
        s_mat2.append(s)
    for e in e_mat:
        e = e[:l]
        e_mat2.append(v)

    AI_mat = np.stack(ai_mat2)
    VI_mat = np.stack(vi_mat2)
    SMA_mat = np.stack(s_mat2)
    EN_mat = np.stack(e_mat2)
    return AI_mat, VI_mat, SMA_mat, EN_mat

def generate_label_data(subjectid):
    measurement_id, on_off, dyskinesia, tremor = readsubject(subjectid)
    yon=[]
    yd = []
    yt = []
    for each in on_off:
        each = np.array(each)
        yon.append(each)
    for each in dyskinesia:
        each = np.array(each)
        yd.append(each)
    for each in tremor:
        each = np.array(each)
        yt.append(each)
    return yon,yd,yt

#Compute ridge regression using closed form
def ridge_regression(feature, target, lam):  # least square add regularization to control overfitting
    feature_dim = feature.shape[1]
    w = np.dot(np.linalg.inv(np.dot(np.transpose(feature),feature) + lam*np.eye(feature_dim)) ,  np.dot(np.transpose(feature),target))
    return w

def mean_squared_error(true_label, predicted_label):  #Root Mean Square Error
    mse = np.sqrt(np.sum((true_label - predicted_label)**2)/len(true_label))
    return mse

def rand_split_train_test(data, label, train_perc):
    if train_perc >= 1 or train_perc <= 0:
        raise Exception('train_perc should be between (0,1).')
    sample_size = data.shape[0]
    if sample_size < 2:
        raise Exception('Sample size should be larger than 1. ')

    num_train_sample = np.max([np.floor(sample_size * train_perc).astype(int), 1])
    data, label = shuffle(data, label)

    data_tr = data[:num_train_sample]
    data_te = data[num_train_sample:]

    label_tr = label[:num_train_sample]
    label_te = label[num_train_sample:]

    return data_tr, data_te, label_tr, label_te


def exp(feature_all, target_all,subjectid,featurename,targetname):
    different_value_lam = [1e-10,1e-5,1e-2,1e-1,1,10,100,1000]

    feature_train, feature_test, target_train, target_test = \
        rand_split_train_test(feature_all, target_all, train_perc=0.6)

    train_performance = []
    test_performance = []
    for lam in different_value_lam:
        reg_model = ridge_regression(feature_train, target_train, lam)
        train_performance += [mean_squared_error(target_train, np.dot(feature_train, reg_model))]
        test_performance += [mean_squared_error(target_test, np.dot(feature_test, reg_model))]


    plt.figure()
    train_plot, = plt.plot(np.log(different_value_lam), train_performance, linestyle='-', color='b',
                           label='Training Error')
    test_plot, = plt.plot(np.log(different_value_lam), test_performance, linestyle='-', color='r', label='Testing '
                                                                                                         'Error')
    plt.xlabel("Sample lam (log)")
    plt.ylabel("Error")
    plt.title(str(subjectid)+" "+str(targetname)+" performance: "+str(featurename))
    plt.legend(handles=[train_plot, test_plot])
    plt.show()




if __name__ == '__main__':
    # print(generate_rnd_data(1004))
    # X, Y, Z, T = generatedata("3444e818-0ee3-4a2b-953a-f4dbc43b5d13")
    # a = generatesma(X,Y,Z)
    # b= generatenergy(X,Y,Z)

    plt.interactive(False)
    # AI, VI, SMA, ENERGY = generate_feature_data(1004)
    # on_off, dyskinesia, tremor = generate_label_data(1004)
    # exp(SMA, on_off, 1004, "normalized signal area", "on_off")

    for subjectid in subjectlist():
        AI, VI, SMA, ENERGY = generate_feature_data(subjectid)
        on_off, dyskinesia, tremor = generate_label_data(subjectid)

        exp(AI, on_off,subjectid,"mean of movement intensity","on_off")
        exp(AI, dyskinesia, subjectid, "mean of movement intensity", "dyskinesia")
        exp(AI, tremor, subjectid, "mean of movement intensity", "tremor")

        exp(VI, on_off, subjectid, "variance of movement intensity", "on_off")
        exp(VI, dyskinesia, subjectid, "variance of movement intensity", "dyskinesia")
        exp(VI, tremor, subjectid, "variance of movement intensity", "tremor")

        exp(SMA, on_off, subjectid, "normalized signal area", "on_off")
        exp(SMA, dyskinesia, subjectid, "normalized signal area", "dyskinesia")
        exp(SMA, tremor, subjectid, "normalized signal area", "tremor")

        exp(ENERGY, on_off, subjectid, "ENERGY", "on_off")
        exp(ENERGY, dyskinesia, subjectid, "ENERGY", "dyskinesia")
        exp(ENERGY, tremor, subjectid, "ENERGY", "tremor")







