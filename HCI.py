import sys
import biosppy.signals.ecg as ecg
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy
import scipy.integrate as integrate
import statsmodels.api as sm
from scipy.signal import butter, filtfilt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.append("C:\\Users\\El-Wattaneya\\Downloads\\HCIP\\HCIP\\HCI.py")
# Assuming svm is your trained SVM model

# i,ii,iii,avr,avl,avf, v1,v2,v3,v4, v5,v6,vx,  vy,vz
# (0,1,  2,          5),/6, (7, 8, 9),(10,11,12),13,14
def loadSubjectSamples(path):
    df = pd.read_csv(path)
    subjectElectrodes = []
    ecgX = df['sample interval'].tolist()
    for i in df.columns:
        if i == 'i' or i == 'avf' or i == 'v1' or i == 'v4' or i == 'vz' or i == 'vy':
            ecgY = df[i].tolist()
            subjectElectrodes.append(ecgY)
    return ecgX, subjectElectrodes


def butter_Bandpass_filter(data, Low_Cutoff, High_Cutoff, SamplingRate, order):
    nyq = 0.5 * SamplingRate
    low = Low_Cutoff / nyq
    high = High_Cutoff / nyq
    b, a = butter(order, [low, high], btype='band', analog=False, fs=None)
    Filtered_Data = filtfilt(b, a, data)
    return Filtered_Data


def filterSignal(ecgY):
    filtered_signal = []
    for i in ecgY:
        tmp = butter_Bandpass_filter(i, Low_Cutoff=1, High_Cutoff=40, SamplingRate=1000, order=2)
        filtered_signal.append(tmp)
    return filtered_signal


def Convert_to_2D_array(signal):
    res = []
    for i in range(len(signal)):
        res.append(signal[i][0])
    return res


def SegmentHeartBeats(signal, ecgX, sampling_rate=1000):
    segmented_signals = []
    for electrode_signal in signal:
        # Process the ECG signal for each electrode
        ecg_analysis = ecg.ecg(signal=electrode_signal, sampling_rate=sampling_rate, show=False)
        rpeaks = ecg_analysis['rpeaks']
        heartbeats = ecg.extract_heartbeats(signal=electrode_signal, rpeaks=rpeaks, sampling_rate=sampling_rate)
        segmented_signals.append(heartbeats)
    return Convert_to_2D_array(segmented_signals)


def concatenate_beats(data):
    concatenated_data = []

    for electrode_data in data:
        num_samples = len(electrode_data)
        num_concatenated_samples = num_samples // 4
        concatenated_electrode = []

        for i in range(num_concatenated_samples):
            concatenated_sample = np.concatenate(electrode_data[i * 4:(i + 1) * 4], axis=0)
            concatenated_electrode.append(concatenated_sample)

        concatenated_data.append(concatenated_electrode)

    return concatenated_data


def AutoCorrelate(sig):
    res = []
    for i in range(len(sig)):
        elec = []
        for j in range(len(sig[i])):
            tmp = sm.tsa.acf(sig[i][j], nlags=len(sig[i][j]))
            elec.append(tmp)
        res.append(elec)
    '''for i in range(len(res)):
        plt.figure(figsize=(24, 10))
        plt.plot(res[i][0])
        plt.show()'''
    return DCT(res)


def DCT(sig, threshold=170):
    res = []
    for i in range(len(sig)):
        elec = []
        for j in range(len(sig[i])):
            tmp = scipy.fftpack.dct(sig[i][j], type=2)
            tmp = tmp[:threshold]
            elec.append(tmp)
        res.append(elec)
    return res


def Normalize(lst):
    mx = max(lst)
    mn = min(lst)
    norm = [(x - mn) / (mx - mn) for x in lst]
    return norm


def Normalize_Features(sig):
    res = []
    for i in range(len(sig)):
        elec = []
        for j in range(len(sig[i])):
            # Plot_Features((sig[))
            tmp = Normalize(sig[i][j])
            elec.append(tmp)
        res.append(elec)
    return res


def Wavelet(ecg_segments):
    all_features = []
    for electrode_data in ecg_segments:
        electrode_features = []
        for ecg_segment in electrode_data:
            segment_features = extract_wavelet_coefficients(ecg_segment)
            electrode_features.append(segment_features)
        all_features.append(electrode_features)
    return all_features


def extract_wavelet_coefficients(ecg_signal, sampling_rate=1000, wavelet='db4', level=5, freq_band=(1, 40)):
    nyquist_freq = sampling_rate / 2
    max_freq_band = min(nyquist_freq, freq_band[1])
    min_freq_band = max(freq_band[0], 1)
    max_level = int(np.floor(np.log2(nyquist_freq / min_freq_band))) + 1
    level = min(level, max_level)

    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)

    coeffs_band = []
    for level_coeffs in coeffs[1:]:
        level_freq = np.linspace(0, nyquist_freq, len(level_coeffs))
        indices_band = np.where((level_freq >= min_freq_band) & (level_freq <= max_freq_band))[0]
        coeffs_band.append(level_coeffs[indices_band])

    feature_vector = np.concatenate(coeffs_band).ravel()
    feature_vector = feature_vector[:90]

    return feature_vector


def Unify_Electrodes(sig):
    res = []
    for i in range(len(sig)):
        for j in range(len(sig[i])):
            res.append(sig[i][j])
    return res


def identify_subject(classifier, data, thres):
    pred = classifier.predict(data)
    # print(pred)
    z = 0
    o = 0
    t = 0
    tr = 0
    for i in range(len(pred)):
        if pred[i] == 0:
            z = z + 1
        if pred[i] == 1:
            o = o + 1
        if pred[i] == 2:
            t = t + 1
        if pred[i] == 3:
            tr = tr + 1
    if z > len(pred) * thres:
        return 1
    elif o > len(pred) * thres:
        return 2
    elif t > len(pred) * thres:
        return 3
    elif tr > len(pred) * thres:
        return 4
    else:
        return 5


def Plot_Features(sig):
    plt.figure(figsize=(24, 6))
    # plt.subplot(121)
    plt.plot(np.arange(0, len(sig)), sig)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def main():
    print("Executing main function in HCI module")

    sub1X, sub1Y = loadSubjectSamples('samples1.csv')
    sub2X, sub2Y = loadSubjectSamples('samples2.csv')
    sub3X, sub3Y = loadSubjectSamples('samples3.csv')
    sub4X, sub4Y = loadSubjectSamples('samples4.csv')
    # supposedly not Identified
    sub5X, sub5Y = loadSubjectSamples('samples5.csv')

    # 0,1,2,5,/6,(7,8,9),(10,11,12),13,14
    '''for i in range(15):
        
        plt.figure(figsize=(24, 10))
        plt.plot(sub2X[:3000], sub2Y[i][:3000])
        plt.show()
    '''

    filtered_signal_sub_1 = filterSignal(sub1Y)
    filtered_signal_sub_2 = filterSignal(sub2Y)
    filtered_signal_sub_3 = filterSignal(sub3Y)
    filtered_signal_sub_4 = filterSignal(sub4Y)

    filtered_signal_sub_5 = filterSignal(sub5Y)

    # Segmentation
    segmented_signal_sub_1 = SegmentHeartBeats(filtered_signal_sub_1, sub1X)
    segmented_signal_sub_1 = concatenate_beats(segmented_signal_sub_1)

    segmented_signal_sub_2 = SegmentHeartBeats(filtered_signal_sub_2, sub2X)
    segmented_signal_sub_2 = concatenate_beats(segmented_signal_sub_2)

    segmented_signal_sub_3 = SegmentHeartBeats(filtered_signal_sub_3, sub3X)
    segmented_signal_sub_3 = concatenate_beats(segmented_signal_sub_3)

    segmented_signal_sub_4 = SegmentHeartBeats(filtered_signal_sub_4, sub4X)
    segmented_signal_sub_4 = concatenate_beats(segmented_signal_sub_4)

    segmented_signal_sub_5 = SegmentHeartBeats(filtered_signal_sub_5, sub5X)
    segmented_signal_sub_5 = concatenate_beats(segmented_signal_sub_5)

    # Plot_Features(segmented_signal_sub_1[0][0])
    '''plt.figure(figsize=(24, 10))
    plt.plot(segmented_signal_sub_5[7][20])
    plt.show()
    '''
    # for i in range(len(segmented_signal_sub_5)): #count the number of segments (78 Segment)
    # print(len(segmented_signal_sub_5[i]))

    # Feature Extraction (First by AC/DCT)
    dct1 = AutoCorrelate(segmented_signal_sub_1)
    dct1 = Normalize_Features(dct1)
    # Plot_Features(dct1[2][0])

    dct2 = AutoCorrelate(segmented_signal_sub_2)
    dct2 = Normalize_Features(dct2)

    dct3 = AutoCorrelate(segmented_signal_sub_3)
    dct3 = Normalize_Features(dct3)

    dct4 = AutoCorrelate(segmented_signal_sub_4)
    dct4 = Normalize_Features(dct4)

    dct5 = AutoCorrelate(segmented_signal_sub_5)
    dct5 = Normalize_Features(dct5)

    # ----------------------------------------------------Wavelet Features
    wl1 = Wavelet(segmented_signal_sub_1)
    # wl1 = Normalize_Features(wl1)

    wl2 = Wavelet(segmented_signal_sub_2)
    # wl2 = Normalize_Features(wl2)

    wl3 = Wavelet(segmented_signal_sub_3)
    # wl3 = Normalize_Features(wl3)

    wl4 = Wavelet(segmented_signal_sub_4)
    # wl4 = Normalize_Features(wl4)

    wl5 = Wavelet(segmented_signal_sub_5)
    # wl5 = Normalize_Features(wl5)

    # Plot_Features(dct1[2][0])
    # Plot_Features(dct2)

    # The final shape of the arrays that will go to the classifiers are 1d array, arr[#electrode][#segment]
    sub1_Wavelet = Unify_Electrodes((wl1))
    sub2_Wavelet = Unify_Electrodes((wl2))
    sub3_Wavelet = Unify_Electrodes((wl3))
    sub4_Wavelet = Unify_Electrodes((wl4))
    sub5_Wavelet = Unify_Electrodes((wl5))
    print(len(sub1_Wavelet))
    print(len(sub2_Wavelet))
    print(len(sub3_Wavelet))
    print(len(sub4_Wavelet))
    print(len(sub5_Wavelet))

    sub1_dct = Unify_Electrodes((dct1))
    sub2_dct = Unify_Electrodes((dct2))
    sub3_dct = Unify_Electrodes((dct3))
    sub4_dct = Unify_Electrodes((dct4))
    sub5_dct = Unify_Electrodes((dct5))
    print(len(sub1_dct))
    print(len(sub2_dct))
    print(len(sub3_dct))
    print(len(sub4_dct))
    print(len(sub5_dct))

    '''sub1 = sub1[:872]
    sub2=sub2[:872]
    sub3=sub3[:872]
    sub4=sub4[:872]
    
    sub1= sub1[300:]
    sub2 = sub2[200:]
    sub4 = sub4[400:]
    
    
    print(len(sub1))
    print(len(sub2))
    print(len(sub3))
    print(len(sub4))
    '''
    sub1l = []
    sub2l = []
    sub3l = []
    sub4l = []
    sub5l = []

    # Label Encoder
    for i in range(len(sub1_dct)):
        sub1l.append(0)
    for i in range(len(sub2_dct)):
        sub2l.append(1)
    for i in range(len(sub3_dct)):
        sub3l.append(2)
    for i in range(len(sub4_dct)):
        sub4l.append(3)
    for i in range(len(sub5_dct)):
        sub5l.append(4)

        # Splitting dct data
    x1_traind, x1_testd, y1_traind, y1_testd = train_test_split(sub1_dct, sub1l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x2_traind, x2_testd, y2_traind, y2_testd = train_test_split(sub2_dct, sub2l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x3_traind, x3_testd, y3_traind, y3_testd = train_test_split(sub3_dct, sub3l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x4_traind, x4_testd, y4_traind, y4_testd = train_test_split(sub4_dct, sub4l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x5_traind, x5_testd, y5_traind, y5_testd = train_test_split(sub5_dct, sub5l, test_size=0.2, random_state=42,
                                                                shuffle=True)

    XTraind = x1_traind + x2_traind + x3_traind + x4_traind
    XTestd = x1_testd + x2_testd + x3_testd + x4_testd
    YTraind = y1_traind + y2_traind + y3_traind + y4_traind
    YTestd = y1_testd + y2_testd + y3_testd + y4_testd

    svm = SVC(random_state=42)
    svm.fit(XTraind, YTraind)
    predictions = svm.predict(XTestd)
    print("SVM test score BY DCT is ", metrics.accuracy_score(predictions, YTestd) * 100, "%")
    identify_subject(svm, x5_traind, 0.5)
    joblib.dump(svm, 'svmACDCT_model.pkl')

    # RANDOM FOREST
    rf = RandomForestClassifier(n_estimators=1000, max_depth=1000, random_state=42)
    rf.fit(XTraind, YTraind)
    predictions_rf = rf.predict(XTestd)
    print("Random Forest test score for DCT is ", metrics.accuracy_score(predictions_rf, YTestd) * 100, "%")
    identify_subject(rf, x5_traind, 0.5)
    joblib.dump(rf, 'RandomForest_ACDCT_model.pkl')

    # WAVELET
    x1_trainw, x1_testw, y1_trainw, y1_testw = train_test_split(sub1_Wavelet, sub1l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x2_trainw, x2_testw, y2_trainw, y2_testw = train_test_split(sub2_Wavelet, sub2l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x3_trainw, x3_testw, y3_trainw, y3_testw = train_test_split(sub3_Wavelet, sub3l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x4_trainw, x4_testw, y4_trainw, y4_testw = train_test_split(sub4_Wavelet, sub4l, test_size=0.2, random_state=42,
                                                                shuffle=True)
    x5_trainw, x5_testw, y5_trainw, y5_testw = train_test_split(sub5_Wavelet, sub5l, test_size=0.2, random_state=42,
                                                                shuffle=True)

    XTrainw = x1_trainw + x2_trainw + x3_trainw + x4_trainw
    XTestw = x1_testw + x2_testw + x3_testw + x4_testw
    YTrainw = y1_trainw + y2_trainw + y3_trainw + y4_trainw
    YTestw = y1_testw + y2_testw + y3_testw + y4_testw

    svm1 = SVC(random_state=42)
    svm1.fit(XTrainw, YTrainw)
    predictionsw = svm1.predict(XTestw)
    print("SVM test score by WAVELET is ", metrics.accuracy_score(predictionsw, YTestw) * 100, "%")
    identify_subject(svm1, x5_trainw, 0.5)
    joblib.dump(svm1, 'svm_wavelet_model.pkl')

    rfw = RandomForestClassifier(n_estimators=1000, max_depth=1000, random_state=42)
    rfw.fit(XTrainw, YTrainw)
    predictions_rw = rfw.predict(XTestw)
    print("Random Forest test score for Wavelet is ", metrics.accuracy_score(predictions_rw, YTestw) * 100, "%")
    identify_subject(rfw, x5_trainw, 0.5)
    joblib.dump(rfw, 'RandomForest_wavelet_model.pkl')


