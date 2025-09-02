#import
import numpy as np
from scipy.linalg import inv
from scipy.linalg import sqrtm
from scipy.signal import butter, lfilter


##function


#butterworth带通滤波
def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y

#对EEG滤波
def butter_bandpass_one_subject(data, subj, lowcut, highcut, fs, interval=None):
    print('Processing ', subj)

    # Create new key 'EEG_filtered' to store filtered EEG of each subject
    data[subj]['EEG_filtered'] = {}

    # Current raw EEG
    temp_raw_EEG = data[subj]['raw_EEG']

    if interval is not None:
        startband = np.arange(lowcut, highcut, step=interval)

        for start in startband:
            # This will be new key inside the EEG_filtered
            band = "{:02d}_{:02d}".format(start, start + interval)

            print('Filtering through {} Hz band'.format(band))
            # Bandpass filtering
            data[subj]['EEG_filtered'][band] = {}
            data[subj]['EEG_filtered'][band]['EEG_all'] = butter_bandpass_filter(temp_raw_EEG, start, start + interval,fs)

    # else:
    #     # This will be new key inside the EEG_filtered
    #     band = "{:02d}_{:02d}".format(lowcut, highcut)
    #
    #     data[subj]['EEG_filtered'][band]['EEG_all'] = butter_bandpass_filter(temp_raw_EEG, lowcut, highcut, fs)

#划分训练集和测试集数据
from sklearn.model_selection import train_test_split


def split_EEG_one_class(EEG_one_class, percent_train=0.8, random_state=None):
    '''
    split_EEG_one_class will receive EEG data of one class, with size of T x N x M, where
    T = number of trial
    N = number of electrodes
    M = sample number

    INPUT:
    EEG_data_one_class: the data of one class of EEG data
    percent_train: allocation percentage of training data, default is 0.8
    random_state: random seed for reproducibility, default is None

    OUTPUT:
    EEG_train: EEG data for training
    EEG_test: EEG data for test

    Both have type of np.array dimension of T x M x N
    '''

    # Number of all trials
    n = EEG_one_class.shape[0]

    # Create dummy indices for train_test_split
    indices = np.arange(n)

    # Use sklearn's train_test_split for proper randomization
    train_indices, test_indices = train_test_split(
        indices,
        train_size=percent_train,
        random_state=random_state,
        shuffle=True
    )

    EEG_train = EEG_one_class[train_indices]
    EEG_test = EEG_one_class[test_indices]

    return EEG_train, EEG_test

#计算协方差
def compute_cov(EEG_data):
    '''
    INPUT:
    EEG_data : EEG_data in shape T x N x S

    OUTPUT:
    avg_cov : covariance matrix of averaged over all trials
    '''
    cov = []
    for i in range(EEG_data.shape[0]):
        cov.append(EEG_data[i] @ EEG_data[i].T / np.trace(EEG_data[i] @ EEG_data[i].T))

    cov = np.mean(np.array(cov), 0)

    return cov

#求解特征值与特征矩阵
def decompose_cov(avg_cov):
    '''
    This function will decompose average covariance matrix of one class of each subject into
    eigenvalues denoted by lambda and eigenvector denoted by V
    Both will be in descending order

    Parameter:
    avgCov = the averaged covariance of one class

    Return:
    λ_dsc and V_dsc, i.e. eigenvalues and eigenvector in descending order

    '''
    λ, V = np.linalg.eig(avg_cov)
    idx_dsc = np.argsort(λ)[::-1]  # Find index in descending order
    # Sort eigenvalues and eigenvectors in descending order
    λ_dsc = λ[idx_dsc]
    V_dsc = V[:, idx_dsc]  # Sort eigenvectors descending order
    λ_dsc = np.diag(λ_dsc)  # Diagonalize λ_dsc

    return λ_dsc, V_dsc


# 求解白化矩阵
def white_matrix(λ_dsc, V_dsc):
    '''
    '''
    λ_dsc_sqr = sqrtm(inv(λ_dsc))
    P = (λ_dsc_sqr) @ (V_dsc.T)

    return P



#特征值分解目标矩阵并排序
def decompose_S(S_one_class, order='descending'):
    '''
    This function will decompose the S matrix of one class to get the eigen vector
    Both eigenvector will be the same but in opposite order

    i.e the highest eigenvector in S left will be equal to lowest eigenvector in S right matrix
    '''
    # Decompose S
    global idx
    λ, B = np.linalg.eig(S_one_class)

    # Sort eigenvalues either descending or ascending
    if order == 'ascending':
        idx = λ.argsort()  # Use this index to sort eigenvector smallest -> largest
    elif order == 'descending':
        idx = λ.argsort()[::-1]  # Use this index to sort eigenvector largest -> smallest
    else:
        print('Wrong order input')

    λ = λ[idx]
    B = B[:, idx]

    return B, λ




#计算空间滤波器滤波后结果Z
def compute_Z(W, E, m):
    '''
    Will compute the Z
    Z = W @ E,

    E is in the shape of N x M, N is number of electrodes, M is sample
    In application, E has nth trial, so there will be n numbers of Z

    Z, in each trial will have dimension of m x M,
    where m is the first and last m rows of W, corresponds to smallest and largest eigenvalues
    '''
    Z = []

    W = np.delete(W, np.s_[m:-m:], 0)

    for i in range(E.shape[0]):
        Z.append(W @ E[i])

    return np.array(Z)

#计算Z的方差作为特征向量
def feat_vector(Z):
    '''
    Will compute the feature vector of Z matrix

    INPUT:
    Z : projected EEG shape of T x N x S

    OUTPUT:
    feat : feature vector shape of T x m

    T = trial
    N = channel
    S = sample
    m = number of filter
    '''

    feat = []

    for i in range(Z.shape[0]):
        var = np.var(Z[i], ddof=1, axis=1)
        varsum = np.sum(var)

        feat.append(np.log10(var / varsum))

    return np.array(feat)