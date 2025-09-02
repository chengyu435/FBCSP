##编程：程宇  时间：20250902
##基于fbcsp.ipynb进行优化，删除冗余，优化结构，修复在线测试和跨被试训练中遇到的bug
##针对测试结果准确率虚高，按照交叉验证→最终训练→最终测试的流程，避免数据泄露，解决了相关问题
##将function的定义放在了FBCSP_utils.py中
##加入了标准化处理
##优化结果显示部分

#import

import os
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from FBCSP_utils import *

#paradigm setting
# Number of subject, n + 1 for iteration purpose (there are 9 subjects)
ns = 2
# Bandpass filtering all subject
lowcut=4
highcut=40
fs = 250
percent_train=0.5
#选择最佳的滤波器对数
# Select number of filter
m = 2
#时间窗选择
# Now take EEG data within [0.5 3.5] seconds after cue onset position of each class
start = 0.5
end = 3.5


#不建议修改的参数
# 左/右运动想象的mark编号
left_class_code = 769
right_class_code = 770

#原始数据
ori_data = dict()
#处理后数据
mod_data = dict()




##主程序

# Path manipulation if running in dev container
if os.getcwd() == '/':
    print("Running the notebook it inside dev container..")
    base_dir = os.environ["DOCKER_WORKDIR"]
else:
    print("Running the notebook locally..")
    base_dir = os.getcwd()

train_cv_mean = []
train_cv_std = []
test_acc_all = []
# 读取数据
# Iter over all data path then store them in sub0X variable
for i in range(1, ns):
    data_path = os.path.join(base_dir, 'datasets/A{:02d}T.npz'.format(i))
    subject = 'subject{:02d}'.format(i)

    # Load EEG data from datapath and store into subject variable then store into data dictionary
    ori_data[subject] = np.load(data_path)


    subj = 'subject0{}'.format(i)
    mod_data[subj] = {}
    mod_data[subj]['raw_EEG'] = ori_data[subj]['s']
    #去除raw_EEG中EOG的部分(22-25通道)
    mod_data[subj]['raw_EEG'] = np.delete(mod_data[subj]['raw_EEG'], np.s_[22:], 1)

    while mod_data[subj]['raw_EEG'].shape[0] != 22:
        mod_data[subj]['raw_EEG'] = mod_data[subj]['raw_EEG'].T

    print(mod_data[subj]['raw_EEG'].shape)
    #带通滤波
    butter_bandpass_one_subject(mod_data, subj, lowcut, highcut, fs, interval=4)

    #确定mark点
    mod_data[subj]['left_pos'] = ori_data[subj]['epos'][ori_data[subj]['etyp'] == left_class_code]
    mod_data[subj]['right_pos'] = ori_data[subj]['epos'][ori_data[subj]['etyp'] == right_class_code]

    #分段
    # Temporary variable of left and right pos
    temp_pos_left = mod_data[subj]['left_pos']
    temp_pos_right = mod_data[subj]['right_pos']

    mod_data[subj]['train'] = {}
    mod_data[subj]['test'] = {}

    feat_left_all = []
    feat_right_all = []
    mod_data[subj]['CSP'] = {}
    # 遍历band
    for band in mod_data[subj]['EEG_filtered'].keys():
        temp_EEG_all = mod_data[subj]['EEG_filtered'][band]['EEG_all']
        temp_EEG_left = []
        temp_EEG_right = []

        # LEFT
        for j in range(len(temp_pos_left)):
            #list.append
            temp_EEG_left.append(temp_EEG_all[:, temp_pos_left[j] + int(start * fs): temp_pos_left[j] + int(end * fs)])
        #再转化为numpy.array
        mod_data[subj]['EEG_filtered'][band]['EEG_left'] = np.array(temp_EEG_left)

        # RIGHT
        for j in range(len(temp_pos_right)):
            temp_EEG_right.append(
                temp_EEG_all[:, temp_pos_right[j] + int(start * fs): temp_pos_right[j] + int(end * fs)])
        mod_data[subj]['EEG_filtered'][band]['EEG_right'] = np.array(temp_EEG_right)


    #划分训练集和测试机

        # Temporary variable for left and right class of each band
        temp_EEG_left = mod_data[subj]['EEG_filtered'][band]['EEG_left']
        temp_EEG_right = mod_data[subj]['EEG_filtered'][band]['EEG_right']

        # Temporary variable to access each band
        temp_filt = mod_data[subj]['EEG_filtered'][band]

        temp_filt['EEG_left_train'], temp_filt['EEG_left_test'] = split_EEG_one_class(temp_EEG_left, percent_train)
        temp_filt['EEG_right_train'], temp_filt['EEG_right_test'] = split_EEG_one_class(temp_EEG_right,percent_train)



    #计算协方差

        # New key to store result
        temp_band = mod_data[subj]['CSP'][band] = {}

        # Compute left and right covariance
        # LEFT
        temp_band['cov_left'] = compute_cov(mod_data[subj]['EEG_filtered'][band]['EEG_left_train'])

        # RIGHT
        temp_band['cov_right'] = compute_cov(mod_data[subj]['EEG_filtered'][band]['EEG_right_train'])

        # Add covariance of left and right class as composite covariance
        temp_band['cov_comp'] = temp_band['cov_left'] + temp_band['cov_right']

        mod_data[subj]['CSP'][band]['whitening'] = {}

        temp_whitening = mod_data[subj]['CSP'][band]['whitening']


        # Decomposing composite covariance into eigenvector and eigenvalue
        temp_whitening['eigval'], temp_whitening['eigvec'] = decompose_cov(mod_data[subj]['CSP'][band]['cov_comp'])

        # White matrix
        temp_whitening['P'] = white_matrix(temp_whitening['eigval'], temp_whitening['eigvec'])

        mod_data[subj]['CSP'][band]['S_left'] = {}
        mod_data[subj]['CSP'][band]['S_right'] = {}
        # Where to access data
        temp_P = temp_whitening['P']
        Cl = mod_data[subj]['CSP'][band]['cov_left']
        Cr = mod_data[subj]['CSP'][band]['cov_right']

        # Where to store result
        temp_Sl = mod_data[subj]['CSP'][band]['S_left']
        temp_Sr = mod_data[subj]['CSP'][band]['S_right']
        # LEFT
        Sl = temp_P @ Cl @ temp_P.T
        temp_Sl['eigvec'], temp_Sl['eigval'] = decompose_S(Sl, 'descending')

        # RIGHT
        Sr = temp_P @ Cr @ temp_P.T
        temp_Sr['eigvec'], temp_Sr['eigval'] = decompose_S(Sr, 'ascending')



        temp_eigvec = temp_Sl['eigvec']
        temp_P = temp_whitening['P']

        mod_data[subj]['CSP'][band]['W'] = temp_eigvec.T @ temp_P

        mod_data[subj]['train'][band] = {}
        mod_data[subj]['test'][band] = {}
        temp_W = mod_data[subj]['CSP'][band]['W']
        temp_EEG_left = mod_data[subj]['EEG_filtered'][band]['EEG_left_train']
        temp_EEG_right = mod_data[subj]['EEG_filtered'][band]['EEG_right_train']

        # LEFT
        mod_data[subj]['train'][band]['Z_left'] = compute_Z(temp_W, temp_EEG_left, m)
        mod_data[subj]['train'][band]['feat_left'] = feat_vector(mod_data[subj]['train'][band]['Z_left'])

        left_label = np.zeros([len(mod_data[subj]['train'][band]['feat_left']), 1])

        # RIGHT
        mod_data[subj]['train'][band]['Z_right'] = compute_Z(temp_W, temp_EEG_right, m)
        mod_data[subj]['train'][band]['feat_right'] = feat_vector(mod_data[subj]['train'][band]['Z_right'])

        right_label = np.ones([len(mod_data[subj]['train'][band]['feat_right']), 1])

        left = np.c_[mod_data[subj]['train'][band]['feat_left'], left_label]
        right = np.c_[mod_data[subj]['train'][band]['feat_right'], right_label]

        mod_data[subj]['train'][band]['feat_train'] = np.vstack([left, right])

        np.random.shuffle(mod_data[subj]['train'][band]['feat_train'])
        feat_left = mod_data[subj]['train'][band]['feat_left']

        feat_left_all.append(feat_left)

        # Access RIGHT each band
        feat_right = mod_data[subj]['train'][band]['feat_right']

        feat_right_all.append(feat_right)
    # 收集所有band的训练特征
    feat_left_all = []
    feat_right_all = []

    for band in mod_data[subj]['EEG_filtered'].keys():
        feat_left_all.append(mod_data[subj]['train'][band]['feat_left'])
        feat_right_all.append(mod_data[subj]['train'][band]['feat_right'])

    # 一次性合并所有训练特征
    merge_left = np.hstack(feat_left_all)
    merge_right = np.hstack(feat_right_all)

    # 创建标签
    y_merged = np.hstack([np.zeros(merge_left.shape[0]), np.ones(merge_right.shape[0])])
    X_merged = np.vstack([merge_left, merge_right])



    # 在训练集上进行特征选择
    select = SelectKBest(mutual_info_classif).fit(X_merged, y_merged)
    X_temp_selected = X_merged[:, select.get_support()]


    # 数据标准化
    scaler = StandardScaler()
    X_temp_selected = scaler.fit_transform(X_temp_selected)



    model = SVC(gamma='scale')

    # 在训练集上进行交叉验证
    cv_scores = cross_val_score(model, X_temp_selected, y_merged, cv=5)

    # 存储每个fold的准确率（如果需要后续分析）
    cv_fold_scores = cv_scores * 100  # 转换为百分比
    train_cv_mean.append(cv_scores.mean() * 100)
    train_cv_std.append(cv_scores.std() * 100)

    # 训练最终模型
    model.fit(X_temp_selected, y_merged)


    # 处理测试数据
    feat_left_all_test = []
    feat_right_all_test = []

    for band in mod_data[subj]['EEG_filtered'].keys():
        temp_W = mod_data[subj]['CSP'][band]['W']
        temp_EEG_left = mod_data[subj]['EEG_filtered'][band]['EEG_left_test']
        temp_EEG_right = mod_data[subj]['EEG_filtered'][band]['EEG_right_test']

        mod_data[subj]['test'][band]['Z_left'] = compute_Z(temp_W, temp_EEG_left, m)
        mod_data[subj]['test'][band]['feat_left'] = feat_vector(mod_data[subj]['test'][band]['Z_left'])

        mod_data[subj]['test'][band]['Z_right'] = compute_Z(temp_W, temp_EEG_right, m)
        mod_data[subj]['test'][band]['feat_right'] = feat_vector(mod_data[subj]['test'][band]['Z_right'])

        feat_left_all_test.append(mod_data[subj]['test'][band]['feat_left'])
        feat_right_all_test.append(mod_data[subj]['test'][band]['feat_right'])

    # 合并测试特征
    merge_left_test = np.hstack(feat_left_all_test)
    merge_right_test = np.hstack(feat_right_all_test)

    # 创建测试集标签和数据
    X_test_merged = np.vstack([merge_left_test, merge_right_test])
    y_test_merged = np.hstack([np.zeros(merge_left_test.shape[0]), np.ones(merge_right_test.shape[0])])

    # 应用特征选择和标准化到测试集
    X_test_selected = X_test_merged[:, select.get_support()]
    X_test_selected = scaler.transform(X_test_selected)



    # 在真正的测试集上测试
    test_acc = model.score(X_test_selected, y_test_merged) * 100
    test_acc_all.append(test_acc)

    # 在循环内部打印每个subject的详细结果
    print(f"Subject {i }: "
          f"Train CV = {cv_scores.mean() * 100:.2f}±{cv_scores.std() * 100:.2f}%, "
          f"Test = {test_acc:.2f}%")

# 循环结束后打印汇总结果
print("\n" + "=" * 50)
print("最终测试结果汇总:")
print("=" * 50)
for i in range(ns - 1):
    print(f"Subject {i + 1}: Test Accuracy = {test_acc_all[i]:.2f}%")

mean_acc = np.mean(test_acc_all)
std_acc = np.std(test_acc_all)
print("-" * 50)
print(f"FBCSP算法平均测试准确率: {mean_acc:.2f}%")
print(f"FBCSP算法测试准确率标准差: {std_acc:.2f}%")
print("=" * 50)

# todo：画图,暂时还没想好怎么画


