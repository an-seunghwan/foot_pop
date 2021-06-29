# -*- coding: utf-8 -*-
#%%
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import kneighbors_graph
# from matplotlib import pyplot as plt
#%%
import numpy as np
import pandas as pd
import os

os.chdir("/Users/anseunghwan/Documents/GitHub/floating_pop")
data_directory = '/Users/anseunghwan/Documents/GitHub/floating_pop_data'
#%%
df_20 = pd.read_csv(data_directory + '/covid_all_data20ver2.csv', encoding='cp949')
df_district = pd.read_csv(data_directory + '/행정동좌표.csv', encoding='cp949')
# train, test 데이터 분할
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=528)

for train_idx, test_idx in split.split(df_20, df_20.loc[:,['행정동코드', 'corona']]):
    df_train = df_20.loc[train_idx]
    df_test = df_20.loc[test_idx]

temp_mean = df_train['평균온도'].mean()
temp_sd = df_train['평균온도'].std()

dust_mean = df_train['일평균미세먼지'].mean()
dust_sd = df_train['일평균미세먼지'].std()

squ_mean = df_train['동별면적'].mean()
squ_sd = df_train['동별면적'].std()

# 행정동 코드와 고유번호 태깅   
M = df_20.iloc[:, 1].unique().shape[0] # 행정동 개수 
admin_dict = {key:val for key, val in enumerate(df_20.iloc[:, 1].unique())}

# train, test 데이터에서 corona 비율 맞추기

def pop_prepro(df, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd):
    df_ = df.iloc[:,3:17]
    df_.columns = ['day', 'hol', 'time', 'avg_temp', 'avg_moi', 'month', 'dust', 'id_pop', 'corp', 'worker', 'square', 'corona', 'crn_count', 'y']

    # 요일 one-hot 코딩
    day_lst = df_.day.tolist()
    day_oh = tf.one_hot(day_lst, 7)

    # 시간 one-hot 코딩
    time_lst = df_.time.tolist()
    time_oh = tf.one_hot(time_lst, 24)

    # 평균 온도 정규화
    df_['avg_temp'] = (df_['avg_temp'] - temp_mean) / temp_sd
    
    # 습도(%)/100
    df_['avg_moi'] = df_['avg_moi']/100
    
    # 월 one-hot 코딩
    month_lst = df_.month.tolist()
    month_oh = tf.one_hot(month_lst, 12)
    
    # 미세먼지 정규화
    df_['dust'] = (df_['dust']- dust_mean) / dust_sd
    
    # 주민등록상 거주자 만단위로 변경
    df_['id_pop'] = df_['id_pop']/10000

    # 동별 사업처 수 천 단위로 변경
    df_['corp'] = df_['corp']/1000

    # 동별종사자 만단위로 변경
    df_['worker'] = df_['worker']/10000

    # 면적 정규화
    df_['square'] = (df_['square'] - squ_mean)/squ_sd

    # 유동인구 만단위로 변경
    df_['y'] = df_['y']/10000
    
    df_['crn_count'] = df_['crn_count']/10
    
    # commona predictor 행렬
    comm_mat_ = np.array(df_.loc[:, ['hol','avg_temp', 'avg_moi', 'dust', 'corona', 'crn_count']]) 

    # 행정동별 common predictor 저장
    comm_mat = np.concatenate([day_oh, time_oh, month_oh, comm_mat_], axis=1) # common predictor 
    comm_mat = comm_mat.astype(np.float32)
    comm_data = [comm_mat[df.iloc[:, 1] == admin_dict.get(i), :] for i in range(M)]

    # 행정동별 specific predictor 저장
    spec_mat = np.array(df_.loc[:, ['id_pop', 'corp', 'worker', 'square']], dtype=np.float32)
    spec_data = [spec_mat[df.iloc[:, 1] == admin_dict.get(i), :] for i in range(M)]

    # 행정동별 유동인구 저장
    y = np.array(df_.y, dtype=np.float32)
    y_ = [y[df.iloc[:, 1] == admin_dict.get(i)] for i in range(M)]

    return comm_data, spec_data, y_
#%% 
comm_data, spec_data, y = pop_prepro(df_train, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)
#%% Adjacency matrix 
coord = df_district.loc[:,['위도', '경도']].to_numpy()

adj_mat_ = kneighbors_graph(coord, 100, mode='connectivity', include_self=True)
adj_mat = adj_mat_.toarray()

adj_mat = tf.convert_to_tensor(adj_mat, dtype=tf.float32)
#%%
# adj_mat_ = tf.random.uniform(shape=[424, 424], maxval=2, dtype=tf.int32, seed=1)

# adj_mat = tf.linalg.set_diag(adj_mat_, np.ones(424))

# data 
x_input = spec_data
shared_x_input = comm_data
n = df_train.shape[0] 

# hidden layer output shape
d1 = 49
d2 = 30 
d3 = 20
d4 = 10

# loss function
def loss(y, y_pred):
    loss_ = 0
    for i in range(M):
        loss_ = loss_ + tf.math.reduce_sum(tf.math.square(tf.squeeze(y[i])-y_pred[i]))
        
    return loss_

# model structure
    
shared_dense1 = layers.Dense(d1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')
shared_dense2 = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu')

# geometric information mapping
w_loc = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu')

#
dense_layers1 = [layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out1 = [layers.Dropout(0.2, input_shape=(d2, )) for _ in range(M)]

dense_layers2 = [layers.Dense(d3, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out2 = [layers.Dropout(0.2, input_shape=(d3, )) for _ in range(M)]

dense_layers3 = [layers.Dense(d4, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out3 = [layers.Dropout(0.2, input_shape=(d4, )) for _ in range(M)]

output_layers = [layers.Dense(1) for _ in range(M)]

lr = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# optimization
for i in range(2000):
    with tf.GradientTape() as tape:
        shared_h = [shared_dense1(x) for x in shared_x_input]
        shared_h2 = [shared_dense2(x) for x in shared_h]
        wt_loc = w_loc(adj_mat)
        wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
        shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(shared_h2)]
        concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(x_input, shared_h2_geo)]
        hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
        do1 = [d(h) for d, h in zip(drop_out1, hs1)]
        hs2 = [d(h) for d, h in zip(dense_layers2, do1)]
        do2 = [d(h) for d, h in zip(drop_out2, hs2)]
        hs3 = [d(h) for d, h in zip(dense_layers3, do2)]
        do3 = [d(h) for d, h in zip(drop_out3, hs3)]
        output = [d(h) for d, h in zip(output_layers, do3)]
        tmp_loss = tf.math.divide(loss(output, y), tf.constant([n], tf.float32))
        trainable_weights = [shared_dense1.weights[0]] + [shared_dense1.weights[1]] + [shared_dense2.weights[0]] + [shared_dense2.weights[1]] + \
                                [w_loc.weights[0]] + [w_loc.weights[1]] + [dense_layers1[i].weights[0] for i in range(M)] + \
                                    [dense_layers1[i].weights[1] for i in range(M)] + [dense_layers2[i].weights[0] for i in range(M)]  +\
                                        [dense_layers2[i].weights[1] for i in range(M)] + [dense_layers3[i].weights[0] for i in range(M)] +\
                                            [dense_layers3[i].weights[1] for i in range(M)] + [output_layers[i].weights[0] for i in range(M)] + [output_layers[i].weights[1] for i in range(M)] 
        if (i+1) % 20 == 0:
            print(i+1,'iter loss:' ,tmp_loss.numpy()[0]) 
        if (i+1) % 500 == 0:
            lr *= (2/3)
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    grads = tape.gradient(tmp_loss, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))
    
#%%
comm_test, spec_test, y_test = pop_prepro(df_test, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

shared_test_input = comm_test
test_input = spec_test
n_test = df_test.shape[0]

shared_h = [shared_dense1(x) for x in shared_test_input]
shared_h2 = [shared_dense2(x) for x in shared_h]
wt_loc = w_loc(adj_mat)
wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(shared_h2)]
concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, shared_h2_geo )]
hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
hs2 = [d(h) for d, h in zip(dense_layers2, hs1)]
hs3 = [d(h) for d, h in zip(dense_layers3, hs2)]
output = [d(h) for d, h in zip(output_layers, hs3)]

print(tf.math.divide(loss(output, y_test), tf.constant([n], tf.float32))) 
df_test['tween'].mean() # 0.3881
np.sqrt(tf.math.divide(loss(output, y_test), tf.constant([n], tf.float32))) # 0.033


#%% 주말 
simul_data1 = df_train.drop_duplicates(['행정동코드', '시간']).copy()
simul_data1 = simul_data1.sort_values(by=['행정동코드', '시간'], axis=0)

simul_data1['평균온도'] = 21
simul_data1['요일'] = 7 # 토요일
simul_data1['평균상대습도'] = 60
simul_data1['일평균미세먼지'] = 40
simul_data1['월'] = 9
simul_corona = [0, 1, 1, 1]
simul_corona_num = [0, 20, 50, 100] 

for i, j in zip(simul_corona, simul_corona_num):
    simul_data1['corona'] = i
    simul_data1['covid_count'] = j
    
    comm_simul1, spec_simul1, _ = pop_prepro(simul_data1, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

    shared_test_input = comm_simul1
    test_input = spec_simul1
    n_test = simul_data1.shape[0]
    
    simul_shared_h = [shared_dense1(x) for x in shared_test_input]
    simul_shared_h2 = [shared_dense2(x) for x in simul_shared_h]
    wt_loc = w_loc(adj_mat)
    wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
    simul_shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(simul_shared_h2)]
    simul_concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, simul_shared_h2_geo)]
    simul_hs1 = [d(h) for d, h in zip(dense_layers1, simul_concat_h)]
    simul_hs2 = [d(h) for d, h in zip(dense_layers2, simul_hs1)]
    simul_hs3 = [d(h) for d, h in zip(dense_layers3, simul_hs2)]
    simul_output = [d(h) for d, h in zip(output_layers, simul_hs3)]

    simul_result = np.squeeze(np.array([x.numpy() for x in simul_output])).reshape((424*24)).tolist()
    result = {'H_SDNG_CD': [admin_dict.get(i) for i in range(424) for _ in range(24)], "시간": [ t for t in range(24)]*424, "유동인구":simul_result}

    result = pd.DataFrame(result)

    result.to_csv('result20s_weekend'+ str(i) + '_' + str(j) + '_v4.csv', index=False, encoding='utf-8')
#%% 평일
simul_data1['요일'] = 3

for i, j in zip(simul_corona, simul_corona_num):
    simul_data1['corona'] = i
    simul_data1['covid_count'] = j
    
    comm_simul1, spec_simul1, _ = pop_prepro(simul_data1, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

    shared_test_input = comm_simul1
    test_input = spec_simul1
    n_test = simul_data1.shape[0]
    
    simul_shared_h = [shared_dense1(x) for x in shared_test_input]
    simul_shared_h2 = [shared_dense2(x) for x in simul_shared_h]
    wt_loc = w_loc(adj_mat)
    wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
    simul_shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(simul_shared_h2)]
    simul_concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, simul_shared_h2_geo)]
    simul_hs1 = [d(h) for d, h in zip(dense_layers1, simul_concat_h)]
    simul_hs2 = [d(h) for d, h in zip(dense_layers2, simul_hs1)]
    simul_hs3 = [d(h) for d, h in zip(dense_layers3, simul_hs2)]
    simul_output = [d(h) for d, h in zip(output_layers, simul_hs3)]

    simul_result = np.squeeze(np.array([x.numpy() for x in simul_output])).reshape((424*24)).tolist()
    result = {'H_SDNG_CD': [admin_dict.get(i) for i in range(424) for _ in range(24)], "시간": [ t for t in range(24)]*424, "유동인구":simul_result}

    result = pd.DataFrame(result)

    result.to_csv('result20s_week'+ str(i) + '_' + str(j) + '_v4.csv', index=False, encoding='utf-8')
#%%
df_60 = pd.read_csv(data_directory + '/covid_all_data60ver2.csv', encoding='cp949')
df_district = pd.read_csv(data_directory + '/행정동좌표.csv', encoding='cp949')
# train, test 데이터 분할
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=528)

for train_idx, test_idx in split.split(df_60, df_60.loc[:,['행정동코드', 'corona']]):
    df_train = df_60.loc[train_idx]
    df_test = df_60.loc[test_idx]

temp_mean = df_train['평균온도'].mean()
temp_sd = df_train['평균온도'].std()

dust_mean = df_train['일평균미세먼지'].mean()
dust_sd = df_train['일평균미세먼지'].std()

squ_mean = df_train['동별면적'].mean()
squ_sd = df_train['동별면적'].std()

# 행정동 코드와 고유번호 태깅   
M = df_60.iloc[:, 1].unique().shape[0] # 행정동 개수 
admin_dict = {key:val for key, val in enumerate(df_60.iloc[:, 1].unique())}
#%%
comm_data, spec_data, y = pop_prepro(df_train, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)
#%% Adjacency matrix 
coord = df_district.loc[:,['위도', '경도']].to_numpy()

adj_mat_ = kneighbors_graph(coord, 100, mode='connectivity', include_self=True)
adj_mat = adj_mat_.toarray()
adj_mat = tf.convert_to_tensor(adj_mat, dtype=tf.float32)
#%%
# data 
x_input = spec_data
shared_x_input = comm_data
n = df_train.shape[0] 

# hidden layer output shape
d1 = 49
d2 = 30 
d3 = 20
d4 = 10

# loss function
def loss(y, y_pred):
    loss_ = 0
    for i in range(M):
        loss_ = loss_ + tf.math.reduce_sum(tf.math.square(tf.squeeze(y[i])-y_pred[i]))
        
    return loss_

# model structure
    
shared_dense1 = layers.Dense(d1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='relu')
shared_dense2 = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu')

# geometric information mapping
w_loc = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu')

#
dense_layers1 = [layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out1 = [layers.Dropout(0.2, input_shape=(d2, )) for _ in range(M)]

dense_layers2 = [layers.Dense(d3, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out2 = [layers.Dropout(0.2, input_shape=(d3, )) for _ in range(M)]

dense_layers3 = [layers.Dense(d4, kernel_regularizer=tf.keras.regularizers.l2(0.02), activation='relu') for _ in range(M)]
drop_out3 = [layers.Dropout(0.2, input_shape=(d4, )) for _ in range(M)]

output_layers = [layers.Dense(1) for _ in range(M)]

lr = 0.005
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
# optimization
for i in range(2000):
    with tf.GradientTape() as tape:
        shared_h = [shared_dense1(x) for x in shared_x_input]
        shared_h2 = [shared_dense2(x) for x in shared_h]
        wt_loc = w_loc(adj_mat)
        wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
        shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(shared_h2)]
        concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(x_input, shared_h2_geo)]
        hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
        do1 = [d(h) for d, h in zip(drop_out1, hs1)]
        hs2 = [d(h) for d, h in zip(dense_layers2, do1)]
        do2 = [d(h) for d, h in zip(drop_out2, hs2)]
        hs3 = [d(h) for d, h in zip(dense_layers3, do2)]
        do3 = [d(h) for d, h in zip(drop_out3, hs3)]
        output = [d(h) for d, h in zip(output_layers, do3)]
        tmp_loss = tf.math.divide(loss(output, y), tf.constant([n], tf.float32))
        trainable_weights = [shared_dense1.weights[0]] + [shared_dense1.weights[1]] + [shared_dense2.weights[0]] + [shared_dense2.weights[1]] + \
                                [w_loc.weights[0]] + [w_loc.weights[1]] + [dense_layers1[i].weights[0] for i in range(M)] + \
                                    [dense_layers1[i].weights[1] for i in range(M)] + [dense_layers2[i].weights[0] for i in range(M)]  +\
                                        [dense_layers2[i].weights[1] for i in range(M)] + [dense_layers3[i].weights[0] for i in range(M)] +\
                                            [dense_layers3[i].weights[1] for i in range(M)] + [output_layers[i].weights[0] for i in range(M)] + [output_layers[i].weights[1] for i in range(M)] 
        if (i+1) % 20 == 0:
            print(i+1,'iter loss:' ,tmp_loss.numpy()[0]) 
        if (i+1) % 500 == 0:
            lr *= (2/3)
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr)
    grads = tape.gradient(tmp_loss, trainable_weights)
    optimizer.apply_gradients(zip(grads, trainable_weights))    
#%%
comm_test, spec_test, y_test = pop_prepro(df_test, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

shared_test_input = comm_test
test_input = spec_test
n_test = df_test.shape[0]

shared_h = [shared_dense1(x) for x in shared_test_input]
shared_h2 = [shared_dense2(x) for x in shared_h]
wt_loc = w_loc(adj_mat)
wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(shared_h2)]
concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, shared_h2_geo )]
hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
hs2 = [d(h) for d, h in zip(dense_layers2, hs1)]
hs3 = [d(h) for d, h in zip(dense_layers3, hs2)]
output = [d(h) for d, h in zip(output_layers, hs3)]

print(tf.math.divide(loss(output, y_test), tf.constant([n], tf.float32))) 
df_test['older'].mean() # 0.5514
np.sqrt(tf.math.divide(loss(output, y_test), tf.constant([n], tf.float32))) #0.029
#%%
simul_data1 = df_train.drop_duplicates(['행정동코드', '시간']).copy()
simul_data1 = simul_data1.sort_values(by=['행정동코드', '시간'], axis=0)

simul_data1['평균온도'] = 21
simul_data1['요일'] = 7 # 토요일
simul_data1['평균상대습도'] = 60
simul_data1['일평균미세먼지'] = 40
simul_data1['월'] = 9
simul_corona = [0, 1, 1, 1]
simul_corona_num = [0, 20, 50, 100] 

for i, j in zip(simul_corona, simul_corona_num):
    simul_data1['corona'] = i
    simul_data1['covid_count'] = j
    
    comm_simul1, spec_simul1, _ = pop_prepro(simul_data1, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

    shared_test_input = comm_simul1
    test_input = spec_simul1
    n_test = simul_data1.shape[0]
    
    simul_shared_h = [shared_dense1(x) for x in shared_test_input]
    simul_shared_h2 = [shared_dense2(x) for x in simul_shared_h]
    wt_loc = w_loc(adj_mat)
    wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
    simul_shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(simul_shared_h2)]
    simul_concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, simul_shared_h2_geo)]
    simul_hs1 = [d(h) for d, h in zip(dense_layers1, simul_concat_h)]
    simul_hs2 = [d(h) for d, h in zip(dense_layers2, simul_hs1)]
    simul_hs3 = [d(h) for d, h in zip(dense_layers3, simul_hs2)]
    simul_output = [d(h) for d, h in zip(output_layers, simul_hs3)]

    simul_result = np.squeeze(np.array([x.numpy() for x in simul_output])).reshape((424*24)).tolist()
    result = {'H_SDNG_CD': [admin_dict.get(i) for i in range(424) for _ in range(24)], "시간": [ t for t in range(24)]*424, "유동인구":simul_result}

    result = pd.DataFrame(result)

    result.to_csv('result60s_weekend'+ str(i) + '_' + str(j) + '_v4.csv', index=False, encoding='utf-8')
#%% 평일
simul_data1['요일'] = 3

for i, j in zip(simul_corona, simul_corona_num):
    simul_data1['corona'] = i
    simul_data1['covid_count'] = j
    
    comm_simul1, spec_simul1, _ = pop_prepro(simul_data1, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

    shared_test_input = comm_simul1
    test_input = spec_simul1
    n_test = simul_data1.shape[0]
    
    simul_shared_h = [shared_dense1(x) for x in shared_test_input]
    simul_shared_h2 = [shared_dense2(x) for x in simul_shared_h]
    wt_loc = w_loc(adj_mat)
    wt_loc_ = tf.reshape(wt_loc, [M, 1, d2])
    simul_shared_h2_geo = [tf.math.multiply(z, wt_loc_[i]) for i, z in enumerate(simul_shared_h2)]
    simul_concat_h = [layers.Concatenate(axis=1)([x, sh_h]) for x, sh_h in zip(test_input, simul_shared_h2_geo)]
    simul_hs1 = [d(h) for d, h in zip(dense_layers1, simul_concat_h)]
    simul_hs2 = [d(h) for d, h in zip(dense_layers2, simul_hs1)]
    simul_hs3 = [d(h) for d, h in zip(dense_layers3, simul_hs2)]
    simul_output = [d(h) for d, h in zip(output_layers, simul_hs3)]

    simul_result = np.squeeze(np.array([x.numpy() for x in simul_output])).reshape((424*24)).tolist()
    result = {'H_SDNG_CD': [admin_dict.get(i) for i in range(424) for _ in range(24)], "시간": [ t for t in range(24)]*424, "유동인구":simul_result}

    result = pd.DataFrame(result)

    result.to_csv('result60s_week'+ str(i) + '_' + str(j) + '_v4.csv', index=False, encoding='utf-8')