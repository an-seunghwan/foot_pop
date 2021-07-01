#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import kneighbors_graph
# from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
#%% path
# os.chdir("/Users/anseunghwan/Documents/GitHub/floating_pop")
# data_directory = '/Users/anseunghwan/Documents/GitHub/floating_pop_data'
os.chdir(r"D:\foot_pop")
data_directory = r"D:\foot_pop_data"
#%% Load data
df = pd.read_csv(data_directory + '/covid_all_data수정.csv', encoding='cp949')
df_district = pd.read_csv(data_directory + '/행정동좌표.csv', encoding='cp949')
df.columns
df.head()
#%%
print('train, test 데이터 분할')
split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)

for train_idx, test_idx in split.split(df, df.loc[:,['행정동코드', 'corona']]):
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

temp_mean = df_train['평균온도'].mean()
temp_sd = df_train['평균온도'].std()

dust_mean = df_train['일평균미세먼지'].mean()
dust_sd = df_train['일평균미세먼지'].std()

squ_mean = df_train['동별면적'].mean()
squ_sd = df_train['동별면적'].std()

# 행정동 코드와 고유번호 태깅   
M = df.iloc[:, 1].unique().shape[0] # 행정동 개수 
admin_dict = {key:val for key, val in enumerate(df.iloc[:, 1].unique())}
#%% 전처리
def pop_prepro(df, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd):
    df_ = df.iloc[:,3:17]
    df_.columns = ['day', 'hol', 'time', 'avg_temp', 'avg_moi', 'month', 'dust', 'id_pop', 'corp', 'worker', 'square', 'y', 'corona', 'crn_count']

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

    # 유동인구 천단위로 변경
    df_['y'] = df_['y']/1000
    
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
    y = np.array(df_.y)
    y_ = [y[df.iloc[:, 1] == admin_dict.get(i)] for i in range(M)]

    return comm_data, spec_data, y_
#%% 
'''
comm_data: 공통 데이터
spec_data: 각 행정동별 데이터
y: 유동인구
'''
print('전처리')
comm_data, spec_data, y = pop_prepro(df_train, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

# '''log 변환'''
# y = [np.log(y[i] + 1e-8) for i in range(M)]
#%% Adjacency matrix 
coord = df_district.loc[:,['위도', '경도']].to_numpy()

adj_mat_ = kneighbors_graph(coord, 50, mode='connectivity', include_self=True)
adj_mat = adj_mat_.toarray()

adj_mat = tf.convert_to_tensor(adj_mat, dtype=tf.float32)
#%% x_input, layer shape
x_input = spec_data
shared_x_input = comm_data
n = df_train.shape[0] 

# hidden layer unit number
d1 = 40
d2 = 30 
d3 = 20
d4 = 10
#%%
def build_model(x_input, shared_x_input):
    input_layer = [layers.Input(x_input[0].shape[1]) for _ in range(M)]
    shared_input_layer = [layers.Input(shared_x_input[0].shape[1]) for _ in range(M)]

    shared_dense1 = layers.Dense(d1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish')
    shared_dense2 = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish')

    shared_h = [shared_dense2(shared_dense1(x)) for x in shared_input_layer]

    # adjacency matrix embedding
    w_loc_dense = layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish')
    w_loc = w_loc_dense(adj_mat)
    w_loc = tf.split(w_loc, num_or_size_splits=M, axis=0)

    shared_h_geo = [tf.math.multiply(h, w) for h, w in zip(shared_h, w_loc)]
    concat_h = [layers.Concatenate(axis=1)([x, h]) for x, h in zip(input_layer, shared_h_geo)]

    dense_layers1 = [layers.Dense(d2, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish') for _ in range(M)]
    drop_out1 = [layers.Dropout(0.1) for _ in range(M)]

    dense_layers2 = [layers.Dense(d3, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish') for _ in range(M)]
    drop_out2 = [layers.Dropout(0.1) for _ in range(M)]

    dense_layers3 = [layers.Dense(d4, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='swish') for _ in range(M)]
    drop_out3 = [layers.Dropout(0.1) for _ in range(M)]

    hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
    do1 = [d(h) for d, h in zip(drop_out1, hs1)]

    hs2 = [d(h) for d, h in zip(dense_layers2, do1)]
    do2 = [d(h) for d, h in zip(drop_out2, hs2)]

    hs3 = [d(h) for d, h in zip(dense_layers3, do2)]
    do3 = [d(h) for d, h in zip(drop_out3, hs3)]

    # hs1 = [d(h) for d, h in zip(dense_layers1, concat_h)]
    # hs2 = [d(h) for d, h in zip(dense_layers2, hs1)]
    # hs3 = [d(h) for d, h in zip(dense_layers3, hs2)]

    output_layers = [layers.Dense(1, activation='exponential') for _ in range(M)]

    output = [d(h) for d, h in zip(output_layers, do3)]
    # output = [d(h) for d, h in zip(output_layers, hs3)]

    model = K.Model(inputs=[input_layer, shared_input_layer], outputs=output)

    model.summary()
    
    return model
#%%
@tf.function
def loss_fun(y, y_pred):
    loss_ = 0
    for i in range(M):
        loss_ = loss_ + tf.math.reduce_mean(tf.math.square(tf.cast(tf.squeeze(y[i]), tf.float32) - tf.cast(y_pred[i], tf.float32)))
    return loss_ 
#%% 
print('Training model')

model = build_model(x_input, shared_x_input)

lr = 0.0005
optimizer = K.optimizers.RMSprop(learning_rate=lr)
epochs = 1000
batch_size = 1024
loss_history = []
for i in range(epochs):
    idx = np.random.choice(range(x_input[0].shape[0]), batch_size)
    x_batch = [x[idx] for x in x_input]
    shared_x_batch = [x[idx] for x in shared_x_input]
    y_batch = [y_[idx] for y_ in y]
    with tf.GradientTape() as tape:
        y_pred = model([x_batch, shared_x_batch])
        loss = loss_fun(y_batch, y_pred)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
    
    if (i+1) % 5 == 0:
        print(i+1,'iter loss:', loss.numpy()) 
    loss_history.append(loss.numpy())
#%% weights save   
date = datetime.today().strftime("%Y%m%d")
model.save_weights('./assets/weights_{}/weights'.format(date))
#%%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(loss_history)
ax.set_title('loss')
plt.savefig('./assets/loss.png')
# plt.show()
plt.close()
#%% load model
model = build_model(x_input, shared_x_input)
date = datetime.today().strftime("%Y%m%d")
model.load_weights('./assets/weights_{}/weights'.format(date))
#%% Test
comm_test, spec_test, y_test = pop_prepro(df_test, admin_dict, temp_mean, temp_sd, dust_mean, dust_sd, squ_mean, squ_sd)

shared_test_input = comm_test
test_input = spec_test
# n_test = df_test.shape[0]
# n_test = test_input[0].shape[0]

# '''log 변환'''
# y_test = [np.log(y_test[i] + 1e-8) for i in range(M)]

test_pred = model([test_input, shared_test_input])

print('test dataset loss:', (loss_fun(test_pred, y_test)).numpy())
#%%
i = 0
loss_M = [tf.math.reduce_mean(tf.math.square(tf.cast(tf.squeeze(test_pred[i]), tf.float32) - tf.cast(y_test[i], tf.float32))) for i in range(M)]
maxi = np.argmax(np.array(loss_M))
mini = np.argmin(np.array(loss_M))
#%% plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(test_pred[maxi], y_test[maxi])
ax.set_title('prediction (maximum loss)')
plt.savefig('./assets/pred_max.png')
# plt.show()
plt.close()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(test_pred[mini], y_test[mini])
ax.set_title('prediction (minimum loss)')
plt.savefig('./assets/pred_min.png')
# plt.show()
plt.close()
#%%