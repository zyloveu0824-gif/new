import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input
import numpy as np
import scipy.io as sio
import gym
import time
import random
import datetime
import os
import imageio
import glob
import tqdm
import json

# tf.random.set_seed(11)

# tf.keras.backend.set_floatx('float64')


# 获得圆形区域离散采样点
# 输入：UAV移动速率
# 输出：以UAV移动速率为半径的圆形区域内，离散点的个数，离散点的坐标（修正后）
def discrete_circle_sample_count(n):
    count = 0
    move_dict = {}
    for x in range(-n, n + 1):
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))
        for y in range(-y_l, y_l + 1):
            move_dict[count] = np.array([y, x])
            count += 1
    return (count), move_dict


def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.abs(error) <= clip_delta
    squared_loss = 0.5 * tf.square(error)
    quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
    return tf.math.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

# sensor actor网络
# 输入：input_dim_list=[buffstate_shape]，即sensor_buffer经过roi pooling
# 输出：computing_vec, collecting_vec
def sensor_actor(input_dim_list, cnn_kernel_size):
    # Input
    sensor_buffer_state = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为sensor_buffer_state
    
    # MLP前的数据预处理
    sensor_mlp = tf.transpose(sensor_buffer_state, perm=[0, 2, 1])
    sensor_mlp = layers.Dense(1, activation='relu')(sensor_mlp)
    
    # Transformer block
    reshape_sensor_mlp = layers.Reshape((sensor_mlp.shape[1], sensor_mlp.shape[2]))(sensor_mlp)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=sensor_mlp.shape[2])
    attn_output = transformer_layer(reshape_sensor_mlp, reshape_sensor_mlp)
    attn_output = tf.squeeze(attn_output, axis=-1)  # 确保维度符合期望输出
    
    computing_dev = layers.Dense(1, activation='sigmoid')(attn_output)  # 全连接层 神经元数量为1

    model = keras.Model(inputs=[sensor_buffer_state], outputs=[computing_dev])  # 构造模型
    return model

# agent actor网络
# 输入：input_dim_list=[state_map_shape, buffstate_shape, buffstate_shape, band_shape]，即state map,total_buffer,done_buffer,operation,bandwidth
# 输出：move_out,op_dist即move,operation
# agent actor net: inputs state map,pos,buffer,operation,bandwidth; outputs: move,operation

def agent_actor(input_dim_list, cnn_kernel_size, move_r):
    # Inputs
    state_map = keras.Input(shape=input_dim_list[0])
    total_buffer = keras.Input(shape=input_dim_list[1])
    done_buffer = keras.Input(shape=input_dim_list[2])
    bandwidth = keras.Input(shape=input_dim_list[3])
    
    # CNN for map
    cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(state_map)
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(cnn_map)
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)

    # Transformer block for map
    reshape_map = layers.Reshape((cnn_map.shape[1] * cnn_map.shape[2], cnn_map.shape[3]))(cnn_map)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=cnn_map.shape[3])
    attn_output = transformer_layer(reshape_map, reshape_map)
    attn_output = layers.Reshape((cnn_map.shape[1], cnn_map.shape[2], cnn_map.shape[3]))(attn_output)

    # Adjust output shape to match (None, 13, 13, 1)
    move_out = layers.Conv2D(1, (1, 1), activation='relu')(attn_output)  # 1x1 Conv to adjust channel dimensions

    # MLP for total and done buffer and bandwidth
    # Same as previous code...
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.transpose(total_mlp, perm=[0, 2, 1])
    exe_op = layers.Dense(input_dim_list[1][1], activation='softmax')(total_mlp)

    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.transpose(done_mlp, perm=[0, 2, 1])
    bandwidth_in = tf.expand_dims(bandwidth, axis=-1)
    bandwidth_in = layers.Dense(1, activation='relu')(bandwidth_in)
    done_mlp = layers.concatenate([done_mlp, bandwidth_in], axis=-1)
    off_op = layers.Dense(input_dim_list[2][1], activation='softmax')(done_mlp)

    op_dist = layers.concatenate([exe_op, off_op], axis=1)
    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, bandwidth], outputs=[move_out, op_dist])
    return model
# center actor网络
# 输入：input_dim_list = [buffer_list_shape, pos_list_shape, bandvec_shape]，即sensor_map, agent_map, bandwidth_vector
# 输出：bandwidth_vec
# center actor net: inputs sensor_map,agent_map,bandwidth_vector; outputs: bandwidth_vec
def center_actor(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])  # 输入层1 keras.Input用于实例化一个keras张量
    pos_list = keras.Input(shape=input_dim_list[1])  # 输入层2 keras.Input用于实例化一个keras张量

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)  # 全连接层1 输入为done_buffer_list; 输出shape（神经元数量）为1; 激活函数为relu
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # tf.squeeze()函数用于从张量shape中移除大小为1的维度
    # 此处buffer_state张量的shape为(input_dim_list[0],)

    # pos list
    pos = layers.Dense(2, activation='relu')(pos_list)  # 全连接层2 输入为pos_list; 输出shape（神经元数量）为2; 激活函数为relu

    bandwidth_out = layers.concatenate([buffer_state, pos], axis=-1)  # layers.concatenate()在指定的维度拼接数组
    # bandwidth_out = layers.AlphaDropout(0.2)(bandwidth_out)
    bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)  # 全连接层3 输入为buffer_state和pos拼接成的一个张量; 输出shape（神经元数量）为1; 激活函数为relu
    bandwidth_out = tf.squeeze(bandwidth_out, axis=-1)  # tf.squeeze()函数用于从张量shape中移除大小为1的维度
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    bandwidth_out = layers.Softmax()(bandwidth_out)  # softmax激活函数，输出为与bandwidth_out相同的softmaxed输出
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    # bandwidth_out = bandwidth_out / tf.reduce_sum(bandwidth_out, 1, keepdims=True)
    # bandwidth_out = bandwidth_out / tf.expand_dims(tf.reduce_sum(bandwidth_out, 1), axis=-1)

    model = keras.Model(inputs=[done_buffer_list, pos_list], outputs=bandwidth_out, name='center_actor_net')  # 创建模型
    return model

# sensor critic网络
# 输入：input_dim_list=[buffstate_shape, computing_dev_shape, collecting_dev_shape]
# 即[sensor_buffer经过roi pooling, computing_dev, collecting_dev]
# 输出：reward_out
def sensor_critic(input_dim_list, cnn_kernel_size):
    # Input
    sensor_buffer = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为sensor_buffer
    computing_dev = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为computing_dev

    # MLP前的数据预处理
    sensor_mlp = tf.transpose(sensor_buffer, perm=[0, 2, 1])
    sensor_mlp = layers.Dense(1, activation='relu')(sensor_mlp)
    
    # Transformer block
    reshape_sensor_mlp = layers.Reshape((sensor_mlp.shape[1], sensor_mlp.shape[2]))(sensor_mlp)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=sensor_mlp.shape[2])
    attn_output = transformer_layer(reshape_sensor_mlp, reshape_sensor_mlp)
    attn_output = tf.squeeze(attn_output, axis=-1)  # 确保维度符合期望输出
    
    # Combine with computing device input
    r_out = layers.concatenate([attn_output, computing_dev])  # layers.concatenate()在指定的维度拼接数组
    r_out = layers.Dense(1, activation='relu')(r_out)  # 全连接层 神经元数量为1

    model = keras.Model(inputs=[sensor_buffer, computing_dev], outputs=r_out, name='sensor_critic_net')  # 创建模型
    return model

# agent critic网络
# 输入：input_dim_list = [state_map_shape, buffstate_shape, buffstate_shape, movemap_shape, op_shape, band_shape]
# 输出：reward_out
def agent_critic(input_dim_list, cnn_kernel_size):
    # Inputs
    state_map = keras.Input(shape=input_dim_list[0])
    total_buffer = keras.Input(shape=input_dim_list[1])
    done_buffer = keras.Input(shape=input_dim_list[2])
    move = keras.Input(shape=input_dim_list[3])
    onehot_op = keras.Input(shape=input_dim_list[4])
    bandwidth = keras.Input(shape=input_dim_list[5])

    # CNN for map state
    map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(state_map)
    map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)
    map_cnn = layers.AlphaDropout(0.2)(map_cnn)

    # Transformer block for map state
    reshape_map_cnn = layers.Reshape((map_cnn.shape[1] * map_cnn.shape[2], map_cnn.shape[3]))(map_cnn)
    transformer_layer = layers.MultiHeadAttention(num_heads=2, key_dim=map_cnn.shape[3])
    attn_output_map = transformer_layer(reshape_map_cnn, reshape_map_cnn)
    attn_output_map = layers.Reshape((map_cnn.shape[1], map_cnn.shape[2], map_cnn.shape[3]))(attn_output_map)
    map_cnn = layers.Flatten()(attn_output_map)
    map_cnn = layers.Dense(2, activation='relu')(map_cnn)

    # MLPs for buffers and operations
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.squeeze(total_mlp, axis=-1)
    total_mlp = layers.Dense(2, activation='relu')(total_mlp)

    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.squeeze(done_mlp, axis=-1)
    done_mlp = layers.Dense(2, activation='relu')(done_mlp)

    band_mlp = layers.Dense(1, activation='relu')(bandwidth)

    move_mlp = layers.Flatten()(move)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)

    onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)
    onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)

    # Concatenate all features
    all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)

    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)
    return model

# def agent_critic(input_dim_list, cnn_kernel_size):
#     # Input
#     state_map = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为state_map_shape
#     # position = keras.Input(shape=input_dim_list[1])
#     total_buffer = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为buffstate_shape
#     done_buffer = keras.Input(shape=input_dim_list[2])  # 输入层3 shape为buffstate_shape
#     move = keras.Input(shape=input_dim_list[3])  # 输入层4 shape为movemap_shape
#     onehot_op = keras.Input(shape=input_dim_list[4])  # 输入层5 shape为op_shape
#     bandwidth = keras.Input(shape=input_dim_list[5])  # 输入层6 shape为band_shape

#     # map CNN
#     # merge last dim
#     map_cnn = layers.Dense(1, activation='relu')(state_map)  # 全连接层1 神经元数量为1
#     map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)  # 卷积层1 （卷积核个数，卷积核大小，激活函数，补零策略）
#     map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)  # 平均池化层1 （池化核大小）
#     map_cnn = layers.AlphaDropout(0.2)(map_cnn)  # 保持原始均值和方差的 Dropout
#     # map_cnn = layers.Conv2D(input_dim_list[0][2], kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
#     # map_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(map_cnn)
#     # map_cnn = layers.Dropout(0.2)(map_cnn)
#     map_cnn = layers.Flatten()(map_cnn)  # 将数据压成一维数据
#     map_cnn = layers.Dense(2, activation='relu')(map_cnn)  # 全连接层2 神经元数量为2

#     # mlp
#     # pos_mlp = layers.Dense(1, activation='relu')(position)
#     band_mlp = layers.Dense(1, activation='relu')(bandwidth)  # 全连接层3 神经元数量为1
#     total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置）
#     total_mlp = layers.Dense(1, activation='relu')(total_mlp)  # 全连接层4 神经元数量为1
#     total_mlp = tf.squeeze(total_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
#     total_mlp = layers.Dense(2, activation='relu')(total_mlp)  # 全连接层5 神经元数量为 2
#     done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置）
#     done_mlp = layers.Dense(1, activation='relu')(done_mlp)  # 全连接层6 神经元数量为1
#     done_mlp = tf.squeeze(done_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
#     done_mlp = layers.Dense(2, activation='relu')(done_mlp)  # 全连接层7 神经元数量为2

#     move_mlp = layers.Flatten()(move)  # 将数据压成一维数据
#     move_mlp = layers.Dense(1, activation='relu')(move_mlp)  # 全连接层8 神经元数量为1
#     onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)  # 全连接层9 神经元数量为1
#     onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)  # 从张量shape中移除大小为1的维度

#     all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)  # 数组拼接
#     reward_out = layers.Dense(1, activation='relu')(all_mlp)  # 全连接层10 神经元数量为1

#     model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)  # 创建模型
#     # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
#     return model
# center critic 网络
# 输入：input_dim_list = [buffer_list_shape, pos_list_shape, bandvec_shape]，即sensor_map, agent_map, bandwidth_vector
# 输出：bandwidth_vec
# center actor net: inputs sensor_map,agent_map,bandwidth_vector; outputs: bandwidth_vec
def center_critic(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为buffer_list_shape
    pos_list = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为pos_list_shape
    bandwidth_vec = keras.Input(shape=input_dim_list[2])  # 输入层3 shape为bandvec_shape

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)  # 全连接层1 输出维度（神经元数量）为1
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # 从张量shape中移除维度为1的维度
    buffer_state = layers.Dense(1, activation='relu')(buffer_state)  # 全连接层2 输出维度（神经元数量）为1
    buffer_state = tf.squeeze(buffer_state, axis=-1)  # 从张量shape中移除维度为1的维度
    # 以上四行代码是为了压缩shape [x,y,z]->[x,]

    # pos list
    pos = layers.Dense(1, activation='relu')(pos_list)  # 全连接层3 输出维度（神经元数量）为1
    pos = tf.squeeze(pos, axis=-1)  # 从张量shape中移除维度为1的维度
    # 压缩shape [x,y]->[x,]

    # bandvec
    # band_in = layers.Dense(2, activation='relu')(bandwidth_vec)

    r_out = layers.concatenate([buffer_state, pos, bandwidth_vec])  # layers.concatenate()在指定的维度拼接数组
    # r_out = layers.AlphaDropout(0.2)(r_out)
    r_out = layers.Dense(1, activation='relu')(r_out)  # 全连接层4 输出维度（神经元数量）为1
    model = keras.Model(inputs=[done_buffer_list, pos_list, bandwidth_vec], outputs=r_out, name='center_critic_net')  # 创建模型
    return model


# 更新target网络权重
def update_target_net(model, target, tau=0.8):
    weights = model.get_weights()  # 获取model模型的全部参数（一个列表数组，第一层w，第一层b，第二层w，...）
    target_weights = target.get_weights()  # 获取target模型的全部参数
    for i in range(len(target_weights)):  # 将target模型的tau%设置为新的权重 set tau% of target model to be new weights
        target_weights[i] = weights[i] * (1 - tau) + target_weights[i] * tau
    target.set_weights(target_weights)

# def calculate_gamma_k(grads_t_plus_1, grads_t, weights_t_plus_1, weights_t, mu):

#     # 计算梯度差 
#     grad_diff = grads_t_plus_1 + mu * (weights_t_plus_1 - weights_t)
#     # 计算梯度范数 
#     grad_norm_t = np.linalg.norm(grads_t)
#     # 计算 gamma_k
#     gamma_k = np.linalg.norm(grad_diff) / grad_norm_t
#     return gamma_k
# def calculate_I_k(global_grad, old_grad, gamma_k, psi):
#     # 计算内积 
#     inner_product = np.dot(global_grad, old_grad)
#     # 计算范数平方项 
#     norm_squared = -psi * gamma_k * np.linalg.norm(global_grad)**2
#     I_k = inner_product + norm_squared
#     return I_k
# def calculate_P(I_values):
#     # 计算所有 I_t^k 值的总和
#     I_total = np.sum([np.abs(I) for I in I_values])
    
#     # 计算每个网络的 P_{lbk}^t
#     P_values = [np.abs(I) / I_total for I in I_values]
     
#     return P_values
# def merge_fl_old(nets, omega=0.5):
#     # 对所有UAV的actor/critic网络进行循环
#     for agent_no in range(len(nets)):
#         target_params = nets[agent_no].get_weights()  # 获取当前UAV模型的全部参数
#         other_params = []
#         for i, net in enumerate(nets):
#             if i == agent_no:
#                 continue
#             other_params.append(net.get_weights())  # 获取其他UAV模型的全部参数列表
#         # 对当前UAV模型中的“所有层的参数”进行循环
#         for i in range(len(target_params)):
#             others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)  # 其他UAV（边缘智能体）的参数 PS：求均值
#             target_params[i] = omega * target_params[i] + others * (1 - omega)  # 当前UAV保留以权值（系数）为ω的参数，并混合其他UAV的参数
#             # print([others.shape, target_params[i].shape])
#         nets[agent_no].set_weights(target_params)  # 将新参数装载入模型
# # def sample_devices(P_values, K):
# #     if not isinstance(P_values, list) or len(P_values) == 0:
# #         raise ValueError("P_values must be a non-empty list")
# #     P_values = np.nan_to_num(P_values, nan=0.0)
# #     total = np.sum(P_values)

# #     if total == 0:
# #         # 如果总概率和为 0，则平均分配概率
# #         P_values = [1.0 / len(P_values)] * len(P_values)
# #     else:
# #         # 正规化概率，确保概率之和为 1
# #         P_values = P_values / total

# #     # 根据 P 值作为权重进行采样
# #     C_t = np.random.choice(len(P_values), size=K, replace=False, p=P_values)
# #     print('-----------------p_value--------------------------')
# #     print(P_values)
# #     print('--------------------C_t-----------------------')
# #     print(C_t)
# #     return C_t

# def sample_devices(P_values, K):
#     print('-----------------p_value--------------------------')
#     print(P_values)

#     # 根据 P 值作为权重进行采样
#     C_t = np.random.choice(len(P_values), size=K, replace=False, p=P_values)
#     print('--------------------C_t-----------------------')
#     print(C_t)
#     return C_t
# def compute_delta_weights(new_weights_list, global_weights):
#     delta_weights = [new_w - global_weights for new_w in new_weights_list]
#     return delta_weights

# def update_global_weights(C_t, I_values, new_weights, global_weights):

#     sum_I_values = np.sum([np.abs(I_values[k]) for k in C_t])
 
#     weighted_updates = np.zeros_like(global_weights)
    
#     # 为选中的每个设备计算加权更新
#     for k in C_t:
#         delta_w = new_weights[k] - global_weights
#         weighted_update = (I_values[k] / sum_I_values) * delta_w
#         weighted_updates += weighted_update

#     updated_global_weights = global_weights + weighted_updates
    
#     return updated_global_weights
# def reshape_weights(flat_weights, shapes):
#     reshaped_weights = []
#     index = 0
#     for shape in shapes:
#         size = np.product(shape)
#         weight = flat_weights[index:index + size].reshape(shape)
#         reshaped_weights.append(weight)
#         index += size
#     return reshaped_weights
# def merge_fl(nets, gard, old_gard, weights, old_weights, psi, knum, mu=0.1, omega=0.5):
#     epsilon = 1e-8  
#     all_gards = []
#     all_old_gards = []
#     all_weights = []
#     all_old_weights = []
#     gammas = []
#     I_values = []

#     accumulated_old_gards = None

#     for no in range(len(nets)):

#         current_gards = [grad.flatten() + epsilon for grad in gard[no]]
#         all_gards.append(np.concatenate(current_gards))

#         current_old_gards = [grad.flatten() + epsilon for grad in old_gard[no]]
#         flattened_old_gards = np.concatenate(current_old_gards)
#         all_old_gards.append(flattened_old_gards)
#         if accumulated_old_gards is None:
#             accumulated_old_gards = flattened_old_gards
#         else:
#             accumulated_old_gards += flattened_old_gards

#         current_weights = [weight.flatten() + epsilon for weight in weights[no]]
#         all_weights.append(np.concatenate(current_weights))

#         current_old_weights = [weight.flatten() + epsilon for weight in old_weights[no]]
#         all_old_weights.append(np.concatenate(current_old_weights))

#     for no in range(len(nets)):
#         grad_diff = all_gards[no] + mu * (all_weights[no] - all_old_weights[no])
#         grad_norm_t = np.linalg.norm(all_old_gards[no])
#         gamma_k = np.linalg.norm(grad_diff) / (grad_norm_t + epsilon)
#         gammas.append(gamma_k)

#     global_grad = np.zeros_like(all_old_gards[0])
#     global_weights  = np.zeros_like(all_old_weights[0])
#     for no in range(len(nets)):
#         global_grad += all_gards[no]
#         global_weights += all_weights[no]
        
#     global_grad = global_grad / len(nets)
#     global_weights = global_weights / len(nets)

#     for no in range(len(nets)):
#         inner_product = np.dot(global_grad, all_old_gards[no])
#         norm_squared = -psi * gammas[no] * np.linalg.norm(global_grad)**2
#         I_k = inner_product + norm_squared
#         I_values.append(I_k)

#     I_total = np.sum([np.abs(I) for I in I_values])

#     P_values = [np.abs(I) / I_total for I in I_values]

#     C_t = sample_devices(P_values, knum)

#     new_weights = update_global_weights(C_t, I_values, all_weights, global_weights)
#     # for k in C_t:
#     #     expected_shapes = [w.shape for w in nets[0].get_weights()]
#     #     reshaped_weights = reshape_weights(new_weights, expected_shapes)
#     #     nets[k].set_weights(reshaped_weights)
#     for agent_no in range(len(nets)):
#         expected_shapes = [w.shape for w in nets[no].get_weights()]
#         reshaped_weights = reshape_weights(new_weights, expected_shapes)
#         nets[agent_no].set_weights(reshaped_weights)

# def merge_fl(nets, gard, old_gard, weights, old_weights, psi, knum, mu=0.1, omega=0.5):
#     # Initialize lists to hold the flattened values for all networks
#     all_gards = []
#     all_old_gards = []
#     all_weights = []
#     all_old_weights = []
#     gammas = []
#     I_values = []

#     accumulated_old_gards = None

#     for no in range(len(nets)):

#         current_gards = [grad.flatten() for grad in gard[no]]
#         all_gards.append(np.concatenate(current_gards))

#         current_old_gards = [grad.flatten() for grad in old_gard[no]]
#         flattened_old_gards = np.concatenate(current_old_gards)
#         all_old_gards.append(flattened_old_gards)
#         if accumulated_old_gards is None:
#             accumulated_old_gards = flattened_old_gards
#         else:
#             accumulated_old_gards += flattened_old_gards

#         current_weights = [weight.flatten() for weight in weights[no]]
#         all_weights.append(np.concatenate(current_weights))

#         current_old_weights = [weight.flatten() for weight in old_weights[no]]
#         all_old_weights.append(np.concatenate(current_old_weights))

#     for no in range(len(nets)):
#         grad_diff = all_gards[no] + mu * (all_weights[no] - all_old_weights[no])
#         grad_norm_t = np.linalg.norm(all_old_gards[no])
#         epsilon = 1e-8
#         gamma_k = np.linalg.norm(grad_diff) / (grad_norm_t + epsilon)
#         gammas.append(gamma_k)
#     global_grad = np.zeros_like(all_old_gards[0])
#     global_weights  = np.zeros_like(all_old_weights[0])
#     for no in range(len(nets)):
#         # global_grad += all_old_gards[no]
#         # global_weights += all_old_weights[no]
#         global_grad += all_gards[no]
#         global_weights += all_weights[no]
        
#     global_grad = global_grad/(len(nets))
#     global_weights = global_weights/(len(nets))
#     for no in range(len(nets)):
#         inner_product = np.dot(global_grad, all_old_gards[no])
#         norm_squared = -psi * gammas[no] * np.linalg.norm(global_grad)**2
#         I_k = inner_product + norm_squared
#         I_values.append(I_k)
#     I_total = np.sum([np.abs(I) for I in I_values])
#     # if(I_total == 0):
#     #     print('skip')
#     #     return nets
#     #     for agent_no in range(len(nets)):
#     #         target_params = nets[agent_no].get_weights()  # 获取当前UAV模型的全部参数
#     #         other_params = []
#     #         for i, net in enumerate(nets):
#     #             if i == agent_no:
#     #                 continue
#     #             other_params.append(net.get_weights())  # 获取其他UAV模型的全部参数列表
#     #         # 对当前UAV模型中的“所有层的参数”进行循环
#     #         for i in range(len(target_params)):
#     #             others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)  # 其他UAV（边缘智能体）的参数 PS：求均值
#     #             target_params[i] = omega * target_params[i] + others * (1 - omega)  # 当前UAV保留以权值（系数）为ω的参数，并混合其他UAV的参数
#     #             # print([others.shape, target_params[i].shape])
#     #         nets[agent_no].set_weights(target_params)  # 将新参数装载入模型
#     #     return nets
#     P_values = [np.abs(I) / I_total for I in I_values]
#     # delta_weights = compute_delta_weights(all_weights, global_weights)
#     C_t = sample_devices(P_values, knum)

#     new_weights = update_global_weights(C_t, I_values, all_weights, global_weights)
#     # for agent_no in range(len(nets)):
#     #     expected_shapes = [w.shape for w in nets[no].get_weights()]
#     #     reshaped_weights = reshape_weights(new_weights, expected_shapes)
#     #     nets[agent_no].set_weights(reshaped_weights)
#     for k in C_t:
#         expected_shapes = [w.shape for w in nets[0].get_weights()]
#         reshaped_weights = reshape_weights(new_weights, expected_shapes)
#         nets[k].set_weights(reshaped_weights)

#     return nets


# def fedsgd(nets, learning_rate=0.01):
#     """
#     将所有 UAV 的 actor/critic 网络的参数进行联邦平均，并使用 SGD（随机梯度下降）更新参数。
    
#     参数：
#     nets: 包含所有 UAV 的网络列表。
#     learning_rate: 学习率，用于 SGD 更新。
#     """
#     # 获取所有 UAV 网络的权重
#     global_params = [net.get_weights() for net in nets]
    
#     # 初始化全局参数为第一个 UAV 的参数
#     aggregated_params = [np.zeros_like(param) for param in global_params[0]]
    
#     # 累加所有 UAV 的参数
#     for params in global_params:
#         for i in range(len(params)):
#             aggregated_params[i] += params[i]
    
#     # 计算参数的均值
#     for i in range(len(aggregated_params)):
#         aggregated_params[i] /= len(nets)
    
#     # 使用 SGD 更新每个 UAV 的参数
#     for agent_no in range(len(nets)):
#         current_params = nets[agent_no].get_weights()
#         for i in range(len(current_params)):
#             current_params[i] = current_params[i] - learning_rate * (current_params[i] - aggregated_params[i])
#         nets[agent_no].set_weights(current_params)

# # 联邦更新
# def merge_fl(nets, omega=0.5):
#     # 对所有UAV的actor/critic网络进行循环
#     for agent_no in range(len(nets)):
#         target_params = nets[agent_no].get_weights()  # 获取当前UAV模型的全部参数
#         other_params = []
#         for i, net in enumerate(nets):
#             if i == agent_no:
#                 continue
#             other_params.append(net.get_weights())  # 获取其他UAV模型的全部参数列表
#         # 对当前UAV模型中的“所有层的参数”进行循环
#         for i in range(len(target_params)):
#             others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)  # 其他UAV（边缘智能体）的参数 PS：求均值
#             target_params[i] = omega * target_params[i] + others * (1 - omega)  # 当前UAV保留以权值（系数）为ω的参数，并混合其他UAV的参数
#             # print([others.shape, target_params[i].shape])
#         nets[agent_no].set_weights(target_params)  # 将新参数装载入模型
def calculate_gamma_k(grads_t_plus_1, grads_t, weights_t_plus_1, weights_t, mu):

    # 计算梯度差 
    grad_diff = grads_t_plus_1 + mu * (weights_t_plus_1 - weights_t)
    # 计算梯度范数 
    grad_norm_t = np.linalg.norm(grads_t)
    # 计算 gamma_k
    gamma_k = np.linalg.norm(grad_diff) / grad_norm_t
    return gamma_k
def calculate_I_k(global_grad, old_grad, gamma_k, psi):
    # 计算内积 
    inner_product = np.dot(global_grad, old_grad)
    # 计算范数平方项 
    norm_squared = -psi * gamma_k * np.linalg.norm(global_grad)**2
    I_k = inner_product + norm_squared
    return I_k
def calculate_P(I_values):
    # 计算所有 I_t^k 值的总和
    I_total = np.sum([np.abs(I) for I in I_values])
    
    # 计算每个网络的 P_{lbk}^t
    P_values = [np.abs(I) / I_total for I in I_values]
     
    return P_values
# def merge_fl_old(nets, omega=0.5):
#     # 对所有UAV的actor/critic网络进行循环
#     for agent_no in range(len(nets)):
#         target_params = nets[agent_no].get_weights()  # 获取当前UAV模型的全部参数
#         other_params = []
#         for i, net in enumerate(nets):
#             if i == agent_no:
#                 continue
#             other_params.append(net.get_weights())  # 获取其他UAV模型的全部参数列表
#         # 对当前UAV模型中的“所有层的参数”进行循环
#         for i in range(len(target_params)):
#             others = np.sum([w[i] for w in other_params], axis=0) / len(other_params)  # 其他UAV（边缘智能体）的参数 PS：求均值
#             target_params[i] = omega * target_params[i] + others * (1 - omega)  # 当前UAV保留以权值（系数）为ω的参数，并混合其他UAV的参数
#             # print([others.shape, target_params[i].shape])
#         nets[agent_no].set_weights(target_params)  # 将新参数装载入模型
# # def sample_devices(P_values, K):
# #     if not isinstance(P_values, list) or len(P_values) == 0:
# #         raise ValueError("P_values must be a non-empty list")
# #     P_values = np.nan_to_num(P_values, nan=0.0)
# #     total = np.sum(P_values)

# #     if total == 0:
# #         # 如果总概率和为 0，则平均分配概率
# #         P_values = [1.0 / len(P_values)] * len(P_values)
# #     else:
# #         # 正规化概率，确保概率之和为 1
# #         P_values = P_values / total

# #     # 根据 P 值作为权重进行采样
# #     C_t = np.random.choice(len(P_values), size=K, replace=False, p=P_values)
# #     print('-----------------p_value--------------------------')
# #     print(P_values)
# #     print('--------------------C_t-----------------------')
# #     print(C_t)
# #     return C_t

def sample_devices(P_values, K):
    print('-----------------p_value--------------------------')
    print(P_values)

    # 根据 P 值作为权重进行采样
    C_t = np.random.choice(len(P_values), size=K, replace=False, p=P_values)
    print('--------------------C_t-----------------------')
    print(C_t)
    return C_t
def compute_delta_weights(new_weights_list, global_weights):
    delta_weights = [new_w - global_weights for new_w in new_weights_list]
    return delta_weights

def update_global_weights(C_t, I_values, new_weights, global_weights):

    sum_I_values = np.sum([np.abs(I_values[k]) for k in C_t])
 
    weighted_updates = np.zeros_like(global_weights)
    
    # 为选中的每个设备计算加权更新
    for k in C_t:
        delta_w = new_weights[k] - global_weights
        weighted_update = (I_values[k] / sum_I_values) * delta_w
        weighted_updates += weighted_update

    updated_global_weights = global_weights + weighted_updates
    
    return updated_global_weights
def reshape_weights(flat_weights, shapes):
    reshaped_weights = []
    index = 0
    for shape in shapes:
        size = np.product(shape)
        weight = flat_weights[index:index + size].reshape(shape)
        reshaped_weights.append(weight)
        index += size
    return reshaped_weights
def merge_fl(nets, gard, old_gard, weights, old_weights, psi, knum, mu=0.1, omega=0.5):
    epsilon = 1e-8  
    all_gards = []
    all_old_gards = []
    all_weights = []
    all_old_weights = []
    gammas = []
    I_values = []

    accumulated_old_gards = None

    for no in range(len(nets)):

        current_gards = [grad.flatten() + epsilon for grad in gard[no]]
        all_gards.append(np.concatenate(current_gards))

        current_old_gards = [grad.flatten() + epsilon for grad in old_gard[no]]
        flattened_old_gards = np.concatenate(current_old_gards)
        all_old_gards.append(flattened_old_gards)
        if accumulated_old_gards is None:
            accumulated_old_gards = flattened_old_gards
        else:
            accumulated_old_gards += flattened_old_gards

        current_weights = [weight.flatten() + epsilon for weight in weights[no]]
        all_weights.append(np.concatenate(current_weights))

        current_old_weights = [weight.flatten() + epsilon for weight in old_weights[no]]
        all_old_weights.append(np.concatenate(current_old_weights))

    for no in range(len(nets)):
        grad_diff = all_gards[no] + mu * (all_weights[no] - all_old_weights[no])
        grad_norm_t = np.linalg.norm(all_old_gards[no])
        gamma_k = np.linalg.norm(grad_diff) / (grad_norm_t + epsilon)
        gammas.append(gamma_k)

    global_grad = np.zeros_like(all_old_gards[0])
    global_weights  = np.zeros_like(all_old_weights[0])
    for no in range(len(nets)):
        global_grad += all_gards[no]
        global_weights += all_weights[no]
        
    global_grad = global_grad / len(nets)
    global_weights = global_weights / len(nets)

    for no in range(len(nets)):
        inner_product = np.dot(global_grad, all_old_gards[no])
        norm_squared = -psi * gammas[no] * np.linalg.norm(global_grad)**2
        I_k = inner_product + norm_squared
        I_values.append(I_k)

    I_total = np.sum([np.abs(I) for I in I_values])

    P_values = [np.abs(I) / I_total for I in I_values]

    C_t = sample_devices(P_values, knum)

    new_weights = update_global_weights(C_t, I_values, all_weights, global_weights)
    # for k in C_t:
    #     expected_shapes = [w.shape for w in nets[0].get_weights()]
    #     reshaped_weights = reshape_weights(new_weights, expected_shapes)
    #     nets[k].set_weights(reshaped_weights)
    for agent_no in range(len(nets)):
        expected_shapes = [w.shape for w in nets[no].get_weights()]
        reshaped_weights = reshape_weights(new_weights, expected_shapes)
        nets[agent_no].set_weights(reshaped_weights)


# 返回①最大概率且②最小移动距离的UAV移动位置的索引
def circle_argmax(move_dist, move_r):
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))  # 找到最大概率UAV移动位置的索引
    # print(tf.squeeze(move_dist, axis=-1))
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)  # 假设有多个概率最大值，则返回移动距离最短的
    # print(max_pos)
    return max_pos[np.argmin(pos_dist)]

# MAACAgent类
class MAACAgent(object):
    def __init__(self, env, tau, gamma, lr_sa, lr_sc, lr_aa, lr_ac, lr_ca, lr_cc, batch, epsilon=0.2):
        self.env = env  # MEC环境
        self.sensors = self.env.sensors  # sensor智能体列表*
        self.sensor_num = self.env.sensor_num  # sensor智能体数量*
        self.agents = self.env.agents  # agent列表
        self.agent_num = self.env.agent_num  # agent数量
        self.index_dim = 2  # 索引维度
        self.s_index_dim = 1  # 索引维度
        self.obs_r = self.env.obs_r  # UAV观察半径
        self.state_map_shape = (self.obs_r * 2 + 1, self.obs_r * 2 + 1, self.index_dim)  # UAV观察到的状态地图维度
        self.pos_shape = (2)  # 位置坐标维度
        self.band_shape = (1)  # 带宽维度
        self.s_buffstate_shape = (self.index_dim, self.env.max_buffer_size * 2)  # （sensor有的）源缓冲区状态维度
        self.buffstate_shape = (self.index_dim, self.env.max_buffer_size)  # （UAV有的）待计算数据缓冲区状态/计算完成数据缓冲区状态维度
        # self.sensor_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        # self.agent_map_shape = (self.env.map_size, self.env.map_size, self.index_dim)
        self.buffer_list_shape = (self.agent_num, self.index_dim, self.env.max_buffer_size)  # （center有的）所有UAV计算完成数据缓冲区列表维度
        self.pos_list_shape = (self.agent_num, 2)  # （center有的）所有UAV位置坐标列表维度
        self.bandvec_shape = (self.env.agent_num)  # （center有的）为所有agent分配带宽维度
        self.op_shape = (self.index_dim, self.env.max_buffer_size)  # （UAV有的）UAV动作中的“执行和卸载”的维度
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.move_r)  # UAV可移动圆形范围内“离散点的个数，离散点的坐标（修正后）”
        self.movemap_shape = (self.env.move_r * 2 + 1, self.env.move_r * 2 + 1)  # UAV移动地图维度
        self.epsilon = epsilon  # 强制UAV随机动作的概率
        self.UAVs_total_distance = 0  # 所有UAV的总距离  # ***

        # 强化学习参数 learning params
        self.tau = tau  # target网络更新中，保留原始target网络参数的比例系数
        self.cnn_kernel_size = 3  # 神经网络中的卷积核尺寸
        self.gamma = gamma  # replay training中，target critic网络输出的比例系数
        self.lr_sa = lr_sa  # sensor actor网络学习率（用于设置Adam优化器）
        self.lr_sc = lr_sc  # sensor critic网络学习率（用于设置Adam优化器）
        self.lr_aa = lr_aa  # agent actor网络学习率（用于设置Adam优化器）
        self.lr_ac = lr_ac  # agent critic网络学习率（用于设置Adam优化器）
        self.lr_ca = lr_ca  # center actor网络学习率（用于设置Adam优化器）
        self.lr_cc = lr_cc  # center critic网络学习率（用于设置Adam优化器）
        self.batch_size = batch
        self.sensor_memory = {}  # 所有sensor的经验缓冲区 sensor_memory字典
        self.agent_memory = {}  # 所有UAV的经验缓冲区 agent_memory字典
        self.softmax_memory = {}  # 没用到
        self.center_memory = []  # center的经验缓冲区 center_memory字典
        self.sample_prop = 1 / 2  # replay training采样中，使用最新经验数据的比例

        # 强化学习网络初始化 net init
        # 创建sensor actor网络列表
        self.sensor_actors = []
        # 创建agent actor网络列表
        self.agent_actors = []
        # 创建center actor网络
        self.center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        # 创建sensor critic网络列表
        self.sensor_critics = []
        # 创建agent critic网络列表
        self.agent_critics = []
        # 创建center critic网络
        self.center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)

        # 创建target sensor actor网络列表
        self.target_sensor_actors = []
        # 创建target agent actor网络列表
        self.target_agent_actors = []
        # 创建target center actor网络
        self.target_center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        # 更新target center actor网络权重参数
        update_target_net(self.center_actor, self.target_center_actor, tau=0)
        # 创建target sensor critic网络列表
        self.target_sensor_critics = []
        # 创建target agent critic网络列表
        self.target_agent_critics = []
        # 创建target center critic网络
        self.target_center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        # 更新target center critic网络权重参数
        update_target_net(self.center_critic, self.target_center_critic, tau=0)
        self.agent_actor_weights={}
        self.agent_actor_old_weights={}
        self.agent_critic_weights={}
        self.agent_critic_old_weights={}
        self.agent_actor_gard = {}
        self.agent_actor_old_gard = {}
        self.agent_critic_gard = {}
        self.agent_critic_old_gard = {}
        self.sensor_actor_weights={}
        self.sensor_actor_old_weights={}
        self.sensor_critic_weights={}
        self.sensor_critic_old_weights={}
        self.sensor_actor_gard = {}
        self.sensor_actor_old_gard = {}
        self.sensor_critic_gard = {}
        self.sensor_critic_old_gard = {}
        
        self.center_actor_weights=[]
        self.center_actor_old_weights=[]
        self.center_critic_weights=[]
        self.center_critic_old_weights=[]
        self.center_actor_gard = []
        self.center_actor_old_gard = []
        self.center_critic_gard = []
        self.center_critic_old_gard = []
        # sensor actor优化器列表
        self.sensor_actor_opt = []
        # sensor critic优化器列表
        self.sensor_critic_opt = []
        # agent actor优化器列表
        self.agent_actor_opt = []
        # agent critic优化器列表
        self.agent_critic_opt = []
        # center actor优化器
        self.center_actor_opt = keras.optimizers.Adam(learning_rate=lr_ca)  # Adam优化器（学习率衰减）
        # center critic优化器
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)  # Adam优化器（学习率衰减）

        # tensorboard可视化的记录文件
        self.summaries = {}

        # 对所有sensor进行循环
        for i in range(self.env.sensor_num):
            # 设置优化器
            self.sensor_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_sc))  # Adam优化器（学习率衰减）
            self.sensor_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_sa))  # Adam优化器（学习率衰减）

            # 创建sensor actor及对应target网络
            new_sensor_actor = sensor_actor([self.s_buffstate_shape], self.cnn_kernel_size)
            target_sensor_actor = sensor_actor([self.s_buffstate_shape], self.cnn_kernel_size)
            # 更新target权重参数
            update_target_net(new_sensor_actor, target_sensor_actor, tau=0)
            # 将sensor actor及对应target网络加入列表
            self.sensor_actors.append(new_sensor_actor)
            self.target_sensor_actors.append(target_sensor_actor)

            # 创建sensor critic及对应的target网络
            new_sensor_critic = sensor_critic([self.s_buffstate_shape, self.s_index_dim], self.cnn_kernel_size)
            target_sensor_critic = sensor_critic([self.s_buffstate_shape, self.s_index_dim], self.cnn_kernel_size)
            # 更新target权重参数
            update_target_net(new_sensor_critic, target_sensor_critic, tau=0)
            # 将sensor critic及对应target网络加入列表
            self.sensor_critics.append(new_sensor_critic)
            self.target_sensor_critics.append(target_sensor_critic)

        # 对所有UAV进行循环
        for i in range(self.env.agent_num):
            # 设置优化器
            self.agent_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_ac))  # Adam优化器（学习率衰减）
            self.agent_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_aa))  # Adam优化器（学习率衰减）
            # 创建agent actor及对应target网络
            new_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            target_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # new_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # target_agent_actor = agent_actor([self.state_map_shape, self.pos_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            # 更新target权重参数
            update_target_net(new_agent_actor, target_agent_actor, tau=0)

            # 将agent actor及对应target网络加入列表
            self.agent_actors.append(new_agent_actor)
            self.target_agent_actors.append(target_agent_actor)

            # 创建agent critic及对应的target网络
            new_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape,
                                             self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            t_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            # 更新target权重参数
            update_target_net(new_agent_critic, t_agent_critic, tau=0)
            # 将agent critic及对应target网络加入列表
            self.agent_critics.append(new_agent_critic)
            self.target_agent_critics.append(t_agent_critic)
        for no in range(5):
            # 初始化权重为零
            zero_weights = [tf.zeros_like(var, dtype=tf.float32) for var in self.agent_critics[no].trainable_variables]
            self.agent_critic_old_weights[no] = zero_weights
            self.agent_actor_old_weights[no] = zero_weights
            self.agent_critic_weights[no] = zero_weights
            self.agent_actor_weights[no] = zero_weights
        # Initialize self.center_actor_old_weights and self.center_critic_old_weights to zero
        self.center_actor_old_weights = [tf.Variable(tf.zeros_like(weight), trainable=False) for weight in self.center_actor.trainable_variables]
        self.center_critic_old_weights = [tf.Variable(tf.zeros_like(weight), trainable=False) for weight in self.center_critic.trainable_variables]


        for no in range(60):
            # 初始化权重为零
            zero_weights = [tf.zeros_like(var, dtype=tf.float32) for var in self.sensor_critics[no].trainable_variables]
            self.sensor_critic_old_weights[no] = zero_weights
            self.sensor_actor_old_weights[no] = zero_weights
            self.sensor_critic_weights[no] = zero_weights
            self.sensor_actor_weights[no] = zero_weights
        # plot_model绘制神经网络(keras模型)结构示意图
        os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # 浩哲添加
        keras.utils.plot_model(self.center_actor, 'logs/model_figs/new_center_actor.png', show_shapes=True)
        keras.utils.plot_model(self.center_critic, 'logs/model_figs/new_center_critic.png', show_shapes=True)
        keras.utils.plot_model(self.agent_actors[0], 'logs/model_figs/new_agent_actor.png', show_shapes=True)
        keras.utils.plot_model(self.agent_critics[0], 'logs/model_figs/new_agent_critic.png', show_shapes=True)
        keras.utils.plot_model(self.sensor_actors[0], 'logs/model_figs/new_sensor_actor.png', show_shapes=True)
        keras.utils.plot_model(self.sensor_critics[0], 'logs/model_figs/new_sensor_critic.png', show_shapes=True)

    # 所有UAV和云中心的动作（包含了env.step环境更新！）
    def actor_act(self, epoch):
        tmp = random.random()  # 生成一个0到1的随机浮点数
        if tmp >= self.epsilon and epoch >= 16:  # 若①(1-epsilon)概率；②epoch>16
            # ***************************sensor act***************************
            sensor_state_list = []  # sensor状态列表，目标只有一种状态——sensor_buffer
            sensor_softmax_list = []  # sensor动作的softmax形式列表,两种动作——sensor_computing
            sensor_act_list = []  # sensor动作列表,两种动作——sensor_collecting
            for i, sensor in enumerate(self.sensors):
                # 从MEC环境中获取agent_actor的输入，并打包成“UAV状态列表”
                sensor_data_state = tf.expand_dims(sensor.get_sensor_data(), axis=0)  # （shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                sensor_state = [sensor_data_state]  # sensor状态
                sensor_state_list.append(sensor_state)  # sensor状态列表

                # 将sensor状态送入sensor_actor网络进行预测，获得神经网络的输出（computing_dev, collecting_dev）
                sensor_action_output = self.sensor_actors[i].predict(sensor_state)
                # sensor_comdev = sensor_action_output[0][0]  # sensor_actor输出中的sensor本地计算策略（shape为2）
                sensor_comdev = sensor_action_output[0]  # sensor_actor输出中的sensor本地计算策略（shape为2）

                # 将神经网络的输出（computing_dev, collecting_cev）转变成softmax形式（参考center，数据原本就是softmax形式，只需要扩展维度即可）
                sensor_com_softmax = tf.expand_dims(sensor_comdev, axis=0)

                sensor_act_list.append([sensor_comdev])
                sensor_softmax_list.append([sensor_com_softmax])

            # ***************************agent act***************************
            agent_act_list = []  # agent动作列表
            softmax_list = []  # agent动作的softmax形式列表
            cur_state_list = []  # 当前状态列表
            band_vec = np.zeros(self.agent_num)  # UAV动作中的带宽列表
            for i, agent in enumerate(self.agents):
                # 从MEC环境中获取agent_actor的输入，并打包成“组合状态列表”
                # actor = self.agent_actors[i]
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)  # 对某个agent的观察（shape为obs_x*obs_y*2）增加一个维度（shape为1*obs_x*obs_y*2）
                # pos = tf.expand_dims(agent.position, axis=0)
                # print('agent{}pos:{}'.format(i, pos))
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)  # 对待计算数据（shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)  # 对计算完成数据（shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                band = tf.expand_dims(agent.action.bandwidth, axis=0)  # 对带宽（shape为1）增加一个维度（shape为1*1）
                # print('band{}'.format(agent.action.bandwidth))
                band_vec[i] = agent.action.bandwidth
                assemble_state = [state_map, total_data_state, done_data_state, band]  # 组合状态列表
                # print(['agent%s' % i, sum(sum(state_map))])
                cur_state_list.append(assemble_state)
                # print(total_data_state.shape)

                # 将组合状态列表送入agent_actor网络进行预测，获得神经网络的输出（move,operation）
                action_output = self.agent_actors[i].predict(assemble_state)
                move_dist = action_output[0][0]  # agent_actor输出中的UAV动作中的移动(0~move_r*2+1, 0~move_r*2+1)
                # print(move_dist)
                # print(move_dist.shape)
                sio.savemat('debug.mat', {'state': self.env.get_obs(agent), 'move': move_dist})  # 保存mat文件（包括UAV观察到的状态和UAV移动策略）
                # print(move_dist)
                # print(move_dist.shape)
                op_dist = action_output[1][0]  # agent_actor输出中的收集、卸载
                # print(op_dist)
                # print(op_dist.shape)
                # move_ori = np.unravel_index(np.argmax(move_dist), move_dist.shape)

                # 将神经网络的输出（move,operation）转变成坐标形式（UAV的移动）or单热点形式（UAV的计算和卸载）
                # UAV动作中的移动 概率形式->位置坐标形式
                move_ori = circle_argmax(move_dist, self.env.move_r)  # UAV移动位置坐标（非负）
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]  # UAV移动位置坐标（修正后）
                # 记录UAV在当前episode的移动距离  # ***
                agent.t_distance = np.linalg.norm(np.array(move))
                agent.e_distance += agent.t_distance  # ***
                agent.total_distance += agent.t_distance  # ***
                self.UAVs_total_distance += agent.t_distance  # ***
                # 总结UAV的移动距离信息 summary info  # ***
                self.summaries['agent%s-e_distance' % i] = agent.e_distance  # ***
                self.summaries['agent%s-total_distance' % i] = agent.total_distance  # ***

                # UAV动作中的计算和卸载策略 概率形式->单热点形式
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(op_dist[0])] = 1
                offloading[np.argmax(op_dist[1])] = 1

                # 将神经网络的输出（move,operation）转变成softmax形式
                move_softmax = np.zeros(move_dist.shape) # shape()
                op_softmax = np.zeros(self.buffstate_shape)
                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(op_dist[0])] = 1
                op_softmax[1][np.argmax(op_dist[1])] = 1

                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                op_softmax = tf.expand_dims(op_softmax, axis=0)

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])
            # print(agent_act_list)

            # ***************************center act***************************
            # 从MEC环境中获取center_actor的输入，并打包成“云中心状态列表”
            done_buffer_list, pos_list = self.env.get_center_state()  # 每个UAV计算完成的数据(UAV数量, 2, 缓冲区上限) UAV位置坐标(UAV数量, 2)
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)

            # 将云中心状态列表送入center_actor网络进行预测，获得神经网络的输出（bandwidth_vec）
            new_bandvec = self.center_actor.predict([done_buffer_list, pos_list])
            # print('new_bandwidth{}'.format(new_bandvec[0]))

            # ***************************step***************************
            # 将agent和center的actor网络中输出的UAV和云中心的动作 输入到MEC环境中 MEC环境step更新
            new_state_maps, new_rewards, average_age, fairness_index, done, info = self.env.step(agent_act_list, new_bandvec[0], sensor_act_list)

            # ***************************record memory***************************
            # 对所有sensor进行循环
            for i, sensor in enumerate(self.sensors):
                # 在MEC环境step更新后，从MEC环境中获取sensor_actor的输入，并打包成“新sensor状态列表”
                sensor_data_state = tf.expand_dims(sensor.get_sensor_data(), axis=0)  # （shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                new_sensor_state = [sensor_data_state]  # step更新后，sensor状态列表
                # 将某个sensor的[当前sensor状态、sensor actor的softmax输出、新reward、新sensor状态]存储到sensor_memory字典
                if sensor.no in self.sensor_memory.keys():
                    self.sensor_memory[sensor.no].append([sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state])
                else:
                    self.sensor_memory[sensor.no] = [[sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state]]

            # 对所有UAV进行循环
            for i, agent in enumerate(self.agents):
                state_map = new_state_maps[i]
                # print(['agent%s' % i, sum(sum(state_map))])
                # pos = agent.position
                total_data_state = agent.get_total_data()
                done_data_state = agent.get_done_data()
                # 在MEC环境step更新后，从MEC环境中获取agent_actor的输入，并打包成“新组合状态列表”
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                new_states = [state_map, total_data_state, done_data_state, band]
                # 将某个UAV的[当前agent状态、agent actor的softmax输出、新reward、新agent状态、done]存储到agent_memory字典
                if agent.no in self.agent_memory.keys():
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]

            # step更新后，获取center相关的数据
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)
            # 将云中心的[当前agent状态、agent actor的softmax输出、新reward、新agent状态、done]存储到agent_memory字典
            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])

        else:
            # 随机动作 random action
            # ***************************sensors***************************
            sensor_state_list = []  # sensor状态列表，目标只有一种状态——sensor_buffer
            sensor_softmax_list = []  # sensor动作的softmax形式列表,两种动作——sensor_computing
            sensor_act_list = []  # sensor动作列表,两种动作——sensor_collecting
            for i, sensor in enumerate(self.sensors):
                # 从MEC环境中获取agent_actor的输入，并打包成“UAV状态列表”
                sensor_data_state = tf.expand_dims(sensor.get_sensor_data(),axis=0)  # （shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                sensor_state = [sensor_data_state]  # sensor状态
                sensor_state_list.append(sensor_state)  # sensor状态列表

                # 随机选择sensor动作
                sensor_comdev = np.random.rand(self.s_index_dim)
                sensor_act_list.append([sensor_comdev])

                # 将神经网络的输出（computing_dev, collecting_cev）转变成softmax形式（参考center，数据原本就是softmax形式，只需要扩展维度即可）
                sensor_com_softmax = tf.expand_dims(sensor_comdev, axis=0)

                sensor_act_list.append([sensor_comdev])
                sensor_softmax_list.append([sensor_com_softmax])
            # ***************************agents***************************
            agent_act_list = []  # agent动作列表
            softmax_list = []  # agent动作的softmax形式列表
            cur_state_list = []  # 当前状态列表
            band_vec = np.zeros(self.agent_num)  # UAV动作中的带宽列表
            for i, agent in enumerate(self.agents):
                # 从MEC环境中获取agent_actor的输入，并打包成“组合状态列表”
                # actor = self.agent_actors[i]
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)  # 对某个agent的观察（shape为obs_x*obs_y*2）增加一个维度（shape为1*obs_x*obs_y*2）
                # pos = tf.expand_dims(agent.position, axis=0)
                # print('agent{}pos:{}'.format(i, pos))
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)  # 对待计算数据（shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)  # 对计算完成数据（shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                band = tf.expand_dims(agent.action.bandwidth, axis=0)  # 对带宽（shape为1）增加一个维度（shape为1*1）
                # print('band{}'.format(agent.action.bandwidth))
                band_vec[i] = agent.action.bandwidth
                assemble_state = [state_map, total_data_state, done_data_state, band]  # 组合状态列表
                # print(['agent%s' % i, sum(sum(state_map))])
                cur_state_list.append(assemble_state)
                # print(total_data_state.shape)

                # 随机选择agent动作
                move = random.sample(list(self.move_dict.values()), 1)[0]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.random.randint(agent.max_buffer_size)] = 1
                offloading[np.random.randint(agent.max_buffer_size)] = 1

                # 记录UAV在当前episode的移动距离  # ***
                agent.t_distance = np.linalg.norm(np.array(move))
                agent.e_distance += agent.t_distance  # ***
                agent.total_distance += agent.t_distance  # ***
                self.UAVs_total_distance += agent.t_distance  # ***
                # 总结UAV的移动距离信息 summary info  # ***
                self.summaries['agent%s-e_distance' % i] = agent.e_distance  # ***
                self.summaries['agent%s-total_distance' % i] = agent.total_distance  # ***


                # 将神经网络的输出（move,operation）转变成softmax形式
                move_softmax = np.zeros((2 * self.env.move_r + 1, 2 * self.env.move_r + 1, 1))  # shape()
                op_softmax = np.zeros(self.buffstate_shape)
                move_ori = [move[1] + self.env.move_r, move[0] + self.env.move_r]  # UAV移动位置坐标（修正后）
                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(execution)] = 1
                op_softmax[1][np.argmax(offloading)] = 1

                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                op_softmax = tf.expand_dims(op_softmax, axis=0)

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])

            # ***************************center***************************
            # 从MEC环境中获取center_actor的输入，并打包成“云中心状态列表”
            done_buffer_list, pos_list = self.env.get_center_state()  # 每个UAV计算完成的数据(UAV数量, 2, 缓冲区上限) UAV位置坐标(UAV数量, 2)
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)

            # 随机选择center动作
            new_bandvec = np.random.rand(self.agent_num)
            new_bandvec = new_bandvec / np.sum(new_bandvec)

            # ***************************step***************************
            new_state_maps, new_rewards, average_age, fairness_index, done, info = self.env.step(agent_act_list, new_bandvec, sensor_act_list)

            # ***************************record memory***************************
            # 对所有sensor进行循环
            for i, sensor in enumerate(self.sensors):
                # 在MEC环境step更新后，从MEC环境中获取sensor_actor的输入，并打包成“新sensor状态列表”
                sensor_data_state = tf.expand_dims(sensor.get_sensor_data(),axis=0)  # （shape为2*缓冲区上限）增加一个维度（shape为1*2*缓冲区上限）
                new_sensor_state = [sensor_data_state]  # step更新后，sensor状态列表
                # 将某个sensor的[当前sensor状态、sensor actor的softmax输出、新reward、新sensor状态]存储到sensor_memory字典
                if sensor.no in self.sensor_memory.keys():
                    self.sensor_memory[sensor.no].append([sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state])
                else:
                    self.sensor_memory[sensor.no] = [[sensor_state_list[i], sensor_softmax_list[i], new_rewards[-1], new_sensor_state]]

            # 对所有UAV进行循环
            for i, agent in enumerate(self.agents):
                state_map = new_state_maps[i]
                # print(['agent%s' % i, sum(sum(state_map))])
                # pos = agent.position
                total_data_state = agent.get_total_data()
                done_data_state = agent.get_done_data()
                # 在MEC环境step更新后，从MEC环境中获取agent_actor的输入，并打包成“新组合状态列表”
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                new_states = [state_map, total_data_state, done_data_state, band]
                # 将某个UAV的[当前agent状态、agent actor的softmax输出、新reward、新agent状态、done]存储到agent_memory字典
                if agent.no in self.agent_memory.keys():
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]

            # step更新后，获取center相关的数据
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)
            # 将云中心的[当前agent状态、agent actor的softmax输出、新reward、新agent状态、done]存储到agent_memory字典
            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])

        # 总结所有UAVs的总移动距离信息 summary info  # ***
        self.summaries['UAVs_total_distance'] = self.UAVs_total_distance  # ***
        # 返回reward（所有sensor的平均年龄）
        return new_rewards[-1], average_age[-1], fairness_index[-1]

    # 经验回放
    # @tf.function(experimental_relax_shapes=True)
    def replay(self):
        # 0. sensor replay
        # 对sensor_memory字典中的所有UAV及UAV的经验进行循环
        mu = 0.1
        for no, sensor_memory in self.sensor_memory.items():
            # 若经验缓冲区长度 < 批大小（经验缓冲区中样本数量不足）
            if len(sensor_memory) < self.batch_size:
                # 跳过sensor replay training
                continue
            # 对经验缓冲区内的数据进行采样
            # 采样规则：最新的0.25*batch_size条经验数据+最新的2*batch_size条经验数据中随机抽取的0.75*batch_size条
            samples = sensor_memory[-int(self.batch_size * self.sample_prop):] + random.sample(sensor_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))

            # 0.1 从采样到的经验数据中提取并整理各类数据
            # state
            sensor_data_state = np.vstack([sample[0][0] for sample in samples])  # 将多条经验数据中的“sensor状态中的源数据状态”，堆叠构成一个新的数组
            # action
            sensor_computing = np.vstack([sample[1][0] for sample in samples])  # 将多条经验数据中的"sensor动作中的本地计算比例"，堆叠构成一个新的数组
            # reward
            s_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1)  # 将多条经验数据中的"（env.step更新后）sensor的reward"，堆叠构成一个新的数组（这个reward是利用所有sensor求均值得到的）
            # new states
            new_sensor_data_state = np.vstack([sample[3][0] for sample in samples])  # 将多条经验数据中的“（env.step更新后）sensor状态中的源数据状态”，堆叠构成一个新的数组

            # 根据（env.step更新后）下一时隙的状态预测下一时隙动作和reward（这个reward是利用target_sensor_critic网络得到的） （next actions & rewards）
            new_sensor_action = self.target_sensor_actors[no].predict([new_sensor_data_state])
            # sq_future = self.target_sensor_critics[no].predict([new_sensor_data_state, new_sensor_action[0]])
            sq_future = self.target_sensor_critics[no].predict([new_sensor_data_state, new_sensor_action])
            # print('qfuture{}'.format(q_future))
            s_target_qs = s_reward + sq_future * self.gamma

            # 0.2 训练critic网络 train critic
            with tf.GradientTape() as tape:  # 创建梯度记录环境（梯度带：自动求导的记录器）
                # tape.watch(self.sensor_critics[no].trainable_variables)
                sq_values = self.sensor_critics[no]([sensor_data_state, sensor_computing])
                sc_error = sq_values - tf.cast(s_target_qs, dtype=tf.float32)  # tf.cast()转换数据类型
                # ac_error = q_values - target_qs
                msc_loss = tf.reduce_mean(tf.math.square(sc_error))  # 计算平均误差
                regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
                                                for current_weight, old_weight in zip(self.sensor_critic_weights[no], self.sensor_critic_old_weights[no])])
                regularization_loss = mu * regularization_term
                sc_loss = msc_loss + regularization_loss
            sc_grad = tape.gradient(sc_loss, self.sensor_critics[no].trainable_variables)

            self.sensor_critic_old_gard[no] = [grad.numpy() for grad in sc_grad]
            self.sensor_critic_old_weights[no] = [var.numpy() for var in self.sensor_critics[no].trainable_variables]
            # print(ac_grad)
            self.sensor_critic_opt[no].apply_gradients(zip(sc_grad, self.sensor_critics[no].trainable_variables))  # 自动更新模型参数
            self.sensor_critic_gard[no] = [grad.numpy() for grad in sc_grad]
            self.sensor_critic_weights[no] = [var.numpy() for var in self.sensor_critics[no].trainable_variables]

            # 0.3 训练actor网络 train actor
            with tf.GradientTape() as tape:  # 创建梯度记录环境（梯度带：自动求导的记录器）
                tape.watch(self.sensor_actors[no].trainable_variables)  # 跟踪指定的tensor变量，即将tensor变量加入自动求导（GradientTape默认只对tf.Variable类型的变量进行监控）
                s_actions = self.sensor_actors[no]([sensor_data_state])
                # s_new_r = self.sensor_critics[no]([sensor_data_state, s_actions[0]])
                s_new_r = self.sensor_critics[no]([sensor_data_state, s_actions])
                # print(new_r)
                msa_loss = tf.reduce_mean(s_new_r)  # 计算平均误差
                regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
                                                for current_weight, old_weight in zip(self.sensor_actor_weights[no], self.sensor_actor_old_weights[no])])
                regularization_loss = mu * regularization_term
                # print(aa_loss)
                sa_loss = msa_loss + regularization_loss
            sa_grad = tape.gradient(sa_loss, self.sensor_actors[no].trainable_variables)  # 自动计算所有梯度

            self.sensor_actor_old_gard[no] = [grad.numpy() for grad in sa_grad ]
            self.sensor_actor_old_weights[no] = [var.numpy() for var in self.sensor_actors[no].trainable_variables]
            # print(ac_grad)
            self.sensor_actor_opt[no].apply_gradients(zip(sa_grad , self.sensor_actors[no].trainable_variables))  # 自动更新模型参数
            self.sensor_actor_gard[no] = [grad.numpy() for grad in sa_grad ]
            self.sensor_actor_weights[no] = [var.numpy() for var in self.sensor_actors[no].trainable_variables]
            # 总结UAV的actor和critic网络的训练平均误差信息 summary info
            self.summaries['sensor%s-critic_loss' % no] = sc_loss
            self.summaries['sensor%s-actor_loss' % no] = sa_loss

        # 1. agent（UAV） replay
        # 对agent_memory字典中的所有UAV及UAV的经验进行循环
        for no, agent_memory in self.agent_memory.items():
            # 若经验缓冲区长度 < 批大小（经验缓冲区中样本数量不足）
            if len(agent_memory) < self.batch_size:
                # 跳过agent replay training
                continue
            # print([len(agent_memory[-100:]), self.batch_size])
            # 对经验缓冲区内的数据进行采样
            # 采样规则：最新的0.25*batch_size条经验数据+最新的2*batch_size条经验数据中随机抽取的0.75*batch_size条
            samples = agent_memory[-int(self.batch_size * self.sample_prop):] + random.sample(agent_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
            # t_agent_actor = self.target_agent_actors[no]
            # t_agent_critic = self.target_agent_critics[no]
            # agent_actor = self.agent_actors[no]
            # agent_critic = self.agent_critics[no]

            # 1.1 从采样到的经验数据中提取并整理各类数据
            # state
            state_map = np.vstack([sample[0][0] for sample in samples])  # 将多条经验数据中的“agent状态中的状态地图”，堆叠构成一个新的数组
            # pos = np.vstack([sample[0][1] for sample in samples])
            total_data_state = np.vstack([sample[0][1] for sample in samples])  # 将多条经验数据中的“agent状态中的待计算数据状态”，堆叠构成一个新的数组
            done_data_state = np.vstack([sample[0][2] for sample in samples])  # 将多条经验数据中的“agent状态中的计算完成数据状态”，堆叠构成一个新的数组
            band = np.vstack([sample[0][3] for sample in samples])  # 将多条经验数据中的“agent状态中的UAV分配到的带宽”，堆叠构成一个新的数组
            # action
            move = np.vstack([sample[1][0] for sample in samples])  # 将多条经验数据中的"agent动作中的移动 move_out"，堆叠构成一个新的数组
            op_softmax = np.vstack([sample[1][1] for sample in samples])  # 将多条经验数据中的"agent动作中的计算和卸载 op_dist"，堆叠构成一个新的数组
            # reward
            a_reward = tf.expand_dims([sample[2] for sample in samples], axis=-1)  # 将多条经验数据中的"（env.step更新后）agent的reward"，堆叠构成一个新的数组（这个reward是利用所有sensor求均值得到的）
            # new states
            new_state_map = np.vstack([sample[3][0] for sample in samples])  # 将多条经验数据中的“（env.step更新后）agent状态中的状态地图”，堆叠构成一个新的数组
            # new_pos = np.vstack([sample[3][1] for sample in samples])
            new_total_data_state = np.vstack([sample[3][1] for sample in samples])  # 将多条经验数据中的“（env.step更新后）agent状态中的待计算数据状态”，堆叠构成一个新的数组
            new_done_data_state = np.vstack([sample[3][2] for sample in samples])  # 将多条经验数据中的“（env.step更新后）agent状态中的计算完成数据状态”，堆叠构成一个新的数组
            new_band = np.vstack([sample[3][3] for sample in samples])  # 将多条经验数据中的“（env.step更新后）agent状态中的UAV分配到的带宽”，堆叠构成一个新的数组
            # # done
            # done = [sample[4] for sample in samples]

            # 根据（env.step更新后）下一时隙的状态预测下一时隙动作和reward（这个reward是利用target_agent_critic网络得到的） （next actions & rewards）
            new_actions = self.target_agent_actors[no].predict([new_state_map, new_total_data_state, new_done_data_state, new_band])
            # new_move = np.array([self.move_dict[np.argmax(single_sample)] for single_sample in new_actions[0]])
            # print(new_actions[1].shape)
            q_future = self.target_agent_critics[no].predict([new_state_map, new_total_data_state, new_done_data_state, new_actions[0], new_actions[1], new_band])
            # print('qfuture{}'.format(q_future))
            target_qs = a_reward + q_future * self.gamma

            with tf.GradientTape() as tape:
                # ... [现有的代码段] ...

                # 计算原始的损失函数部分，也就是均方误差部分
                q_values = self.agent_critics[no]([state_map, total_data_state, done_data_state, move, op_softmax, band])
                ac_error = q_values - tf.cast(target_qs, dtype=tf.float32)
                mse_loss = tf.reduce_mean(tf.math.square(ac_error))

                # 添加新的正则化项，这里假设mu是已经定义的正则化系数，old_weights是 w^t
                # 需要确保self.agent_critic_old_weights[no]已经被正确赋值为上一轮迭代的权重
                regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
                                                for current_weight, old_weight in zip(self.agent_critic_weights[no], self.agent_critic_old_weights[no])])
                regularization_loss = mu * regularization_term
                
                # 总损失函数是均方误差损失和正则化损失的和
                ac_loss = mse_loss + regularization_loss

            # 获取梯度并应用
            ac_grad = tape.gradient(ac_loss, self.agent_critics[no].trainable_variables)
            self.agent_critic_old_gard[no] = [grad.numpy() for grad in ac_grad]
            self.agent_critic_old_weights[no] = [var.numpy() for var in self.agent_critics[no].trainable_variables]
            self.agent_critic_opt[no].apply_gradients(zip(ac_grad, self.agent_critics[no].trainable_variables))

            self.agent_critic_gard[no] = [grad.numpy() for grad in ac_grad]
            self.agent_critic_weights[no] = [var.numpy() for var in self.agent_critics[no].trainable_variables]
            # 1.3 训练actor网络 train actor
            with tf.GradientTape() as tape:
                # 计算当前的动作
                actions = self.agent_actors[no]([state_map, total_data_state, done_data_state, band])
                # 计算对应的新的奖励
                new_r = self.agent_critics[no]([state_map, total_data_state, done_data_state, actions[0], actions[1], band])
                # 原始损失是奖励的均值
                reward_loss = tf.reduce_mean(new_r)

                # 添加正则化项
                # 确保old_weights已经包含了上一个训练周期的权重
                regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
                                                for current_weight, old_weight in zip(self.agent_actor_weights[no], self.agent_actor_old_weights[no])])
                regularization_loss = mu * regularization_term

                # 计算总损失
                aa_loss = reward_loss + regularization_loss

            # 获取梯度并应用
            aa_grad = tape.gradient(aa_loss, self.agent_actors[no].trainable_variables)

            self.agent_actor_old_gard[no] = [grad.numpy() for grad in aa_grad]
            self.agent_actor_old_weights[no] = [var.numpy() for var in self.agent_actors[no].trainable_variables]
            self.agent_actor_opt[no].apply_gradients(zip(aa_grad, self.agent_actors[no].trainable_variables))

            self.agent_actor_gard[no] = [grad.numpy() for grad in aa_grad]
            self.agent_actor_weights[no] = [var.numpy() for var in self.agent_actors[no].trainable_variables]

            # 总结UAV的actor和critic网络的训练平均误差信息 summary info
            self.summaries['agent%s-critic_loss' % no] = ac_loss
            self.summaries['agent%s-actor_loss' % no] = aa_loss

        # 2. center replay
        # 若经验缓冲区长度 < 批大小（经验缓冲区中样本数量不足）
        if len(self.center_memory) < self.batch_size:
            # 跳过center replay training
            return
        # 对经验缓冲区内的数据进行采样
        # 采样规则：最新的0.25*batch_size条经验数据+最新的2*batch_size条经验数据中随机抽取0.75*batch_szie条
        center_samples = self.center_memory[-int(self.batch_size * self.sample_prop):] + random.sample(self.center_memory[-self.batch_size * 2:], int(self.batch_size * (1 - self.sample_prop)))
        # 2.1 从采样到的经验数据中提取并整理各类数据
        done_buffer_list = np.vstack([sample[0][0] for sample in center_samples])  # 将多条经验数据中的“center状态中的所有UAV计算完成数据”，堆叠构成一个新的数组
        pos_list = np.vstack([sample[0][1] for sample in center_samples])  # 将多条经验数据中的“center状态中的所有UAV位置坐标”，堆叠构成一个新的数组
        bandvec_act = np.vstack([sample[1] for sample in center_samples])  # 将多条经验数据中的“center动作中的带宽分配bandwidth_out”，堆叠构成一个新的数组
        c_reward = tf.expand_dims([sample[2] for sample in center_samples], axis=-1)  # 将多条经验数据中的"（env.step更新后）center的reward"，堆叠构成一个新的数组（这个reward是利用所有sensor求均值得到的）
        # new states
        new_done_buffer_list = np.vstack([sample[3][0] for sample in center_samples])  # 将多条经验数据中的“（env.step更新后）center状态中的所有UAV计算完成数据”，堆叠构成一个新的数组
        new_pos_list = np.vstack([sample[3][1] for sample in center_samples])  # 将多条经验数据中的“（env.step更新后）center状态中的所有UAV位置坐标”，堆叠构成一个新的数组
        # 根据（env.step更新后）下一时隙的状态预测下一时隙动作和reward（这个reward是利用target_center_critic网络得到的） next actions & reward
        new_c_actions = self.target_center_actor.predict([new_done_buffer_list, new_pos_list])
        cq_future = self.target_center_critic.predict([new_done_buffer_list, new_pos_list, new_c_actions])
        c_target_qs = c_reward + cq_future * self.gamma
        # 总结云中心的Q值
        self.summaries['cq_val'] = np.average(c_reward[0])


            # # 1.3 训练actor网络 train actor
            # with tf.GradientTape() as tape:
            #     # 计算当前的动作
            #     actions = self.agent_actors[no]([state_map, total_data_state, done_data_state, band])
            #     # 计算对应的新的奖励
            #     new_r = self.agent_critics[no]([state_map, total_data_state, done_data_state, actions[0], actions[1], band])
            #     # 原始损失是奖励的均值
            #     reward_loss = tf.reduce_mean(new_r)

            #     # 添加正则化项
            #     # 确保old_weights已经包含了上一个训练周期的权重
            #     regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
            #                                     for current_weight, old_weight in zip(self.agent_actor_weights[no], self.agent_actor_old_weights[no])])
            #     regularization_loss = mu * regularization_term

            #     # 计算总损失
            #     aa_loss = reward_loss + regularization_loss

            # # 获取梯度并应用
            # aa_grad = tape.gradient(aa_loss, self.agent_actors[no].trainable_variables)
            # 获取梯度并应用


        # 2.2 训练center critic网络 train center critic
        # 2.2 训练center critic网络 train center critic
        with tf.GradientTape() as tape:  # 创建梯度记录环境（梯度带：自动求导的记录器）
            tape.watch(self.center_critic.trainable_variables)  # 跟踪指定的tensor变量，即将tensor变量加入自动求导（GradientTape默认只对tf.Variable类型的变量进行监控）
            cq_values = tf.cast(self.center_critic([done_buffer_list, pos_list, bandvec_act]), dtype=tf.float32)
            lcc_loss = tf.reduce_mean(tf.math.square(cq_values - tf.cast(c_target_qs, dtype=tf.float32)))  # 计算平均误差

            # 添加正则化项
            regularization_term = tf.add_n([tf.nn.l2_loss(current_weight - old_weight)
                                            for current_weight, old_weight in zip(self.center_critic.trainable_variables, self.center_critic_old_weights)])
            regularization_loss = mu * regularization_term

            # 计算总损失
            cc_loss = lcc_loss + regularization_loss

        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)  # 自动计算所有梯度
        self.center_critic_old_gard= [grad.numpy() for grad in cc_grad]
        self.center_critic_old_weights= [var.numpy() for var in self.center_critic.trainable_variables]
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))  # 自动更新模型参数
        self.center_critic_gard = [grad.numpy() for grad in cc_grad]
        self.center_critic_weights = [var.numpy() for var in self.center_critic.trainable_variables]
        with tf.GradientTape() as tape:  # 创建梯度记录环境（梯度带：自动求导的记录器）
            tape.watch(self.center_critic.trainable_variables)  # 跟踪指定的tensor变量，即将tensor变量加入自动求导（GradientTape默认只对tf.Variable类型的变量进行监控）
            cq_values = tf.cast(self.center_critic([done_buffer_list, pos_list, bandvec_act]), dtype=tf.float64)
            lcc_loss = tf.reduce_mean(tf.math.square(cq_values - tf.cast(c_target_qs, dtype=tf.float64)))  # 计算平均误差

            # 添加正则化项
            regularization_term = tf.add_n([tf.nn.l2_loss(tf.cast(current_weight, dtype=tf.float64) - tf.cast(old_weight, dtype=tf.float64))
                                            for current_weight, old_weight in zip(self.center_critic.trainable_variables, self.center_critic_old_weights)])
            regularization_loss = mu * regularization_term

            # 计算总损失
            cc_loss = lcc_loss + regularization_loss

        cc_grad = tape.gradient(cc_loss, self.center_critic.trainable_variables)  # 自动计算所有梯度
        self.center_critic_old_gard = [grad.numpy() for grad in cc_grad]
        self.center_critic_old_weights = [var.numpy() for var in self.center_critic.trainable_variables]
        self.center_critic_opt.apply_gradients(zip(cc_grad, self.center_critic.trainable_variables))  # 自动更新模型参数
        self.center_critic_gard = [grad.numpy() for grad in cc_grad]
        self.center_critic_weights = [var.numpy() for var in self.center_critic.trainable_variables]

        # 2.3 训练center actor网络 train center actor
        with tf.GradientTape() as tape:  # 创建梯度记录环境（梯度带：自动求导的记录器）
            tape.watch(self.center_actor.trainable_variables)  # 跟踪指定的tensor变量，即将tensor变量加入自动求导（GradientTape默认只对tf.Variable类型的变量进行监控）
            c_act = self.center_actor([done_buffer_list, pos_list])
            ca_loss = tf.reduce_mean(tf.cast(self.center_critic([done_buffer_list, pos_list, c_act]), dtype=tf.float64))  # 计算平均误差

            # 添加正则化项
            regularization_term = tf.add_n([tf.nn.l2_loss(tf.cast(current_weight, dtype=tf.float64) - tf.cast(old_weight, dtype=tf.float64))
                                            for current_weight, old_weight in zip(self.center_actor.trainable_variables, self.center_actor_old_weights)])
            regularization_loss = mu * regularization_term

            # 计算总损失
            total_ca_loss = ca_loss + regularization_loss

        ca_grad = tape.gradient(total_ca_loss, self.center_actor.trainable_variables)  # 自动计算所有梯度
        self.center_actor_old_gard = [grad.numpy() for grad in ca_grad]
        self.center_actor_old_weights = [var.numpy() for var in self.center_actor.trainable_variables]
        self.center_actor_opt.apply_gradients(zip(ca_grad, self.center_actor.trainable_variables))  # 自动更新模型参数
        self.center_actor_gard = [grad.numpy() for grad in ca_grad]
        self.center_actor_weights = [var.numpy() for var in self.center_actor.trainable_variables]
        # print(ca_loss)
        # 总结center的actor和critic网络的训练平均误差信息 summary info
        self.summaries['center-critic_loss'] = cc_loss
        self.summaries['center-actor_loss'] = ca_loss

    # 保存模型
    def save_model(self, episode, time_str):
        # 保存所有sensor的actor和critic模型
        for i in range(self.sensor_num):
            self.sensor_actors[i].save('logs/models/{}/sensor-actor-{}_episode{}.h5'.format(time_str, i, episode))
            self.sensor_critics[i].save('logs/models/{}/sensor-critic-{}_episode{}.h5'.format(time_str, i, episode))
        # 保存所有UAV的actor和critic模型
        for i in range(self.agent_num):
            self.agent_actors[i].save('logs/models/{}/agent-actor-{}_episode{}.h5'.format(time_str, i, episode))
            self.agent_critics[i].save('logs/models/{}/agent-critic-{}_episode{}.h5'.format(time_str, i, episode))
        # 保存center的actor和critic模型
        self.center_actor.save('logs/models/{}/center-actor_episode{}.h5'.format(time_str, episode))
        self.center_critic.save('logs/models/{}/center-critic_episode{}.h5'.format(time_str, episode))

    # 训练模型
    # @tf.function
    def train(self, max_epochs=2000, max_step=500, up_freq=8, render=False, render_freq=1, FL=False, FL_omega=0.5, anomaly_edge=False):
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')  # 格式化日期时间参数（年月日-时分秒）
        train_log_dir = 'logs/fit/' + cur_time  # 保存tensorboard画图所需数据（.v2格式文件）的地址
        env_log_dir = 'logs/env/env' + cur_time  # 保存MEC环境示意图（sensors和UAVs的位置）图片的地址
        record_dir = 'logs/records/' + cur_time  # 保存记录（finish_length, finish_size, sensor_ages）的地址
        os.mkdir(env_log_dir)  # 根据地址，创建文件夹 os.mkdir()以数字权限模式创建目录
        os.mkdir(record_dir)  # 根据地址，创建文件夹
        summary_writer = tf.summary.create_file_writer(train_log_dir)  # 创建summary，即tensorboard的记录文件（tensorboard使用①）
        # tf.summary.trace_on(graph=True, profiler=True)
        os.makedirs('logs/models/' + cur_time)  # 根据地址，创建文件夹 os.makedirs()递归创建目录
        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        finish_length = []
        finish_size = []
        sensor_ages = []
        average_age = []
        fairness_index = []

        anomaly_step = 6000
        anomaly_agent = self.agent_num - 1

        # if anomaly_edge:
        #     anomaly_step = np.random.randint(int(max_epochs * 0.5), int(max_epochs * 0.75))
        #     anomaly_agent = np.random.randint(self.agent_num)
        # summary_record = []

        # 若epoch < epoch最大值
        while epoch < max_epochs:
            print('epoch%s' % epoch)  # 打印当前epoch值
            # if anomaly_edge and (epoch == anomaly_step):
            #     self.agents[anomaly_agent].movable = False

            # 若①需要画图（MEC环境示意图），且②每隔20个epoch
            if render and (epoch % 20 == 1):
                self.env.render(env_log_dir, epoch, True)  # 调用画图函数（MEC环境示意图），并保存到env_log_dir地址下
                # sensor_states.append(self.env.DS_state)
            # for critic_id, gradients in self.agent_critic_gard.items():
            #             print(f"Critic ID: {critic_id}")
            #             if gradients is not None:
            #                 for i, grad in enumerate(gradients):
            #                     if grad is not None:
            #                         print(f"  cGradient for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  cGradient for layer {i}: None")
            #             else:
            #                 print("  No cgradients recorded")
            # print('------------------------------')
            # for actor_id, gradients in self.agent_actor_gard.items():
            #             print(f"actor ID: {actor_id}")
            #             if gradients is not None:
            #                 for i, grad in enumerate(gradients):
            #                     if grad is not None:
            #                         print(f"  ac Gradient for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  ac Gradient for layer {i}: None")
            #             else:
            #                 print("  No acgradients recorded")
            # print('------------------------------')
            # for actor_id, gradients in self.sensor_actor_gard.items():
            #             print(f"actor ID: {actor_id}")
            #             if gradients is not None:
            #                 for i, grad in enumerate(gradients):
            #                     if grad is not None:
            #                         print(f"  sensor_actor_gard for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  sensor_actor_gard for layer {i}: None")
            #             else:
            #                 print(" sensor_actor_s recorded")
            # print('------------------------------')
            # for actor_id, weights in self.sensor_critic_gard.items():
            #             print(f"actor ID: {actor_id}")
            #             if weights is not None:
            #                 for i, grad in enumerate(weights):
            #                     if grad is not None:
            #                         print(f"  sensor_critic_gard for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  sensor_critic_gard for layer {i}: None")
            #             else:
            #                 print("  No sensor_critic_gard recorded")
            # print('------------------------------')

            # 若step >= step最大值
            if steps >= max_step:
                # self.env.world.finished_data = []
                episode += 1  # episode+1
                # epsilon调度程序
                self.epsilon = self.epsilon * 0.9  # **
                if self.epsilon < 0.05:  # **
                    self.epsilon = 0.05  # **
                # self.env.reset()
                # 对所有sensor循环
                for n in self.sensor_memory.keys():
                    del self.sensor_memory[n][0:-self.batch_size * 2]  # 只保留sensor_memory字典（sensors经验缓冲区）中最新的batch_size * 2条经验
                # 对所有UAV循环
                for m in self.agent_memory.keys():
                    del self.agent_memory[m][0:-self.batch_size * 2]  # 只保留agent_memory字典（UAVs经验缓冲区）中最新的batch_size * 2条经验
                del self.center_memory[0:-self.batch_size * 2]  # 只保留center_memory字典（center经验缓冲区）中最新的batch_size * 2条经验
                print('episode {}: {} total reward, {} steps, {} epochs'.format(episode, total_reward /  steps, steps, epoch))

                # 对所有UAV进行循环
                for agent in self.agents:
                    agent.t_distance = 0  # 将上一个时隙的UAV累计移动距离清零 **
                    agent.e_distance = 0  # 将上一个episode的UAV累计移动距离清零 **

                with summary_writer.as_default():  # 向summary中喂入需要监听的数据（tensorboard使用②）
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)  # tf.summary.scalar()，记录器将参数写入记录文件中
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=train_log_dir)

                summary_writer.flush()  # 关闭summary记录文件（tensorboard使用③）
                self.save_model(episode, cur_time)  # 保存模型
                steps = 0  # 重置step
                total_reward = 0  # 重置total_reward

            cur_reward, cur_average_age, cur_fairness_index = self.actor_act(epoch)  # 所有UAV和center得动作（包含了env.step环境更新！）
            # print('episode-%s reward:%f' % (episode, cur_reward))
            self.replay()  # 经验回放
            finish_length.append(len(self.env.world.finished_data))  # 每个epoch结束后，计算结束数据包的数量（累计值，会越来越多）
            finish_size.append(sum([data[0] for data in self.env.world.finished_data]))  # 每个epoch结束后，计算结束数据包的总数据量（累计值，会越来越多）
            sensor_ages.append(list(self.env.world.sensor_age.values()))  # 每个epoch结束后，所有sensor的age列表（不是越来越多）
            average_age.append(self.env._get_age())
            fairness_index.append(self.env._get_fairness_index)


            # summary_record.append(self.summaries)

            # 联邦更新 and 更新target update target
            # 每当epoch达到了联邦学习中更新targets的频率
            if epoch % up_freq == 1:
                print('update targets, finished data: {}'.format(len(self.env.world.finished_data)))  # 打印“更新target”以及当前epoch下计算结束数据包的数量

                # finish_length.append(len(self.env.world.finished_data))
                # ①联邦更新
                if FL and epoch>100:
                    merge_fl(self.agent_actors,self.agent_actor_gard,self.agent_actor_old_gard,self.agent_actor_weights,self.agent_actor_old_weights,0.1,1,0.5)
                    merge_fl(self.agent_critics,self.agent_critic_gard,self.agent_critic_old_gard,self.agent_critic_weights,self.agent_critic_old_weights,0.1,1,0.5)
                    merge_fl(self.sensor_actors,self.sensor_actor_gard,self.sensor_actor_old_gard,self.sensor_actor_weights,self.sensor_actor_old_weights,0.1,1,0.5)
                    merge_fl(self.sensor_critics,self.sensor_critic_gard,self.sensor_critic_old_gard,self.sensor_critic_weights,self.sensor_critic_old_weights,0.1,1,0.5)
                    # fedsgd(self.agent_actors)
                    # fedsgd(self.agent_critics)
                    # fedsgd(self.sensor_actors)
                    # fedsgd(self.sensor_critics)
                    # merge_fl(self.agent_actors, FL_omega)
                    # merge_fl(self.agent_critics, FL_omega)
                    # merge_fl(self.sensor_actors, FL_omega)
                    # merge_fl(self.sensor_critics, FL_omega)


                # ②更新atrget363                       
                # 对所有sensor进行循环
                for i in range(self.sensor_num):
                    update_target_net(self.sensor_actors[i], self.target_sensor_actors[i], self.tau)  # 更新target sensor actor网络权重
                    update_target_net(self.sensor_critics[i], self.target_sensor_critics[i], self.tau)  # 更新target sensor critic网络权重
                # 对所有UAV进行循环
                for i in range(self.agent_num):
                    update_target_net(self.agent_actors[i], self.target_agent_actors[i], self.tau)  # 更新target agent actor网络权重
                    update_target_net(self.agent_critics[i], self.target_agent_critics[i], self.tau)  # 更新target agent critic网络权重
                update_target_net(self.center_actor, self.target_center_actor, self.tau)  # 更新target center critic网络权重
                update_target_net(self.center_critic, self.target_center_critic, self.tau)  # 更新target center critic网络权重

            total_reward += cur_reward  # 对每个epoch中“所有UAV的平均reward”进行累加
            steps += 1
            epoch += 1

            # tensorboard
            with summary_writer.as_default():  # 向summary中喂入需要监听的数据（tensorboard使用②）
                # 若center memory字典（center经验缓冲区）中的长度 > batch_size
                if len(self.center_memory) > self.batch_size:
                    tf.summary.scalar('Loss/center_actor_loss', self.summaries['center-actor_loss'], step=epoch)  # tf.summary.scalar()，记录器将参数写入记录文件中
                    tf.summary.scalar('Loss/center_critic_loss', self.summaries['center-critic_loss'], step=epoch)
                    tf.summary.scalar('Stats/cq_val', self.summaries['cq_val'], step=epoch)
                    # 对所有sensor进行循环
                    for s_acount in range(self.sensor_num):
                        tf.summary.scalar('Sensor_Stats/sensor%s_actor_loss' % s_acount, self.summaries['sensor%s-actor_loss' % s_acount], step=epoch)
                        tf.summary.scalar('Sensor_Stats/sensor%s_critic_loss' % s_acount, self.summaries['sensor%s-critic_loss' % s_acount], step=epoch)
                    # 对所有UAV进行循环
                    for acount in range(self.agent_num):
                        tf.summary.scalar('Stats/agent%s_actor_loss' % acount, self.summaries['agent%s-actor_loss' % acount], step=epoch)
                        tf.summary.scalar('Stats/agent%s_critic_loss' % acount, self.summaries['agent%s-critic_loss' % acount], step=epoch)
                        tf.summary.scalar('Stats/agent%s_e_distance' % acount,self.summaries['agent%s-e_distance' % acount], step=epoch)  # ***
                        tf.summary.scalar('Stats/agent%s_total_distance' % acount,self.summaries['agent%s-total_distance' % acount], step=epoch)  # ***
                tf.summary.scalar('Main/UAVs_total_distance', self.UAVs_total_distance, step=epoch)  # ***
                tf.summary.scalar('Main/step_reward', cur_reward, step=epoch)
                tf.summary.scalar('Main/step_average_age', cur_average_age, step=epoch)
                tf.summary.scalar('Main/step_fairness_index', cur_fairness_index, step=epoch)
                tf.summary.scalar('Main/step_finish_length', finish_length[-1], step=epoch)
                tf.summary.scalar('Main/step_finish_size', finish_size[-1], step=epoch)
                tf.summary.scalar('Main/step_worst_sensor_age', sorted(list(self.env.world.sensor_age.values()))[-1], step=epoch)

        summary_writer.flush()  # 关闭summary记录文件（tensorboard使用③）

        # （在epoch达到上限，训练的循环结束后）保存最终模型 save final model
        self.save_model(episode, cur_time)
        # 保存相关数据信息
        sio.savemat(record_dir + '/data.mat',
                    {'finish_len': finish_length,
                     'finish_data': finish_size,
                     'ages': sensor_ages,
                     'average_age': average_age,
                     'fairness_index': fairness_index})
        # sio.savemat(record_dir + '/data.mat',
        #             {'finish_len': finish_length,
        #              'finish_data': finish_size,
        #              'sensor_map': sensor_map,
        #              'sensor_list': sensor_pos_list,
        #              'sensor_state': sensor_states,
        #              'agentpos': agent_pos,
        #              'agentoff': agent_off,
        #              'agentexe': agent_exe,
        #              'agenttran': agent_trans,
        #              'agentbuff': agent_donebuff,
        #              'agentexebuff': exebuff,
        #              'agentdonebuff': donebuff,
        #              'agentband': agent_band,
        #              'anomaly': [anomaly_step,
        #                          anomaly_agent]})
        # with open(record_dir + '/record.json', 'w') as f:
        #     json.dump(summary_record, f)

        # 处理制作gif图的图片
        self.env.render(env_log_dir, epoch, True)  # 在训练结束时（epoch=其最大值），调用画图函数（MEC环境示意图），并保存到env_log_dir地址下
        img_paths = glob.glob(env_log_dir + '/*.png')  # 获取指定目录下的所有png格式的图片 glob.glob()方法 返回所有匹配文件的路径列表
        img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))  # 根据MEC环境示意图对应的epoch值，升序排序 lambda是匿名函数，x是输入，“：”后是匿名函数的内容

        # 制作gif图片
        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))  # 所有MEC示意图RGB内容的列表 imageio.imread()读取图片的RGB内容
        imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=15)  # 生成GIF图片
