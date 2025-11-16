import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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

    # （输入：源数据缓冲区，输出：sensor卸载）
    sensor_mlp = tf.transpose(sensor_buffer_state, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置） 注意：sensor_buffer的shape为(?, 2, max_buffer_size)
    sensor_mlp = layers.Dense(1, activation='relu')(sensor_mlp)  # 全连接层 神经元数量为1
    sensor_mlp = tf.squeeze(sensor_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
    computing_dev = layers.Dense(1, activation='sigmoid')(sensor_mlp)  # 全连接层 神经元数量为2

    model = keras.Model(inputs=[sensor_buffer_state], outputs=[computing_dev])  # 构造模型
    return model

# agent actor网络
# 输入：input_dim_list=[state_map_shape, buffstate_shape, buffstate_shape, band_shape]，即state map,total_buffer,done_buffer,operation,bandwidth
# 输出：move_out,op_dist即move,operation
# agent actor net: inputs state map,pos,buffer,operation,bandwidth; outputs: move,operation
def agent_actor(input_dim_list, cnn_kernel_size, move_r):
    # Input
    state_map = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为state_map_shape
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为buffstate_shape
    done_buffer = keras.Input(shape=input_dim_list[2])  # 输入层3 shape为buffstate_shape
    bandwidth = keras.Input(shape=input_dim_list[3])  # 输入层4 shape为band_shape

    # CNN for map（输入：状态地图，输出：UAV移动）
    cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(state_map)  # 卷积层1 （卷积核个数，卷积核尺寸，激活函数，补零策略）
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(cnn_map)  # 平均池化层1 （池化层大小）
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)  # Dropout层1 AlphaDropout将在Dropout之后保持原始均值和方差
    move_out = layers.Dense(1, activation='relu')(cnn_map)  # 全连接层1 （神经元数量，激活函数）

    # （输入：待计算数据缓冲区，输出：UAV执行计算）
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置） 注意：total_buffer的shape为(?, 2, max_buffer_size)
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)  # 全连接层2 神经元数量为1
    total_mlp = tf.transpose(total_mlp, perm=[0, 2, 1])  # 对数组进行转置（再把最内侧的两维转置回来）
    exe_op = layers.Dense(input_dim_list[1][1], activation='softmax')(total_mlp)  # 全连接层3 神经元数量为input_dim_list[1][1]（应该是待计算数据包的数量）

    # （输入：计算完成数据缓冲区+带宽？？？，输出：UAV卸载）
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置）
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)  # 全连接层3 神经元数量为1
    done_mlp = tf.transpose(done_mlp, perm=[0, 2, 1])  # 对数组进行转置（再把最内侧的两维转置回来）
    bandwidth_in = tf.expand_dims(bandwidth, axis=-1)  # 对bandwidth输入层增加一个维度
    bandwidth_in = layers.Dense(1, activation='relu')(bandwidth_in)  # 全连接层4 神经元数量为1
    done_mlp = layers.concatenate([done_mlp, bandwidth_in], axis=-1)  # 对数组进行拼接
    off_op = layers.Dense(input_dim_list[2][1], activation='softmax')(done_mlp)  # 全连接层5 神经元数量为input_dim_list[2][1]（应该是计算完成数据包的数量）

    op_dist = layers.concatenate([exe_op, off_op], axis=1)  # 对数组进行拼接
    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, bandwidth], outputs=[move_out, op_dist])  # 构造模型
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_aa))
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
    sensor_buffer = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为sensor_buffer经过roi pooling
    computing_dev = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为computing_dev

    # （输入：源数据缓冲区，输出：sensor卸载）
    sensor_mlp = tf.transpose(sensor_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置） 注意：sensor_buffer的shape为(?, 2, 5)
    sensor_mlp = layers.Dense(1, activation='relu')(sensor_mlp)  # 全连接层 神经元数量为1
    sensor_mlp = tf.squeeze(sensor_mlp, axis=-1)  # 从张量shape中移除大小为1的维度

    r_out = layers.concatenate([sensor_mlp, computing_dev])  # layers.concatenate()在指定的维度拼接数组

    r_out = layers.Dense(1, activation='relu')(r_out)  # 全连接层 神经元数量为1

    model = keras.Model(inputs=[sensor_buffer, computing_dev], outputs=r_out, name='sensor_critic_net')  # 创建模型
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_aa))
    return model

# agent critic网络
# 输入：input_dim_list = [state_map_shape, buffstate_shape, buffstate_shape, movemap_shape, op_shape, band_shape]
# 输出：reward_out
def agent_critic(input_dim_list, cnn_kernel_size):
    # Input
    state_map = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为state_map_shape
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])  # 输入层2 shape为buffstate_shape
    done_buffer = keras.Input(shape=input_dim_list[2])  # 输入层3 shape为buffstate_shape
    move = keras.Input(shape=input_dim_list[3])  # 输入层4 shape为movemap_shape
    onehot_op = keras.Input(shape=input_dim_list[4])  # 输入层5 shape为op_shape
    bandwidth = keras.Input(shape=input_dim_list[5])  # 输入层6 shape为band_shape

    # map CNN
    # merge last dim
    map_cnn = layers.Dense(1, activation='relu')(state_map)  # 全连接层1 神经元数量为1
    map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)  # 卷积层1 （卷积核个数，卷积核大小，激活函数，补零策略）
    map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)  # 平均池化层1 （池化核大小）
    map_cnn = layers.AlphaDropout(0.2)(map_cnn)  # 保持原始均值和方差的 Dropout
    # map_cnn = layers.Conv2D(input_dim_list[0][2], kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    # map_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(map_cnn)
    # map_cnn = layers.Dropout(0.2)(map_cnn)
    map_cnn = layers.Flatten()(map_cnn)  # 将数据压成一维数据
    map_cnn = layers.Dense(2, activation='relu')(map_cnn)  # 全连接层2 神经元数量为2

    # mlp
    # pos_mlp = layers.Dense(1, activation='relu')(position)
    band_mlp = layers.Dense(1, activation='relu')(bandwidth)  # 全连接层3 神经元数量为1
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置）
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)  # 全连接层4 神经元数量为1
    total_mlp = tf.squeeze(total_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
    total_mlp = layers.Dense(2, activation='relu')(total_mlp)  # 全连接层5 神经元数量为 2
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置）
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)  # 全连接层6 神经元数量为1
    done_mlp = tf.squeeze(done_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
    done_mlp = layers.Dense(2, activation='relu')(done_mlp)  # 全连接层7 神经元数量为2

    move_mlp = layers.Flatten()(move)  # 将数据压成一维数据
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)  # 全连接层8 神经元数量为1
    onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)  # 全连接层9 神经元数量为1
    onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)  # 从张量shape中移除大小为1的维度

    all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)  # 数组拼接
    reward_out = layers.Dense(1, activation='relu')(all_mlp)  # 全连接层10 神经元数量为1

    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)  # 创建模型
    # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model


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
        self.sample_prop = 1 / 4  # replay training采样中，使用最新经验数据的比例

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
        for no in range(6):
            # 初始化权重为零
            zero_weights = [tf.zeros_like(var, dtype=tf.float32) for var in self.agent_critics[no].trainable_variables]
            self.agent_critic_old_weights[no] = zero_weights
            self.agent_actor_old_weights[no] = zero_weights
            self.agent_critic_weights[no] = zero_weights
            self.agent_actor_weights[no] = zero_weights
        # Initialize self.center_actor_old_weights and self.center_critic_old_weights to zero
        self.center_actor_old_weights = [tf.Variable(tf.zeros_like(weight), trainable=False) for weight in self.center_actor.trainable_variables]
        self.center_critic_old_weights = [tf.Variable(tf.zeros_like(weight), trainable=False) for weight in self.center_critic.trainable_variables]


        for no in range(30):
            # 初始化权重为零
            zero_weights = [tf.zeros_like(var, dtype=tf.float32) for var in self.sensor_critics[no].trainable_variables]
            self.sensor_critic_old_weights[no] = zero_weights
            self.sensor_actor_old_weights[no] = zero_weights
            self.sensor_critic_weights[no] = zero_weights
            self.sensor_actor_weights[no] = zero_weights


    def actor_act(self, epoch, training=True):
        if not training:
            # 在测试模式下总是使用模型进行预测
            sensor_actions = []
            agent_actions = []
            for i, sensor in enumerate(self.sensors):
                sensor_data_state = tf.expand_dims(sensor.get_sensor_data(), axis=0)
                sensor_action = self.sensor_actors[i].predict([sensor_data_state])
                sensor_actions.append(sensor_action)
            state_maps = []
            total_data_states = []
            done_data_states = []
            bands = []
            for i, agent in enumerate(self.agents):
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                agent_action = self.agent_actors[i].predict([state_map, total_data_state, done_data_state, band])
                agent_actions.append(agent_action)
                state_maps.append(state_map)
                total_data_states.append(total_data_state)
                done_data_states.append(done_data_state)
                bands.append(band)
            
            # Center actions
            done_buffer_list, pos_list = self.env.get_center_state()
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            pos_list = tf.expand_dims(pos_list, axis=0)
            center_action = self.center_actor.predict([done_buffer_list, pos_list])

            # Environment interaction
            new_state_maps, new_rewards, average_age, fairness_index, done, info = self.env.step(agent_actions, center_action, sensor_actions)
            
            return new_rewards[-1], average_age[-1], fairness_index[-1]

    def load_models(self, episode=24):
        # 加载所有UAV的actor模型
        for i in range(self.agent_num):
            model_path = f"C:/Users/25447/Desktop/HF/logs/models/20240804-222227/agent-actor-{i}_episode{episode}.h5"
            self.agent_actors[i] = tf.keras.models.load_model(model_path)

        # 如果有其他角色的模型，比如sensors或center，同样方式加载
        # 假设每个sensor和center也有相应的模型保存C:/Users/25447/Desktop/HF/logs/models/20240804-222227/agent-actor-0_episode1.h5
        for i in range(self.sensor_num):
            self.sensor_actors[i] = tf.keras.models.load_model(f"C:/Users/25447/Desktop/HF/logs/models/20240804-222227/sensor-actor-{i}_episode{episode}.h5")
            self.sensor_critics[i] = tf.keras.models.load_model(f"C:/Users/25447/Desktop/HF/logs/models/20240804-222227/sensor-critic-{i}_episode{episode}.h5")
        
        # 加载center的actor和critic模型
        self.center_actor = tf.keras.models.load_model(f"C:/Users/25447/Desktop/HF/logs/models/20240804-222227/center-actor_episode{episode}.h5")
        self.center_critic = tf.keras.models.load_model(f"C:/Users/25447/Desktop/HF/logs/models/20240804-222227/center-critic_episode{episode}.h5")
    
    def test(self, num_episodes=10):
        # 加载模型
        self.load_models()

        # 用于记录测试结果的变量
        total_rewards = []
        average_ages = []
        fairness_indexes = []

        for episode in range(num_episodes):
            # 环境重置
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # 根据当前状态选择动作
                action = self.actor_act(state, training=False)  # 确保actor_act方法有一个training标志或调整逻辑以用于测试
                next_state, reward, done, info = self.env.step(action)

                # 更新状态和累积奖励
                state = next_state
                total_reward += reward

            # 记录每个周期的结果
            total_rewards.append(total_reward)
            average_ages.append(self.env.get_average_age())  # 假设环境有这个方法
            fairness_indexes.append(self.env.get_fairness_index())  # 假设环境有这个方法

            print(f"Episode {episode + 1}: Reward = {total_reward}")

        # 输出测试周期的平均结果
        print(f"Average Reward: {sum(total_rewards) / len(total_rewards)}")
        print(f"Average Age: {sum(average_ages) / len(average_ages)}")
        print(f"Average Fairness Index: {sum(fairness_indexes) / len(fairness_indexes)}")

