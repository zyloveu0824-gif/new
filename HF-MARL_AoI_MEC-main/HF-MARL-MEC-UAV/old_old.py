import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Model
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

def discrete_circle_sample_count(n):
    # 初始化计数器和移动字典
    count = 0
    move_dict = {}
    # 遍历-半径到半径的范围
    for x in range(-n, n + 1):
        # 计算y坐标的范围
        y_l = int(np.floor(np.sqrt(n**2 - x**2)))
        for y in range(-y_l, y_l + 1):
            # 为每个坐标点生成一个移动向量并存储到字典
            move_dict[count] = np.array([y, x])
            # 计数器递增
            count += 1
    # 返回计数器和移动字典
    return (count), move_dict
def custom_loss(y_true, y_pred, w, w_t, mu):
    original_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)  # 或者是其他的原始损失函数
    regularization_loss = mu * tf.reduce_mean(tf.square(w - w_t))
    return original_loss + regularization_loss

# def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
#     # 计算预测值和真实值之间的误差
#     error = y_true - y_pred
#     # 判断误差是否在阈值范围内
#     cond = tf.abs(error) <= clip_delta
#     # 计算平方损失
#     squared_loss = 0.5 * tf.square(error)
#     # 计算Huber损失
#     quadratic_loss = 0.5 * tf.square(clip_delta) + clip_delta * (tf.abs(error) - clip_delta)
#     # 返回条件平均损失
#     return tf.math.reduce_mean(tf.where(cond, squared_loss, quadratic_loss))

def agent_actor(input_dim_list, cnn_kernel_size, move_r):
    # 定义神经网络输入层
    state_map = layers.Input(shape=input_dim_list[0])
    total_buffer = layers.Input(shape=input_dim_list[1])
    done_buffer = layers.Input(shape=input_dim_list[2])
    bandwidth = layers.Input(shape=input_dim_list[3])

    # 使用卷积层处理地图数据
    cnn_map = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(state_map)
    cnn_map = layers.AveragePooling2D(pool_size=int(input_dim_list[0][0] / (2 * move_r + 1)))(cnn_map)
    cnn_map = layers.AlphaDropout(0.2)(cnn_map)
    move_out = layers.Dense(1, activation='relu')(cnn_map)

    # 处理总缓冲区数据
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.transpose(total_mlp, perm=[0, 2, 1])
    exe_op = layers.Dense(input_dim_list[1][1], activation='softmax')(total_mlp)

    # 处理已完成缓冲区和带宽数据
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.transpose(done_mlp, perm=[0, 2, 1])
    bandwidth_in = tf.expand_dims(bandwidth, axis=-1)
    bandwidth_in = layers.Dense(1, activation='relu')(bandwidth_in)
    done_mlp = layers.concatenate([done_mlp, bandwidth_in], axis=-1)
    off_op = layers.Dense(input_dim_list[2][1], activation='softmax')(done_mlp)

    # 合并执行和卸载操作
    op_dist = layers.concatenate([exe_op, off_op], axis=1)
    
    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, bandwidth], outputs=[move_out, op_dist])
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_aa))

    return model
def center_actor(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])
    pos_list = keras.Input(shape=input_dim_list[1])

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)

    # pos list
    pos = layers.Dense(2, activation='relu')(pos_list)

    bandwidth_out = layers.concatenate([buffer_state, pos], axis=-1)
    # bandwidth_out = layers.AlphaDropout(0.2)(bandwidth_out)
    bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)
    bandwidth_out = tf.squeeze(bandwidth_out, axis=-1)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    bandwidth_out = layers.Softmax()(bandwidth_out)
    # bandwidth_out += 1 / (input_dim_list[2] * 5)
    # bandwidth_out = bandwidth_out / tf.reduce_sum(bandwidth_out, 1, keepdims=True)
    # bandwidth_out = bandwidth_out / tf.expand_dims(tf.reduce_sum(bandwidth_out, 1), axis=-1)

    model = keras.Model(inputs=[done_buffer_list, pos_list], outputs=bandwidth_out, name='center_actor_net')
    # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])

    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='softmax')(sensor_cnn)

    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='softmax')(agent_cnn)

    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn], axis=-1)
    # bandwidth_out = layers.Dense(input_dim_list[2], activation='softmax')(bandwidth_out)

    # model = keras.Model(inputs=[sensor_map, agent_map], outputs=bandwidth_out, name='center_actor_net')
    # # model.compile(loss=huber_loss, optimizer=keras.optimizers.Adam(learning_rate=self.lr_ca))
    return model

def agent_critic(input_dim_list, cnn_kernel_size):
    state_map = keras.Input(shape=input_dim_list[0])
    # position = keras.Input(shape=input_dim_list[1])
    total_buffer = keras.Input(shape=input_dim_list[1])
    done_buffer = keras.Input(shape=input_dim_list[2])
    move = keras.Input(shape=input_dim_list[3])
    onehot_op = keras.Input(shape=input_dim_list[4])
    bandwidth = keras.Input(shape=input_dim_list[5])

    # map CNN
    # merge last dim
    map_cnn = layers.Dense(1, activation='relu')(state_map)
    map_cnn = layers.Conv2D(1, kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    map_cnn = layers.AveragePooling2D(pool_size=cnn_kernel_size * 2)(map_cnn)
    map_cnn = layers.AlphaDropout(0.2)(map_cnn)
    # map_cnn = layers.Conv2D(input_dim_list[0][2], kernel_size=cnn_kernel_size, activation='relu', padding='same')(map_cnn)
    # map_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(map_cnn)
    # map_cnn = layers.Dropout(0.2)(map_cnn)
    map_cnn = layers.Flatten()(map_cnn)
    map_cnn = layers.Dense(2, activation='relu')(map_cnn)

    # mlp
    # pos_mlp = layers.Dense(1, activation='relu')(position)
    band_mlp = layers.Dense(1, activation='relu')(bandwidth)
    total_mlp = tf.transpose(total_buffer, perm=[0, 2, 1])
    total_mlp = layers.Dense(1, activation='relu')(total_mlp)
    total_mlp = tf.squeeze(total_mlp, axis=-1)
    total_mlp = layers.Dense(2, activation='relu')(total_mlp)
    done_mlp = tf.transpose(done_buffer, perm=[0, 2, 1])
    done_mlp = layers.Dense(1, activation='relu')(done_mlp)
    done_mlp = tf.squeeze(done_mlp, axis=-1)
    done_mlp = layers.Dense(2, activation='relu')(done_mlp)

    move_mlp = layers.Flatten()(move)
    move_mlp = layers.Dense(1, activation='relu')(move_mlp)
    onehot_mlp = layers.Dense(1, activation='relu')(onehot_op)
    onehot_mlp = tf.squeeze(onehot_mlp, axis=-1)

    all_mlp = layers.concatenate([map_cnn, band_mlp, total_mlp, done_mlp, move_mlp, onehot_mlp], axis=-1)
    reward_out = layers.Dense(1, activation='relu')(all_mlp)

    model = keras.Model(inputs=[state_map, total_buffer, done_buffer, move, onehot_op, bandwidth], outputs=reward_out)
    # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model


def center_critic(input_dim_list, cnn_kernel_size):
    done_buffer_list = keras.Input(shape=input_dim_list[0])
    pos_list = keras.Input(shape=input_dim_list[1])
    bandwidth_vec = keras.Input(shape=input_dim_list[2])

    # buffer
    buffer_state = layers.Dense(1, activation='relu')(done_buffer_list)
    buffer_state = tf.squeeze(buffer_state, axis=-1)
    buffer_state = layers.Dense(1, activation='relu')(buffer_state)
    buffer_state = tf.squeeze(buffer_state, axis=-1)

    # pos list
    pos = layers.Dense(1, activation='relu')(pos_list)
    pos = tf.squeeze(pos, axis=-1)

    # bandvec
    # band_in = layers.Dense(2, activation='relu')(bandwidth_vec)

    r_out = layers.concatenate([buffer_state, pos, bandwidth_vec])
    # r_out = layers.AlphaDropout(0.2)(r_out)
    r_out = layers.Dense(1, activation='relu')(r_out)
    model = keras.Model(inputs=[done_buffer_list, pos_list, bandwidth_vec], outputs=r_out, name='center_critic_net')
    # sensor_map = keras.Input(shape=input_dim_list[0])
    # agent_map = keras.Input(shape=input_dim_list[1])
    # bandwidth_vec = keras.Input(shape=input_dim_list[2])

    # # sensor map:cnn*2
    # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_map)
    # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # # sensor_cnn = layers.Conv2D(input_dim_list[0][2], cnn_kernel_size, activation='relu', padding='same')(sensor_cnn)
    # # sensor_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(sensor_cnn)
    # # sensor_cnn = layers.Dropout(0.2)(sensor_cnn)
    # sensor_cnn = layers.Flatten()(sensor_cnn)
    # sensor_cnn = layers.Dense(4, activation='relu')(sensor_cnn)

    # # agent map
    # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_map)
    # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # # agent_cnn = layers.Conv2D(input_dim_list[1][2], cnn_kernel_size, activation='relu', padding='same')(agent_cnn)
    # # agent_cnn = layers.MaxPooling2D(pool_size=cnn_kernel_size)(agent_cnn)
    # # agent_cnn = layers.Dropout(0.2)(agent_cnn)
    # agent_cnn = layers.Flatten()(agent_cnn)
    # agent_cnn = layers.Dense(4, activation='relu')(agent_cnn)

    # # add bandwidth
    # bandwidth_out = layers.concatenate([sensor_cnn, agent_cnn, bandwidth_vec], axis=-1)
    # bandwidth_out = layers.Dense(1, activation='relu')(bandwidth_out)

    # model = keras.Model(inputs=[sensor_map, agent_map, bandwidth_vec], outputs=bandwidth_out, name='center_critic_net')
    # # model.compile(loss=_huber_loss, optimizer=keras.optimizers.Adam(learning_rate=0.02))
    return model
def update_target_net(model, target, tau=0.8):
    # 获取模型和目标网络的权重
    weights = model.get_weights()
    target_weights = target.get_weights()

    # 更新目标网络的权重
    for i in range(len(target_weights)):
        # 将目标网络的权重更新为模型权重和原目标权重的组合
        target_weights[i] = weights[i] * (1 - tau) + target_weights[i] * tau
    target.set_weights(target_weights)



def circle_argmax(move_dist, move_r):
    # 获取概率分布中最高值的位置
    max_pos = np.argwhere(tf.squeeze(move_dist, axis=-1) == np.max(move_dist))
    # tf.squeeze 函数移除了 move_dist 的最后一个维度，便于处理

    # 计算这些位置与圆心（即原点）的距离
    pos_dist = np.linalg.norm(max_pos - np.array([move_r, move_r]), axis=1)
    # np.linalg.norm 计算向量的欧几里得距离，这里使用圆的半径 move_r 来定位圆心

    # 从最大值位置中选择离圆心最近的位置
    return max_pos[np.argmin(pos_dist)]
    # np.argmin 返回距离最小值的索引，用它来从 max_pos 中选取最佳位置


from tensorflow import keras

class MAACAgent(object):
    def __init__(self, env, tau, gamma, lr_aa, lr_ac, lr_ca, lr_cc, batch, epsilon=0.2):
        # 初始化环境和代理属性
        self.env = env
        self.agents = self.env.agents
        self.agent_num = self.env.agent_num
        self.index_dim = 2
        self.obs_r = self.env.obs_r

        # 定义输入维度
        self.state_map_shape = (self.obs_r * 2 + 1, self.obs_r * 2 + 1, self.index_dim)
        self.pos_shape = (2,)
        self.band_shape = (1,)
        self.buffstate_shape = (self.index_dim, self.env.max_buffer_size)
        self.buffer_list_shape = (self.agent_num, self.index_dim, self.env.max_buffer_size)
        self.pos_list_shape = (self.agent_num, 2)
        self.bandvec_shape = (self.env.agent_num,)
        self.op_shape = (self.index_dim, self.env.max_buffer_size)
        self.move_count, self.move_dict = discrete_circle_sample_count(self.env.move_r)
        self.movemap_shape = (self.env.move_r * 2 + 1, self.env.move_r * 2 + 1)
        self.epsilon = epsilon
        self.agent_actor_weights={}
        self.agent_actor_old_weights={}
        self.agent_critic_weights={}
        self.agent_critic_old_weights={}
        # self.sensor_actor_gard = {}
        # self.sensor_critic_gard = {}
        self.agent_actor_gard = {}
        self.agent_actor_old_gard = {}
        self.agent_critic_gard = {}
        self.agent_critic_old_gard = {}

        # 学习参数
        self.tau = tau
        self.cnn_kernel_size = 3
        self.gamma = gamma
        self.lr_aa = lr_aa
        self.lr_ac = lr_ac
        self.lr_ca = lr_ca
        self.lr_cc = lr_cc
        self.batch_size = batch
        self.global_weights = []
        # 初始化记忆库
        self.agent_memory = {}
        self.softmax_memory = {}
        self.center_memory = []
        self.sample_prop = 1 / 4

        # 初始化网络
        self.agent_actors = []
        self.center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        self.agent_critics = []
        self.center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)

        # 初始化目标网络
        self.target_agent_actors = []
        self.target_center_actor = center_actor([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        update_target_net(self.center_actor, self.target_center_actor, tau=0)
        self.target_agent_critics = []
        self.target_center_critic = center_critic([self.buffer_list_shape, self.pos_list_shape, self.bandvec_shape], self.cnn_kernel_size)
        update_target_net(self.center_critic, self.target_center_critic, tau=0)

        # 初始化优化器
        self.agent_actor_opt = []
        self.agent_critic_opt = []
        self.center_actor_opt = keras.optimizers.Adam(learning_rate=lr_ca)
        self.center_critic_opt = keras.optimizers.Adam(learning_rate=lr_cc)

        # 初始化其他属性
        self.summaries = {}

        # 为每个代理创建演员和评论家网络
        for i in range(self.env.agent_num):
            self.agent_critic_opt.append(keras.optimizers.Adam(learning_rate=lr_ac))
            self.agent_actor_opt.append(keras.optimizers.Adam(learning_rate=lr_aa))

            new_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            target_agent_actor = agent_actor([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.band_shape], self.cnn_kernel_size, self.env.move_r)
            update_target_net(new_agent_actor, target_agent_actor, tau=0)

            self.agent_actors.append(new_agent_actor)
            self.target_agent_actors.append(target_agent_actor)

            new_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            t_agent_critic = agent_critic([self.state_map_shape, self.buffstate_shape, self.buffstate_shape, self.movemap_shape, self.op_shape, self.band_shape], self.cnn_kernel_size)
            update_target_net(new_agent_critic, t_agent_critic, tau=0)
            self.agent_critics.append(new_agent_critic)
            self.target_agent_critics.append(t_agent_critic)



    def actor_act(self, epoch):
        # 随机数，用于决定是否进行探索
        tmp = random.random()
        # 当随机数大于epsilon且epoch大于等于16时，使用网络进行预测，否则进行探索
        if tmp >= 0 and epoch >= 16:
            # agent act
            agent_act_list = []
            softmax_list = []
            cur_state_list = []
            band_vec = np.zeros(self.agent_num)
            # 遍历每个代理
            for i, agent in enumerate(self.agents):
                # 获取代理的状态
                # actor = self.agent_actors[i]
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                # print('agent{}pos:{}'.format(i, pos))
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                # print('band{}'.format(agent.action.bandwidth))
                band_vec[i] = agent.action.bandwidth
                # 预测代理动作
                assemble_state = [state_map, total_data_state, done_data_state, band]
                # print(['agent%s' % i, sum(sum(state_map))])
                cur_state_list.append(assemble_state)
                # print(total_data_state.shape)
                action_output = self.agent_actors[i].predict(assemble_state)
                # 处理预测结果
                move_dist = action_output[0][0]
                sio.savemat('debug.mat', {'state': self.env.get_obs(agent), 'move': move_dist})
                # print(move_dist)
                # print(move_dist.shape)
                op_dist = action_output[1][0]
                # print(op_dist.shape)
                # move_ori = np.unravel_index(np.argmax(move_dist), move_dist.shape)
                move_ori = circle_argmax(move_dist, self.env.move_r)
                move = [move_ori[1] - self.env.move_r, move_ori[0] - self.env.move_r]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.argmax(op_dist[0])] = 1
                offloading[np.argmax(op_dist[1])] = 1
                # 生成Softmax动作分布
                move_softmax = np.zeros(move_dist.shape)
                op_softmax = np.zeros(self.buffstate_shape)

                move_softmax[move_ori] = 1
                op_softmax[0][np.argmax(op_dist[0])] = 1
                op_softmax[1][np.argmax(op_dist[1])] = 1
                # 将Softmax动作分布加入列表
                move_softmax = tf.expand_dims(move_softmax, axis=0)
                # move_softmax = tf.expand_dims(move, axis=0)
                op_softmax = tf.expand_dims(op_softmax, axis=0)

                agent_act_list.append([move, execution, offloading])
                softmax_list.append([move_softmax, op_softmax])
            # print(agent_act_list)
            # center act
            # 预测中心执行者的动作
            done_buffer_list, pos_list = self.env.get_center_state()
            done_buffer_list = tf.expand_dims(done_buffer_list, axis=0)
            # print(done_buffer_list)
            pos_list = tf.expand_dims(pos_list, axis=0)
            band_vec = tf.expand_dims(band_vec, axis=0)
            new_bandvec = self.center_actor.predict([done_buffer_list, pos_list])
            # print('new_bandwidth{}'.format(new_bandvec[0]))
            # 执行步骤并获取新状态
            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec[0])
            new_done_buffer_list, new_pos_list = self.env.get_center_state()
            new_done_buffer_list = tf.expand_dims(new_done_buffer_list, axis=0)
            new_pos_list = tf.expand_dims(new_pos_list, axis=0)

            # record memory
            for i, agent in enumerate(self.agents):
                # 获取新状态
                state_map = new_state_maps[i]
                # print(['agent%s' % i, sum(sum(state_map))])
                # pos = agent.position
                total_data_state = agent.get_total_data()
                done_data_state = agent.get_done_data()
                state_map = tf.expand_dims(self.env.get_obs(agent), axis=0)
                # pos = tf.expand_dims(agent.position, axis=0)
                total_data_state = tf.expand_dims(agent.get_total_data(), axis=0)
                done_data_state = tf.expand_dims(agent.get_done_data(), axis=0)
                band = tf.expand_dims(agent.action.bandwidth, axis=0)
                new_states = [state_map, total_data_state, done_data_state, band]
                # 将代理的当前状态、动作、奖励、新状态和完成状态保存到memory中
                if agent.no in self.agent_memory.keys():
                    self.agent_memory[agent.no].append([cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]])
                else:
                    self.agent_memory[agent.no] = [[cur_state_list[i], softmax_list[i], new_rewards[i], new_states, done[i]]]
            # 保存center的memory
            self.center_memory.append([[done_buffer_list, pos_list], new_bandvec, new_rewards[-1], [new_done_buffer_list, new_pos_list]])
        # 如果需要进行探索
        else:
            # random action
            # agents
            # 随机选择代理的动作
            agent_act_list = []
            for i, agent in enumerate(self.agents):
                move = random.sample(list(self.move_dict.values()), 1)[0]
                execution = [0] * agent.max_buffer_size
                offloading = [0] * agent.max_buffer_size
                execution[np.random.randint(agent.max_buffer_size)] = 1
                offloading[np.random.randint(agent.max_buffer_size)] = 1
                agent_act_list.append([move, execution, offloading])
            # center
            # 随机选择中心控制者的动作
            new_bandvec = np.random.rand(self.agent_num)
            new_bandvec = new_bandvec / np.sum(new_bandvec)
            # 执行动作并获取新状态
            new_state_maps, new_rewards, done, info = self.env.step(agent_act_list, new_bandvec)

        return new_rewards[-1]

    

    def load_models(self, episode):
        # 加载所有UAV的actor模型C:\Users\ZY\Desktop\HF-MARL_AoI_MEC-main (1) (2)\logs\models\20240624-150211\agent-actor-0_episode1.h5
        for i in range(self.agent_num):
            model_path = f"C:/Users/ZY/Desktop/HF-MARL_AoI_MEC-main (1) (2)/logs/models/20240624-150211/agent-actor-{i}_episode{episode}.h5"
            self.agent_actors[i] = tf.keras.models.load_model(model_path)
            self.agent_critics[i] = tf.keras.models.load_model(model_path)

        # 如果有其他角色的模型，比如sensors或center，同样方式加载
        # 假设每个sensor和center也有相应的模型保存C:/Users/25447/Desktop/HF-MARL_AoI_MEC-main (1)/logs/models/20240624-150211/agent-actor-0_episode1.h5
        # for i in range(self.sensor_num):
        #     self.sensor_actors[i] = tf.keras.models.load_model(f"C:/Users/ZY/Desktop/HF-MARL_AoI_MEC-main (1) (2)/logs/models/20240624-150211/sensor-actor-{i}_episode{episode}.h5")
        #     self.sensor_critics[i] = tf.keras.models.load_model(f"C:/Users/ZY/Desktop/HF-MARL_AoI_MEC-main (1) (2)/logs/models/20240624-150211/sensor-critic-{i}_episode{episode}.h5")
        
        # 加载center的actor和critic模型
        self.center_actor = tf.keras.models.load_model(f"C:/Users/ZY/Desktop/HF-MARL_AoI_MEC-main (1) (2)/logs/models/20240624-150211/center-actor_episode{episode}.h5")
        self.center_critic = tf.keras.models.load_model(f"C:/Users/ZY/Desktop/HF-MARL_AoI_MEC-main (1) (2)/logs/models/20240624-150211/center-critic_episode{episode}.h5")

    # @tf.function
    def train(self, max_epochs=2000, max_step=5, up_freq=8, render=False, render_freq=1, FL=False, FL_omega=0.5, anomaly_edge=False):
        cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        train_log_dir = 'logs/fit/' + cur_time
        env_log_dir = 'logs/env/env' + cur_time
        record_dir = 'logs/records/' + cur_time
        os.mkdir(env_log_dir)
        os.mkdir(record_dir)
        summary_writer = tf.summary.create_file_writer(train_log_dir)
        # tf.summary.trace_on(graph=True, profiler=True)
        os.makedirs('logs/models/' + cur_time)
        done, episode, steps, epoch, total_reward = False, 0, 0, 0, 0
        finish_length = []
        finish_size = []
        sensor_ages = []
        self.load_models(24)
        # sensor_map = self.env.DS_map
        # sensor_pos_list = self.env.world.sensor_pos
        # sensor_states = [self.env.DS_state]
        # agent_pos = [[[agent.position[0], agent.position[1]] for agent in self.agents]]
        # agent_off = [[agent.action.offloading for agent in self.agents]]
        # agent_exe = [[agent.action.execution for agent in self.agents]]
        # agent_band = [[agent.action.bandwidth for agent in self.agents]]
        # agent_trans = [[agent.trans_rate for agent in self.agents]]
        # buff, pos = self.env.get_center_state()
        # agent_donebuff = [buff]
        # exe, done = self.env.get_buffer_state()
        # exebuff = [exe]
        # donebuff = [done]

        anomaly_step = 6000
        anomaly_agent = self.agent_num - 1

        # if anomaly_edge:
        #     anomaly_step = np.random.randint(int(max_epochs * 0.5), int(max_epochs * 0.75))
        #     anomaly_agent = np.random.randint(self.agent_num)
        # summary_record = []

        while epoch < max_epochs:
            print('epoch%s' % epoch)
            # if anomaly_edge and (epoch == anomaly_step):
            #     self.agents[anomaly_agent].movable = False

            if render and (epoch % 20 == 1):
                self.env.render(env_log_dir, epoch, True)
                # sensor_states.append(self.env.DS_state)

            if steps >= max_step:
                # self.env.world.finished_data = []
                episode += 1
                # self.env.reset()
                for m in self.agent_memory.keys():
                    del self.agent_memory[m][0:-self.batch_size * 2]
                del self.center_memory[0:-self.batch_size * 2]
                print('episode {}: {} total reward, {} steps, {} epochs'.format(episode, total_reward / steps, steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', total_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)
                    # tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=train_log_dir)

                summary_writer.flush()
                steps = 0
                total_reward = 0

            cur_reward = self.actor_act(epoch)
            # print('episode-%s reward:%f' % (episode, cur_reward))
            finish_length.append(len(self.env.world.finished_data))
            finish_size.append(sum([data[0] for data in self.env.world.finished_data]))
            sensor_ages.append(list(self.env.world.sensor_age.values()))
            # agent_pos.append([[agent.position[0], agent.position[1]] for agent in self.env.world.agents])
            # # print(agent_pos)
            # agent_off.append([agent.action.offloading for agent in self.agents])
            # agent_exe.append([agent.action.execution for agent in self.agents])
            # # agent_band.append([agent.action.bandwidth for agent in self.agents])
            # agent_trans.append([agent.trans_rate for agent in self.agents])
            # buff, pos = self.env.get_center_state()
            # # agent_donebuff.append(buff)
            # exe, done = self.env.get_buffer_state()
            # exebuff.append(exe)
            # donebuff.append(done)
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
            # for actor_id, gradients in self.agent_actor_old_gard.items():
            #             print(f"actor ID: {actor_id}")
            #             if gradients is not None:
            #                 for i, grad in enumerate(gradients):
            #                     if grad is not None:
            #                         print(f"  Old acGradient for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  Old acGradient for layer {i}: None")
            #             else:
            #                 print("  No Oldac gradients recorded")
            # print('------------------------------')
            # for actor_id, weights in self.agent_actor_weights.items():
            #             print(f"actor ID: {actor_id}")
            #             if weights is not None:
            #                 for i, grad in enumerate(weights):
            #                     if grad is not None:
            #                         print(f"  weights for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  weights for layer {i}: None")
            #             else:
            #                 print("  No weights recorded")
            # print('------------------------------')
            # for actor_id, weights in self.agent_actor_old_weights.items():
            #             print(f"actor ID: {actor_id}")
            #             if weights is not None:
            #                 for i, grad in enumerate(weights):
            #                     if grad is not None:
            #                         print(f"  old weights for layer {i}: shape {grad.shape}, norm {tf.norm(grad).numpy()}")
            #                     else:
            #                         print(f"  old weights for layer {i}: None")
            #             else:
            #                 print("  No weights recorded") 
            # summary_record.append(self.summaries)
            # update target

            total_reward += cur_reward
            steps += 1
            epoch += 1

            # tensorboard
            with summary_writer.as_default():

                tf.summary.scalar('Main/step_average_age', cur_reward, step=epoch)
                
                tf.summary.scalar('Main/step_finish_length', finish_length[-1], step=epoch)
                tf.summary.scalar('Main/step_finish_size', finish_size[-1], step=epoch)
                tf.summary.scalar('Main/step_worst_sensor_age', sorted(list(self.env.world.sensor_age.values()))[-1], step=epoch)

            summary_writer.flush()

        # save final model
        self.save_model(episode, cur_time)
        sio.savemat(record_dir + '/data.mat',
                    {'finish_len': finish_length,
                     'finish_data': finish_size,
                     'ages': sensor_ages})
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

        # gif
        self.env.render(env_log_dir, epoch, True)
        img_paths = glob.glob(env_log_dir + '/*.png')
        img_paths.sort(key=lambda x: int(x.split('.')[0].split('\\')[-1]))

        gif_images = []
        for path in img_paths:
            gif_images.append(imageio.imread(path))
        imageio.mimsave(env_log_dir + '/all.gif', gif_images, fps=15)