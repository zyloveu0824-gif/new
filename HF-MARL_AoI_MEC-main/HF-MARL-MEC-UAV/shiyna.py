import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np

def sensor_actor(input_dim_list):
    # Input
    sensor_buffer_state = keras.Input(shape=input_dim_list[0])  # 输入层1 shape为sensor_buffer_state

    # （输入：源数据缓冲区，输出：sensor卸载）
    sensor_mlp = tf.transpose(sensor_buffer_state, perm=[0, 2, 1])  # 对数组进行转置（把最内侧的两维进行转置） 注意：sensor_buffer的shape为(?, 2, max_buffer_size)
    sensor_mlp = layers.Dense(1, activation='relu')(sensor_mlp)  # 全连接层 神经元数量为1
    sensor_mlp = tf.squeeze(sensor_mlp, axis=-1)  # 从张量shape中移除大小为1的维度
    computing_dev = layers.Dense(1, activation='sigmoid')(sensor_mlp)  # 全连接层 神经元数量为2
    model = keras.Model(inputs=[sensor_buffer_state], outputs=[computing_dev])  # 构造模型

    return model

def get_log_probs(actor, sensor_data_state, actions):
    """
    计算在给定状态和动作下的策略的对数概率。

    参数:
    actor -- 策略网络
    sensor_data_state -- 传感器状态
    actions -- 动作（连续的计算资源分配）

    返回:
    对数概率
    """
    # 获取策略网络在给定状态下的动作分布参数（logits）
    logits = actor(sensor_data_state)
    
    # 创建Bernoulli分布对象
    dist = tfp.distributions.Bernoulli(logits=logits)
    
    # 计算给定动作的对数概率
    log_probs = dist.log_prob(actions)
    
    # 对多维动作求和
    log_probs = tf.reduce_sum(log_probs, axis=-1)
    
    return log_probs

# 定义测试输入
input_dim_list = [(2, 10)]  # 假设每个传感器有 2 个特征，时间步长为 10
sensor_data_state = np.random.random((5, 2, 10))  # 生成一些随机输入数据，假设有 5 个样本

# 创建模型并测试输出
actor_model = sensor_actor(input_dim_list)
actions = actor_model(sensor_data_state)
print("Model output (actions):", actions.numpy())

# 计算动作的对数概率
log_probs = get_log_probs(actor_model, sensor_data_state, actions)
print("Log probabilities of actions:", log_probs.numpy())
