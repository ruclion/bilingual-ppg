import tensorflow as tf
import os
import time
import numpy as np
from tqdm import tqdm

from models import CNNBLSTMCalssifier


# 超参数
MFCC_DIM = 39
PPG_DIM = 345


# 输入的路径,其中fname是inference_wavs_path_list中的,只是名字,无后缀,无路径
meta_path = '../inference_wavs_path_list.txt'
mfcc_dir = '../xxx_mfcc_5ms_by_audio_2'

# 注意ckpt是自己路径的,只有一个点
ckpt_path = './LibriSpeech_ckpt_model_zhaoxt_dir/vqvae.ckpt-233000'


# 输出的路径
ppg_dir = '../xxx_ppg_5ms_by_audio_2'
if os.path.exists(ppg_dir) is False:
    os.makedirs(ppg_dir)


def main():
    a = open(meta_path, 'r').readlines()
    a = [i.strip() for i in a]

    # 设置网络结构,指定ppg的输出为softmax之后
    mfcc_pl = tf.placeholder(dtype=tf.float32,
                             shape=[None, None, MFCC_DIM],
                             name='mfcc_pl')
    classifier = CNNBLSTMCalssifier(out_dims=PPG_DIM, n_cnn=3, cnn_hidden=256,
                                    cnn_kernel=3, n_blstm=2, lstm_hidden=128)
    predicted_ppgs = tf.nn.softmax(classifier(inputs=mfcc_pl)['logits'])

    # 启动tf的session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    start_time = time.time()
    sess.run(tf.global_variables_initializer())

    # restore CKPT
    saver = tf.train.Saver()
    print('Restoring model from {}'.format(ckpt_path))
    saver.restore(sess, ckpt_path)


    # 开始逐句预测
    for fname in tqdm(a):
        # 准备数据
        mfcc_f_path = os.path.join(mfcc_dir, fname + '.npy')
        mfcc = np.load(mfcc_f_path)

        # 预测
        ppgs = sess.run(predicted_ppgs, feed_dict={mfcc_pl: np.expand_dims(mfcc, axis=0)})
        assert mfcc.shape[0] == (np.squeeze(ppgs)).shape[0]

        # 存储PPG
        ppg_f_path = os.path.join(ppg_dir, fname + '.npy')
        np.save(ppg_f_path, np.squeeze(ppgs))

    duration = time.time() - start_time
    print("PPGs file generated in {:.3f} seconds".format(duration))
    sess.close()


if __name__ == '__main__':
    main()
