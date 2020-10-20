import os
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.switch_backend('agg')

import torch
from torch.autograd import Variable

from audio import hparams as audio_hparams
from audio import normalized_db_mel2wav, normalized_db_spec2wav, write_wav

from model_torch import DCBHG


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 80,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': True,
}


assert hparams == audio_hparams


use_cuda = torch.cuda.is_available()
assert use_cuda is True
ckpt_path_DataBakerCN = './const_ckpt/checkpoint_step000034800.pth'


# 输入的路径,其中fname是inference_wavs_path_list中的,只是名字,无后缀,无路径
meta_path = '../inference_wavs_path_list.txt'
ppg_dir = '../xxx_ppg_5ms_by_audio_2'


# 输出的路径
rec_wav_dir = '../xxx_rec_wavs_audio_2'

# 全局变量
model = DCBHG()

def tts_load(model, ckpt_path):
    ckpt_load = torch.load(ckpt_path)
    model.load_state_dict(ckpt_load["state_dict"])
    if use_cuda:
        model = model.cuda()
    model.eval()
    return model


def tts_predict(model, ppg):
    # 准备输入的数据并转换到GPU
    ppg = Variable(torch.from_numpy(ppg)).unsqueeze(0).float()
    print(ppg.size())
    print(ppg.shape)
    print(ppg.type())
    if use_cuda:
        ppg = ppg.cuda()

    # 进行预测并数据转换到CPU
    mel_pred, spec_pred = model(ppg)
    mel_pred = mel_pred[0].cpu().data.numpy()
    spec_pred = spec_pred[0].cpu().data.numpy()

    # vocoder合成音频波形文件
    mel_pred_audio = normalized_db_mel2wav(mel_pred)
    spec_pred_audio = normalized_db_spec2wav(spec_pred)

    return mel_pred, spec_pred, mel_pred_audio, spec_pred_audio


def draw_spec(a_path, a):
    plt.imshow(a.T, cmap='hot', interpolation='nearest')
    plt.xlabel('frame nums')
    plt.ylabel('spec')
    plt.tight_layout()
    plt.savefig(a_path, format='png')


def main():
    global model
    model = tts_load(model=model, ckpt_path=ckpt_path_DataBakerCN)

    a = open(meta_path, 'r').readlines()
    a = [i.strip() for i in a]
    for fname in tqdm(a):
        ppg_path = os.path.join(ppg_dir, fname + '.npy')
        ppg = np.load(ppg_path)
        _mel_pred, _spec_pred, mel_pred_audio, spec_pred_audio = tts_predict(model, ppg)

        write_wav(os.path.join(rec_wav_dir, fname + '_mel_databakercn.wav'), mel_pred_audio)
        write_wav(os.path.join(rec_wav_dir, fname + '_spec_databakercn.wav'), spec_pred_audio)

        # np.save(os.path.join(DataBakerCN_log_dir, "{}_sample_mel.npy".format(idx)), mel_pred)
        # np.save(os.path.join(DataBakerCN_log_dir, "{}_sample_spec.npy".format(idx)), spec_pred)

        # draw_spec(os.path.join(DataBakerCN_log_dir, "{}_sample_mel.png".format(idx)), mel_pred)
        # draw_spec(os.path.join(DataBakerCN_log_dir, "{}_sample_spec.png".format(idx)), spec_pred)
      
    

if __name__ == "__main__":
    main()