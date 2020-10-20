import os
import numpy as np
from tqdm import tqdm
from audio import hparams as audio_hparams
from audio import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec
from audio import write_wav, normalized_db_mel2wav, normalized_db_spec2wav


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

# 输入的路径,其中fname是inference_wavs_path_list中的,只是名字,后缀.wav,无路径
meta_path = '../inference_wavs_path_list.txt'
wav_dir_path = '../wav_list'


# 输出的路径
mfcc_dir = '../xxx_mfcc_5ms_by_audio_2'
mel_dir = '../xxx_mel_5ms_by_audio_2'
spec_dir = '../xxx_spec_5ms_by_audio_2'
rec_wav_dir = '../xxx_rec_wavs_audio_2'

os.makedirs(mfcc_dir, exist_ok=True)
os.makedirs(mel_dir, exist_ok=True)
os.makedirs(spec_dir, exist_ok=True)
os.makedirs(rec_wav_dir, exist_ok=True)


def main():
    a = open(meta_path, 'r').readlines()
    a = [i.strip() for i in a]
    cnt = 0
    cnt_list = []

    for fname in tqdm(a):
        wav_f = os.path.join(wav_dir_path, fname)
        wav_arr = load_wav(wav_f)
        
        mfcc_feats = wav2unnormalized_mfcc(wav_arr)
        mel_feats = wav2normalized_db_mel(wav_arr)
        spec_feats = wav2normalized_db_spec(wav_arr)
        
        # 验证声学参数提取的对儿
        save_name = fname + '.npy'
        save_mel_rec_name = fname + '_mel_groundtruth.wav'
        save_spec_rec_name = fname + '_spec_groundtruth.wav'

        assert mfcc_feats.shape[0] == mel_feats.shape[0] and mel_feats.shape[0] == spec_feats.shape[0]
        write_wav(os.path.join(rec_wav_dir, save_mel_rec_name), normalized_db_mel2wav(mel_feats))
        write_wav(os.path.join(rec_wav_dir, save_spec_rec_name), normalized_db_spec2wav(spec_feats))
        
        # 存储声学参数
        mfcc_save_name = os.path.join(mfcc_dir, save_name)
        mel_save_name = os.path.join(mel_dir, save_name)
        spec_save_name = os.path.join(spec_dir, save_name)
        np.save(mfcc_save_name, mfcc_feats)
        np.save(mel_save_name, mel_feats)
        np.save(spec_save_name, spec_feats)

        cnt_list.append(fname)
        cnt += 1

    print(cnt)
    print(cnt_list)

    return


if __name__ == '__main__':
    main()
