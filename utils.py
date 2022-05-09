import os
import librosa
import librosa.display
import resampy
import numpy as np
import matplotlib.pyplot as plt


def get_side(f, t_sr=48000):
    y, y_sr = librosa.load(f, sr=None, mono=False)
    if y_sr != t_sr:
        y = resampy.resample(y, y_sr, t_sr, filter='kaiser_fast')
    return (y[0]-y[1])/2


def get_stft(y_s, n_fft=2048):
    return librosa.stft(np.asfortranarray(y_s), n_fft=n_fft)


def get_spectrum(y_s, fid, root, t_sr=48000, n_fft=512, max=-1, noise=None):
    fig_dpi = 100
    fig_size = (800/fig_dpi, 800/fig_dpi)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
    # Remove blank board
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.set_axis_off()
    # Remove freq
    clip_len = t_sr * 2
    dn_f, up_f = 2400, 20000
    dn_th, up_th = round(n_fft*dn_f/t_sr), round(n_fft*up_f/t_sr)
    # Add noise (for generate dataset)
    if noise is not None:
        y_s = add_noise(y_s, noise[0], noise[1])
    # Render images
    cnt = 1
    for i in range(0, len(y_s), clip_len):
        ys_clip = get_stft(y_s[i:i+clip_len], n_fft)
        show_part = ys_clip[dn_th:up_th,:]
        p_part = np.mean(np.abs(show_part))
        p_std = np.std(show_part)
        # Skip blank
        if p_part < 1e-6:
            continue
        # Skip low std sample
        if p_std < 1e-4:
            continue
        # Let high std more prob be render
        if np.random.random() > np.tanh((p_std*60)**5):
            continue
        # Save
        db = librosa.amplitude_to_db(np.abs(show_part),ref=np.max)
        librosa.display.specshow(db, sr=up_f-dn_f, y_axis='linear', x_axis='time')
        fig.savefig(f'{root}/{fid}_{cnt}.jpg')
        cnt += 1
        if max!=-1 and cnt > max:
            break


def get_file_list(target_root, ext=None):
    file_path_list = []
    for root, dirs, files in os.walk(target_root):
        for name in files:
            if ext is not None:
                if not name.endswith(ext):
                    continue
            file_path_list.append(os.path.join(root, name))
    return file_path_list


def add_noise(y_s, sv_l=0.002,sv_h=0.01):
    # AkkoMode like
    jitter_vector = np.random.uniform(low=sv_l, high=sv_h, size=y_s.shape)+1
    return y_s*jitter_vector
