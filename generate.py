import utils
import random
import os

audio_root = '/home/music'
ds_root = '/home/FLAD_Dataset'

select_cnt = 50
target_list = utils.get_file_list(audio_root, 'flac')
random.shuffle(target_list)
target_list = target_list[:50]

for idx, i in enumerate(target_list):
    o_path = i.replace('\\', '/')
    os.system(f'ffmpeg -i "{o_path}" -map 0:a -af aresample=resampler=soxr -ar 48000 "{ds_root}/temp.flac"')
    os.system(f'ffmpeg -i "{ds_root}/temp.flac" -b:a 128k "{ds_root}/temp.opus"')
    os.system(f'ffmpeg -i "{ds_root}/temp.flac" -b:a 128k "{ds_root}/temp.aac"')
    os.system(f'ffmpeg -i "{ds_root}/temp.flac" -b:a 128k "{ds_root}/temp.mp3"')
    y_s = utils.get_side(o_path)
    utils.get_spectrum(y_s, idx, f'{ds_root}/LL')
    y_s = utils.get_side(f'{ds_root}/temp.opus')
    utils.get_spectrum(y_s, idx, f'{ds_root}/OPUS')
    y_s = utils.get_side(f'{ds_root}/temp.aac')
    utils.get_spectrum(y_s, idx, f'{ds_root}/AAC')
    y_s = utils.get_side(f'{ds_root}/temp.mp3')
    utils.get_spectrum(y_s, idx, f'{ds_root}/MP3')
    os.remove(f'{ds_root}/temp.flac')
    os.remove(f'{ds_root}/temp.opus')
    os.remove(f'{ds_root}/temp.aac')
    os.remove(f'{ds_root}/temp.mp3')
