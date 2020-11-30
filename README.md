# FLAD
Fake Lossless Audio Detector

### Feature

 - Determine the audio source format:  
 Lossless, AAC, MP3(ex. SoundCloud), Opus(ex. YouTube)  
 - Not rely on high frequency information:  
 Frequencies below 2.4kHz or above 20kHz are not used  

### Dependence

eval only:
 - librosa >= 0.8.0  
 - resampy  
 - matplotlib  
 - numpy  
 - Pillow  
 - onnxruntime >= 1.5.2  

train & test:
 - pytorch >= 1.5.1  
 - efficientnet_pytorch  
 - livelossplot (option)  

### Usage

generate dataset:  

```
# set input & output path  
audio_root = '/home/audio'  
ds_root = '/home/dataset'
```
then `python generate.py`  

train model:  

`python train.py`  

test model (in pytorch):  

`python test.py`  

export model to onnx:  

`python export.py`  

eval model (in onnxruntime):  
```
# set input file path  
flad.get_result('fake.flac')
```
run `python eval.py`  