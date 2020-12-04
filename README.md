# FLAD
Fake Lossless Audio Detector

### Feature

 - Determine the audio source format:  
 Lossless, AAC, MP3(ex. SoundCloud), Opus(ex. YouTube)  
 - Not rely on high frequency information:  
 Frequencies below 2.4kHz or above 20kHz are not used  
 - Resists minor noise interference:  
 Noisy samples included in training data  

### Demo

Audio: AAC downloaded from YouTube  
Post-processing: EmiyaEngine  
Spectrum:  
![Spectrum](https://imgur.com/Inw3oPm.png)  
Lossless Audio Checker (not work):  
![LosslessAudioChecker](https://imgur.com/5gugaLb.png)  
Ours:  
![Ours](https://imgur.com/uwI72Jc.png)


### Performance

Test set: 200 samples per category (50% with noise)  
Result: 798 correct identifications out of 800 samples with 99.75 % accuracy

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

export model:
 - onnx-simplifier  

### Usage

generate dataset:  

```
# set input & output path  
audio_root = '/home/audio'  
ds_root = '/home/FLAD_Dataset/noise'  
ds_test = '/home/FLAD_Dataset/origin'
```
then `python generate.py`  

train model:  

`python train.py`  

test model (in pytorch):  

`python test.py`  

export model to onnx:  

`python export.py`  

eval model (in onnxruntime):  
run `python eval.py fake.flac`  
