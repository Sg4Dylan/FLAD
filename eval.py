from librosa.core import spectrum
import numpy as np
from PIL import Image
import onnxruntime
import utils


class FLAD:

    def __init__(self):
        self.model_path = 'onnx/flad.onnx'
        self.img_size = 400
        self.r_map = ['FLAC', 'AAC', 'mp3', 'Opus']
        self.init_model()
    

    def img_preprocess(self, image_path):
        try:
            img = Image.open(image_path).resize((self.img_size, self.img_size),Image.BICUBIC)
            img = img.convert('RGBA').convert('RGB')
        except OSError:
            print(f'\nFile broken: {image_path}')
            return None
        input_data = np.array(img).transpose(2, 0, 1)
        img_data = input_data.astype('float32')
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        norm_img_data = norm_img_data.reshape(1, 3, self.img_size, self.img_size).astype('float32')
        return norm_img_data
    

    def init_model(self):
        self.session_opti = onnxruntime.SessionOptions()
        self.session_opti.enable_mem_pattern = False
        self.provider = 'DmlExecutionProvider' # or CpuExecutionProvider
        self.session = onnxruntime.InferenceSession(self.model_path, self.session_opti)
        self.session.set_providers([self.provider])
        self.model_input = self.session.get_inputs()[0].name
    
    
    def get_result(self, audio_path):

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        print('Generate side channel...')
        y_s = utils.get_side(audio_path)
        print('Rendering spectrum...')
        utils.get_spectrum(y_s, 0, 'temp', max=20)
        spectrum_list = utils.get_file_list('temp')
        print('Valid samples...')
        for i_idx in range(len(spectrum_list)):
            norm_img = self.img_preprocess(spectrum_list[i_idx])
            result = self.session.run([], {self.model_input: norm_img})[0][0]
            result = softmax(result)
            print(f'Sample {i_idx+1} -> {self.r_map[np.argmax(result)]}, Prob：{np.max(result)*100:.3f}%')

flad = FLAD()
flad.get_result('fake.flac')
