import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from dataset import FLADS
from tqdm import tqdm
import multiprocessing


lr = 3e-4
batch = 32
epoch = 10
device = torch.device("cpu")
ds_path = '/home/FLAD_Dataset'


def test():
    train_ds = FLADS(ds_path)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                            num_workers=1, pin_memory=True)

    # heads for: Lossless, AAC, MP3, Opus
    model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # loss
    loss = torch.nn.CrossEntropyLoss()

    # load weights
    model.load_state_dict(torch.load('save/model_fin.pth', map_location=device))

    # test loop
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        

        for it, (batch_x, batch_y) in tqdm(enumerate(train_loader), ascii=True):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predict_y = model(batch_x)

            _, predictions = predict_y.max(1)
            num_correct += (predictions == batch_y).sum()
            num_samples += predictions.size(0)

            false_label = []
            for i in range(len(batch_y)):
                if predictions[i] != batch_y[i]:
                    false_label.append(batch_y[i].detach(). numpy())
            print(false_label)

            print(f'already tested: {num_samples}, correct: {num_correct}')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    test()
