import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from dataset import FLADS
import multiprocessing


lr = 3e-4
batch = 32
epoch = 10
device = torch.device('cuda')
ds_path = ['/home/FLAD_Dataset/noise', '/home/FLAD_Dataset/origin']


def test():
    train_ds = FLADS(ds_path)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, 
                            num_workers=1, pin_memory=True)

    # heads for: Lossless, AAC, MP3, Opus
    model = EfficientNet.from_name('efficientnet-b0', num_classes=4)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # load weights
    model = torch.nn.DataParallel(model, device_ids=None).cuda()
    model.load_state_dict(torch.load('save/model_fin.pth', map_location=device))

    # test loop
    with torch.no_grad():

        num_correct = 0
        num_samples = 0
        

        for it, (batch_x, batch_y) in enumerate(train_loader):

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predict_y = model(batch_x)

            _, predictions = predict_y.max(1)
            num_correct += (predictions == batch_y).sum()
            num_samples += predictions.size(0)

            print(f'already tested: {num_samples}, acc: {100*num_correct/num_samples:.3f} %')


if __name__ == '__main__':
    multiprocessing.freeze_support()
    test()
