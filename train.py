import time
import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from livelossplot import PlotLosses
from dataset import FLADS

lr = 3e-4
batch = 32
epoch = 10
device = torch.device('cuda')
ds_path = ['/home/FLAD_Dataset/noise', '/home/FLAD_Dataset/origin']

train_ds = FLADS(ds_path)
train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True,
                             num_workers=1, pin_memory=True)

# heads for: Lossless, AAC, MP3, Opus
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
model = torch.nn.DataParallel(model, device_ids=None).cuda()
# load weights
model.load_state_dict(torch.load('save/model_fin.pth'))

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr)

# loss
loss = torch.nn.CrossEntropyLoss()

# plot
liveloss = PlotLosses()

# train loop
for ep in range(epoch):
    s_time = time.time()
    p_loss_v = 0
    print(f'start ep: {ep}')

    for it, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predict = model(batch_x)
        p_loss = loss(predict, batch_y)
        p_loss_v = p_loss.item()
        p_loss.backward()
        optimizer.step()

        # plot
        if it%50 == 0:
            liveloss.update({'loss': p_loss_v})
            liveloss.send()
    
    print(f'end ep: {ep} @ {time.time()-s_time:.3f}s')

    if (ep+1) % 2 == 0:
        torch.save(model.state_dict(), f'save/ep_{ep+1}.pth')
