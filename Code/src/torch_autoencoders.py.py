import torch

from torch import nn


LATENT_SHAPE = 32


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim, original_size):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.inference_net = nn.Sequential(
            nn.Linear(original_size, latent_dim),
            nn.ReLU(True)
        )
        self.generative_net = nn.Sequential(
            nn.Linear(latent_dim, original_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.inference_net(x)
        x = self.generative_net(x)
        return x

    def encode(self, x):
        return self.inference_net(x)

    def decode(self, x):
        return self.generative_net(x)


if __name__ == '__main__':
    import imageio
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    import PIL

    from tensorflow.keras import datasets
    from torch.autograd import Variable

    '''
    Sorting out the preprocessed data into modeling folder

    (preprocessed data == [feature_eng.py] ==> modeling)

    '''
    (tr_X, _), (ts_X, _) = datasets.mnist.load_data()

    tr_X = tr_X.astype('float32') / 255.
    ts_X = ts_X.astype('float32') / 255.
    tr_X = tr_X.reshape((len(tr_X), np.prod(tr_X.shape[1:])))
    ts_X = ts_X.reshape((len(ts_X), np.prod(ts_X.shape[1:])))
    print(tr_X.shape)
    print(ts_X.shape)

    num_epochs = 50
    batch_size = 256
    learning_rate = 1.0

    #
    tr_X = torch.utils.data.DataLoader(ts_X, batch_size=batch_size, shuffle=False, num_workers=2)

    model = AutoEncoder(32, 28**2).cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=0)

    # Training
    for epoch in range(num_epochs):
        for data in tr_X:
            img = data
            img = img.view(img.size(0), -1)
            img = Variable(img).cuda()
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, num_epochs, loss.data))

    encoded_imgs = encoded_imgs = model.encode(torch.from_numpy(ts_X).to('cuda'))
    decoded_imgs = model.decode(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.reshape(ts_X[i], (28, 28)))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].cpu().detach().reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()