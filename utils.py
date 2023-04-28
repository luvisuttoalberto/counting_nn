import os.path
import torch.optim

from models import FCNN, LeNet
from torch.utils.data import DataLoader
from dataset import CountingDataset
import numpy as np
from utilities import generator
from pathlib import Path


def train(model, trainloader, validloader, n_epochs, optimizer, loss_fn, device):
    model.to(device)
    for e in range(n_epochs):
        model.train()
        print('Epoch: ', e)
        for i, data in enumerate(trainloader):
            input, label = data
            input = input.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, label)
            # print(loss)
            loss.backward()
            optimizer.step()

        if e % 10 == 0:
            model.eval()
            total = 0.0
            correct = 0.0
            with torch.no_grad():
                for idx, (v_input, v_label) in enumerate(validloader):
                    v_input = v_input.to(device)
                    v_label = v_label.to(device)
                    pred = model(v_input)
                    pred = torch.argmax(pred, dim=-1)
                    total += v_label.size(0)
                    correct += (pred == v_label).sum().item()
                print(f'Accuracy of the network on validation set: {100 * correct // total} %')


def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            image, label = data
            image = image.to(device)
            label = label.to(device)
            # calculate outputs by running images through the network
            outputs = model(image)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')


if __name__ == "__main__":

    n_images = 50000
    n_images_test = int(n_images * 0.2)

    # train data
    # Verify if you have saved dataset to be loaded; if not, generate it
    Path("./data/train/images").mkdir(parents=True, exist_ok=True)
    Path("./data/train/labels").mkdir(parents=True, exist_ok=True)
    imagesPath = "data/train/images/%d.npy" % n_images
    labelsPath = "data/train/labels/%d.npy" % n_images
    if not os.path.isfile(imagesPath) or not os.path.isfile(labelsPath):
        matrices, labels = generator(n_images)
        np.save(imagesPath, matrices)
        np.save(labelsPath, labels)
    else:
        matrices = np.load(imagesPath)
        labels = np.load(labelsPath)
    trainset = CountingDataset(matrices, labels)
    train_samples = int(len(trainset) * 0.8)
    valid_samples = len(trainset) - train_samples
    trainset, validset = torch.utils.data.random_split(trainset, [train_samples, valid_samples])
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=True)

    # test data
    # Verify if you have saved dataset to be loaded; if not, generate it
    Path("./data/test/images").mkdir(parents=True, exist_ok=True)
    Path("./data/test/labels").mkdir(parents=True, exist_ok=True)
    imagesPath = "data/test/images/%d.npy" % n_images_test
    labelsPath = "data/test/labels/%d.npy" % n_images_test
    if not os.path.isfile(imagesPath) or not os.path.isfile(labelsPath):
        test_matrices, test_labels = generator(n_images=n_images_test, seed=14)
        np.save(imagesPath, test_matrices)
        np.save(labelsPath, test_labels)
    else:
        test_matrices = np.load(imagesPath)
        test_labels = np.load(labelsPath)
    testset = CountingDataset(test_matrices, test_labels)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    n_epochs = 100
    # model = FCNN()
    model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, train_loader, valid_loader, n_epochs, optimizer, loss_fn, device)
    test(model, test_loader, device)
