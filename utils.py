import torch.optim

from models import FCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # training data
    trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    train_samples = int(len(trainset) * 0.8)
    valid_samples = len(trainset) - train_samples
    trainset, validset = torch.utils.data.random_split(trainset, [train_samples, valid_samples])
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=True)

    # test data
    testset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True)

    n_epochs = 10
    model = FCNN()
    # model = LeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train(model, train_loader, valid_loader, n_epochs, optimizer, loss_fn, device)
    test(model, test_loader, device)
