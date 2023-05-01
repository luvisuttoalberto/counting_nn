import torch.optim
from torch.utils.data import DataLoader
from dataset import CountingDataset
from utilities import generator
from models import FCNN, LeNet
from pathlib import Path
import numpy as np
import os.path


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
    return 100 * correct // total


if __name__ == "__main__":
    n_images_test = 10000

    # test data
    # Verify if you have saved dataset to be loaded; if not, generate it
    method = "classic"
    # method = "cross"
    # method = "gauss"
    # method = "inverted"
    # method = "luminosity"

    Path("./data/test/%s_images" % method).mkdir(parents=True, exist_ok=True)
    Path("./data/test/%s_labels" % method).mkdir(parents=True, exist_ok=True)
    imagesPath = "data/test/%s_images/%d.npy" % (method, n_images_test)
    labelsPath = "data/test/%s_labels/%d.npy" % (method, n_images_test)
    if not os.path.isfile(imagesPath) or not os.path.isfile(labelsPath):
        test_matrices, test_labels = generator(n_images=n_images_test, seed=14, method=method)
        np.save(imagesPath, test_matrices)
        np.save(labelsPath, test_labels)
    else:
        test_matrices = np.load(imagesPath)
        test_labels = np.load(labelsPath)

    print("Confusion factor: ", method)
    testset = CountingDataset(test_matrices/255., test_labels)
    test_loader = DataLoader(testset, batch_size=256, shuffle=True)

    # model = FCNN()
    model = LeNet()
    model_name = "lenet_tanh"
    saved_state_dict = torch.load("./models/%s.pt" % model_name)
    model.load_state_dict(saved_state_dict)

    accuracy = test(model, test_loader, device='cpu')
    Path("./accuracy/%s" % method).mkdir(parents=True, exist_ok=True)
    np.save("./accuracy/%s/%s.npy" % (method, model_name), accuracy)


