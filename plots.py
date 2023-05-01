import os
import matplotlib as plt
import numpy as np
from models import FCNN, FCNNt, LeNet, LeNett, LeNetl
from utilities import generator
from dataset import CountingDataset
from utils import test
import torch
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_res = []
train_res = []
names_val = []
names_train = []
x = list(range(1, 51))
color = plt.cm.rainbow(np.linspace(0, 1, 12))
for f in os.listdir('./losses/'):
    if '128' in f or '256' in f or '512' in f:
        if 'val' in f:
            val_res.append(np.load(f))
            names_val.append(f)
        elif 'train' in f:
            train_res.append(np.load(f))
            names_train.append(f)

for i, c in zip(range(len(val_res)), color):
    plt.plot(x, val_res[i], label=names_val[i].removesuffix('_val.npy'), c=c)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.title('Validation Loss')
plt.show()

# for i in range(len(train_res)):
#     plt.plot(x, train_res[i], label=names_train[i].removesuffix('_train.npy'), c=c)
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Training Loss')
# plt.show()

# val_res = []
# train_res = []
# names_val = []
# names_train = []
# x = list(range(1, 51))
# color = plt.cm.rainbow(np.linspace(0, 1, 12))
# for f in os.listdir('./losses'):
#     if '1024' in f or '2048' in f or '4096' in f:
#         if 'val' in f:
#             val_res.append(np.load(f))
#             names_val.append(f)
#         elif 'train' in f:
#             train_res.append(np.load(f))
#             names_train.append(f)

# for i, c in zip(range(len(val_res)), color):
#     plt.plot(x, val_res[i], label=names_val[i].removesuffix('_val.npy'), c=c)

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Validation Loss')
# plt.show()

# for i in range(len(train_res)):
#     plt.plot(x, train_res[i], label=names_train[i].removesuffix('_train.npy'), c=c)
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Training Loss')
# plt.show()

# val_res = []
# train_res = []
# names_val = []
# names_train = []
# x = list(range(1, 51))
# color = plt.cm.rainbow(np.linspace(0, 1, 1000))
# for f in os.listdir('./losses'):
#     if 'lenet' in f:
#         if 'val' in f:
#             val_res.append(np.load(f))
#             names_val.append(f)
#         elif 'train' in f:
#             train_res.append(np.load(f))
#             names_train.append(f)

# for i, c in zip(range(len(val_res)), color):
#     plt.plot(x, val_res[i], label=names_val[i].removesuffix('_val.npy'), c=c)

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Validation Loss')
# plt.show()

# for i in range(len(train_res)):
#     plt.plot(x, train_res[i], label=names_train[i].removesuffix('_train.npy'), c=c)
#
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Training Loss')
# plt.show()

##################################### ACCURACY FOR EACH CLASS

# names = []
# results = []
# for f in os.listdir('./models/'):
#     if f == '.ipynb_checkpoints':
#         print('Trovato il figlio di puttana')
#         continue
#     names.append(f)
#     if f.startswith('fcnn'):
#         if '128' in f:
#             if 'tanh' in f:
#                 model = FCNNt(128)
#             else:
#                 model = FCNN(128)
#         elif '256' in f:
#             if 'tanh' in f:
#                 model = FCNNt(256)
#             else:
#                 model = FCNN(256)
#         elif '512' in f:
#             if 'tanh' in f:
#                 model = FCNNt(512)
#             else:
#                 model = FCNN(512)
#         elif '1024' in f:
#             if 'tanh' in f:
#                 model = FCNNt(1024)
#             else:
#                 model = FCNN(1024)
#         elif '2048' in f:
#             if 'tanh' in f:
#                 model = FCNNt(2048)
#             else:
#                 model = FCNN(2048)
#         elif '4096' in f:
#             if 'tanh' in f:
#                 model = FCNNt(4096)
#             else:
#                 model = FCNN(4096)
#     else:
#         if 'tanh' in f:
#             model = LeNett()
#         elif 'relu' in f:
#             model = LeNet()
#         else:
#             model = LeNetl()
#     res = []
#     for i in range(1, 11):
#         model.load_state_dict(torch.load('./models/' + f))
#         model.to(device)
#         test_matrices, test_labels = generator(n_images=1000, seed=11, fixed_n=True, n=i)
#         testset = CountingDataset(test_matrices, test_labels)
#         test_loader = DataLoader(testset, batch_size=256, shuffle=True)
#         res.append(test(model, test_loader, device))
#     results.append(res)
#
# for i in range(len(results)):
#     plt.plot(list(range(1, 11)), results[i], label=names[i].removesuffix('.pt'))
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
# plt.title('Accuracy on different labels')
# plt.show()
