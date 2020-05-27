# Utility to load the not-mnist dataset, pre-process and save it to numpy files
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torchvision.transforms.functional import normalize
from helpers.constants import DATA_PATH, NUMPY_DATA_PATH, NORMALIZE_IMAGES


data_path = os.path.join(DATA_PATH, 'notMNIST_small')
write_path = os.path.join(NUMPY_DATA_PATH, 'notmnist')
dirs = os.listdir(data_path)
label_dict = {}
i = 0
for directory in dirs:
    if directory not in label_dict:
        label_dict[directory] = i
        i = i + 1

print(label_dict)

final_data = None
final_label = None
counter = 0
for directory in dirs:
    dir_path = os.path.join(data_path, directory)
    images = os.listdir(dir_path)

    x = np.array([np.array(Image.open(os.path.join(dir_path, image))) for image in images])
    x_shape = x.shape
    x = x.reshape((x_shape[0], 1, x_shape[1], x_shape[2]))
    label = label_dict[directory] * np.ones((x_shape[0], 1), dtype=np.int)
    if counter == 0:
        final_data = x
        final_label = label
    else:
        final_data = np.vstack((final_data, x))
        final_label = np.vstack((final_label, label))

    counter += 1

final_label = final_label.ravel()
# torch tensor scaled to the range [0, 1]
data_ten = (1. / 255) * torch.tensor(final_data, dtype=torch.float)

# image scaling
params = NORMALIZE_IMAGES['mnist']
data_trans = data_ten.clone()
for i in range(data_ten.size(0)):
    data_trans[i] = normalize(data_ten[i], params[0], params[1])

print("Range of transformed images: {:.6f}, {:.6f}".format(data_trans.min(), data_trans.max()))
# torch tensor to numpy array
final_data = data_trans.cpu().numpy()

# Do a stratified cross-validation split and save the train and test fold data to numpy files
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
fold = 0
for train_index, test_index in skf.split(final_data, final_label):
    final_write_path = os.path.join(write_path, 'fold_' + str(fold+1))
    data_tr, data_te = final_data[train_index, :], final_data[test_index, :]
    labels_tr, labels_te = final_label[train_index], final_label[test_index]

    np.save(os.path.join(final_write_path, 'data_tr.npy'), data_tr)
    np.save(os.path.join(final_write_path, 'labels_tr.npy'), labels_tr)
    np.save(os.path.join(final_write_path, 'data_te.npy'), data_te)
    np.save(os.path.join(final_write_path, 'labels_te.npy'), labels_te)
    fold = fold + 1
