# -*- coding: utf-8 -*-
# @Time    : 2018/8/24 10:20
# @Author  : zhoujun

# NOTE: -1 is reserved for 'blank' required by mxnet ctc
from mxnet import nd
from mxnet.gluon.data import ArrayDataset, DataLoader


class t:
    def __init__(self):
        data_set = [ArrayDataset(list(range(10))), ArrayDataset(list(range(20, 40)))]
        ratio = [3, 4]
        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        for d, r in zip(data_set, ratio):
            self.data_loader_list.append(DataLoader(dataset=d, batch_size=r, last_batch='rollover',shuffle=True))
            self.dataset_len += len(d)
        for s in self.data_loader_list:
            self.dataloader_iter_list.append(iter(s))

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        balanced_batch_images = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image = next(data_loader_iter)
                balanced_batch_images.append(image)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
        batch = nd.concat(*balanced_batch_images, dim=0)
        return batch


a = t()
epochs = 2
print(epochs, a.dataset_len)
for epoch in range(epochs):
    print('epoch', epoch)
    for i, data in enumerate(a):
        if i >= len(a):
            break
        print(data)
