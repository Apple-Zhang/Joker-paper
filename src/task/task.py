import os
import numpy as np
import torch
import scipy.io as sio
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, r2_score, accuracy_score
from typing import Tuple, Callable
from static_config import DATA_DIR

import torch.nn.functional as F


def make_task(dataset, do_transform=True, do_target_transform=True):
    if dataset == 'msd':
        return MillionSong(do_transform=do_transform, do_target_transform=do_target_transform)
    elif dataset == 'susy':
        return Susy(do_transform=do_transform, do_target_transform=do_target_transform)
    elif dataset == 'higgs':
        return Higgs(do_transform=do_transform, do_target_transform=do_target_transform)
    elif dataset == 'cifar5m':
        return Cifar5M(do_transform=do_transform, do_target_transform=do_target_transform)
    elif dataset == 'houseelec':
        return HouseElec(do_transform=do_transform, do_target_transform=do_target_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

class FeatureNormalization:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.istd = 1/(std + torch.finfo(std.dtype).eps)

    def __call__(self, x: torch.Tensor):
        return (x - self.mean) * self.istd
    
    def inverse(self, x: torch.Tensor):
        return x * (1/self.istd) + self.mean

class MillionSong:
    def __init__(self, do_transform=True, do_target_transform=True):
        train = np.load(f'{DATA_DIR}/YP-MSD/msd_train.npz')
        self.X_train = torch.from_numpy(train['X']).float()
        self.y_train = torch.from_numpy(train['Y']).unsqueeze(1).float()
        test = np.load(f'{DATA_DIR}/YP-MSD/msd_test.npz')
        self.X_test = torch.from_numpy(test['X']).float()
        self.y_test = torch.from_numpy(test['Y']).unsqueeze(1).float()

        if do_transform:
            self.transform = FeatureNormalization(self.X_train.mean(axis=0), self.X_train.std(axis=0))
            self.X_train = self.transform(self.X_train)
            self.X_test = self.transform(self.X_test)
        else:
            self.transform = None

        if do_target_transform:
            self.target_transform = FeatureNormalization(self.y_train.mean(axis=0), self.y_train.std(axis=0))
            self.y_train = self.target_transform(self.y_train)
            self.y_test = self.target_transform(self.y_test)
        else:
            self.target_transform = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def metric(self, y_pred: torch.Tensor):
        if self.target_transform is None:
            y_pred_raw = y_pred.to('cpu')
            y_test_raw = self.y_test.to('cpu')
        else:
            y_pred_raw = y_pred.to('cpu') / self.target_transform.istd + self.target_transform.mean
            y_test_raw = self.y_test.to('cpu') / self.target_transform.istd + self.target_transform.mean

        # y_pred_np = y_pred.to('cpu').numpy()
        # y_test_np = self.y_test.to('cpu').numpy()
        y_pred_raw_np = y_pred_raw.numpy()
        y_test_raw_np = y_test_raw.numpy()

        # mse = mean_squared_error(y_test_np, y_pred_np) * 0.5
        mse_raw = mean_squared_error(y_test_raw_np, y_pred_raw_np)
        rel_err = mean_squared_error(y_test_raw_np / y_test_raw_np, y_pred_raw_np / y_test_raw_np) ** 0.5
        # r2  = r2_score(y_test_np, y_pred_np)
        # pearson = np.corrcoef(y_test_np.squeeze(), y_pred_np.squeeze())[0,1]
        # mae_raw = np.mean(np.abs(y_test_raw_np - y_pred_raw_np))

        # return {".5mse": mse, "mse-raw": mse_raw, "rel_err": rel_err, "r2": r2, "pearson": pearson, "mae-raw": mae_raw}
        return {"mse-raw": mse_raw, "rel_err": rel_err}
    
class Susy:
    def __init__(self, do_transform=True, do_target_transform=True):
        train = np.load(f'{DATA_DIR}/susy/susy_train.npz')
        self.X_train = torch.from_numpy(train['X']).float()
        self.y_train = torch.from_numpy(train['Y']).unsqueeze(1).float()
        test = np.load(f'{DATA_DIR}/susy/susy_test.npz')
        self.X_test = torch.from_numpy(test['X']).float()
        self.y_test = torch.from_numpy(test['Y']).unsqueeze(1).float()

        if do_transform:
            self.transform = FeatureNormalization(self.X_train.mean(axis=0), self.X_train.std(axis=0))
            self.X_train = self.transform(self.X_train)
            self.X_test = self.transform(self.X_test)
        else:
            self.transform = None

        if do_target_transform:
            self.target_transform: Callable[[torch.Tensor], torch.Tensor] = lambda y: y*2 - 1
            self.y_train = self.target_transform(self.y_train)
            self.y_test = self.target_transform(self.y_test)
        else:
            self.target_transform = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def metric(self, y_pred: torch.Tensor):
        y_pred_np = y_pred.to('cpu', dtype=torch.float32).numpy()
        y_test_np = self.y_test.to('cpu', dtype=torch.float32).numpy()

        y_pred_cls = self.target_transform((y_pred >= 0)).to("cpu").numpy().astype(int)
        y_test_cls = y_test_np.astype(int)

        f1  = f1_score(y_test_cls, y_pred_cls) * 100
        auc = roc_auc_score(y_test_np, y_pred_np) * 100
        acc = accuracy_score(y_test_cls, y_pred_cls) * 100

        return {"f1": f1, "auc": auc, "acc": acc}
    
class Higgs:
    def __init__(self, do_transform=True, do_target_transform=True):
        train = np.load(f'{DATA_DIR}/higgs/higgs_train.npz')
        self.X_train = torch.from_numpy(train['X']).float()
        self.y_train = torch.from_numpy(train['Y']).unsqueeze(1).float()
        test = np.load(f'{DATA_DIR}/higgs/higgs_test.npz')
        self.X_test = torch.from_numpy(test['X']).float()
        self.y_test = torch.from_numpy(test['Y']).unsqueeze(1).float()

        if do_transform:
            self.transform = FeatureNormalization(self.X_train.mean(axis=0), self.X_train.std(axis=0))
            self.X_train = self.transform(self.X_train)
            self.X_test = self.transform(self.X_test)
        else:
            self.transform = None

        if do_target_transform:
            self.target_transform: Callable[[torch.Tensor], torch.Tensor] = lambda y: y*2 - 1
            self.y_train = self.target_transform(self.y_train)
            self.y_test = self.target_transform(self.y_test)
        else:
            self.target_transform = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def metric(self, y_pred: torch.Tensor):
        y_pred_np = y_pred.to('cpu', dtype=torch.float32).numpy()
        y_test_np = self.y_test.to('cpu', dtype=torch.float32).numpy()

        y_pred_cls = self.target_transform((y_pred >= 0)).to("cpu").numpy().astype(int)
        y_test_cls = y_test_np.astype(int)

        f1  = f1_score(y_test_cls, y_pred_cls) * 100
        auc = roc_auc_score(y_test_np, y_pred_np) * 100
        acc = accuracy_score(y_test_cls, y_pred_cls) * 100

        return {"f1": f1, "auc": auc, "acc": acc}

class Cifar5M:
    def __init__(self, do_transform=True, do_target_transform=True):
        train_0 = np.load(f'{DATA_DIR}/cifar5m/cifar5m_part0.npz')
        train_1 = np.load(f'{DATA_DIR}/cifar5m/cifar5m_part1.npz')
        train_2 = np.load(f'{DATA_DIR}/cifar5m/cifar5m_part2.npz')
        train_3 = np.load(f'{DATA_DIR}/cifar5m/cifar5m_part3.npz')
        self.X_train = torch.from_numpy(np.concatenate([train_0['X'], train_1['X'], train_2['X'], train_3['X']])).view(-1,3072).float()
        self.y_train = torch.from_numpy(np.concatenate([train_0['Y'], train_1['Y'], train_2['Y'], train_3['Y']])).long()
        test_0 = np.load(f'{DATA_DIR}/cifar5m/cifar5m_part4.npz')
        self.X_test = torch.from_numpy(test_0['X']).view(-1,3072).float()
        self.y_test = torch.from_numpy(test_0['Y']).float()

        if do_transform:
            # inplace normalization to avoid extra memory usage
            xmean = self.X_train.mean(axis=0)
            xstd = self.X_train.std(axis=0)
            self.X_train -= xmean
            self.X_train /= xstd
            self.X_test -= xmean
            self.X_test /= xstd
        else:
            self.transform = None

        if do_target_transform:
            self.y_train = F.one_hot(self.y_train.long(), num_classes=10).float()
            self.y_train *= 2
            self.y_train -= 1

            self.y_test = F.one_hot(self.y_test.long(), num_classes=10).float()
            self.y_test *= 2
            self.y_test -= 1
        else:
            self.target_transform = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def metric(self, y_pred: torch.Tensor):
        y_pred_np = y_pred.to('cpu', dtype=torch.float32).numpy()
        y_test_np = self.y_test.to('cpu', dtype=torch.float32).numpy()

        # y_pred_prob = y_pred.softmax()
        # # y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1, keepdims=True)
        # y_pred_prob = y_pred_prob.to('cpu').numpy()

        # multiclassification
        y_pred_cls = np.argmax(y_pred_np, axis=1)
        y_test_cls = np.argmax(y_test_np, axis=1)

        # auc = roc_auc_score(y_test_cls, y_pred_prob, multi_class='ovr')
        acc = accuracy_score(y_test_cls, y_pred_cls) * 100

        return {"acc": acc}

class HouseElec:
    def __init__(self, do_transform=True, do_target_transform=True):
        self.name = "houseelec"
        train = np.load(f'{DATA_DIR}/houseelec/train_data.npz')
        self.X_train = torch.from_numpy(train['x_train']).float()
        self.y_train = torch.from_numpy(train['y_train']).float()
        test = np.load(f'{DATA_DIR}/houseelec/test_data.npz')
        self.X_test = torch.from_numpy(test['x_test']).float()
        self.y_test = torch.from_numpy(test['y_test']).float()

        if do_transform:
            self.transform = FeatureNormalization(self.X_train.mean(axis=0), self.X_train.std(axis=0))
            self.X_train = self.transform(self.X_train)
            self.X_test = self.transform(self.X_test)
        else:
            self.transform = None

        if do_target_transform:
            self.target_transform = None
        else:
            self.target_transform = None

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def metric(self, y_pred: torch.Tensor):
        y_pred_np = y_pred.to('cpu').numpy()
        y_test_np = self.y_test.to('cpu').numpy()
        mse = mean_squared_error(y_test_np, y_pred_np)
        rmse = mse ** 0.5
        rel_err = mean_squared_error(y_test_np / y_test_np, y_pred_np / y_test_np) ** 0.5

        return {"rmse": rmse, "mse": mse, "rel_err": rel_err}