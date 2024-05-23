import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

import random

def get_dataloader(dataset_str, random_seed=0, train_ratio=0.8, batch_size_train=128, batch_size_validation=128,
                   batch_size_test=128):
    batch_size_train = 128
    batch_size_validation = 128
    batch_size_test = 128

    # Useful for random transforms
    random.seed(random_seed)
    torch.random.manual_seed(random_seed)

    print(f"Chargement du jeu de données {dataset_str}...")

    if dataset_str == 'cifar10':
        cifar_mean = (0.4914, 0.4822, 0.4465)
        cifar_std = (0.2023, 0.1994, 0.2010)
        t = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(cifar_mean, cifar_std)])
        dataset_train = torchvision.datasets.CIFAR10(root='Datasets/CIFAR10', train=True, download=True, transform=t)
        # NOTE : We will split the test set into validation / test because we want to use pretrained models that will be used on the whole training set
        dataset_test = torchvision.datasets.CIFAR10(root='Datasets/CIFAR10', train=False, download=True, transform=t)
        # NOTE : here train_ratio will be the size of the validation set (not very pretty)
        dataset_validation, dataset_test = torch.utils.data.random_split(
            dataset_test, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

        

    elif dataset_str == 'cifar100':
        cifar_mean = (0.5071, 0.4867, 0.4408)
        cifar_std = (0.2673, 0.2564, 0.2761)
        t = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(cifar_mean, cifar_std)])
        dataset_train = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train=True, download=True, transform=t)
        # NOTE : We will split the test set into validation / test because we want to use pretrained models that will be used on the whole training set
        dataset_test = torchvision.datasets.CIFAR100(root='Datasets/CIFAR100', train=False, download=True, transform=t)
        # NOTE : here train_ratio will be the size of the validation set (not very pretty)
        dataset_validation, dataset_test = torch.utils.data.random_split(
            dataset_test, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

    elif dataset_str == 'svhn':
        pre_process = transforms.Compose([transforms.ToTensor()])
        dataset_train = torchvision.datasets.SVHN(root='Datasets/SVHN', split='train', download=True,
                                                  transform=pre_process)
        dataset_test = torchvision.datasets.SVHN(root='Datasets/SVHN', split='test', download=True,
                                                 transform=pre_process)
        dataset_train, dataset_validation = torch.utils.data.random_split(
            dataset_train, [train_ratio, 1.0 - train_ratio], generator=torch.Generator().manual_seed(random_seed))

    else:
        raise ValueError(f"Dataset {dataset_str} non pris en charge.")

    print("Jeu de données chargé avec succès.")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size_validation, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size_test, shuffle=True)

    return train_loader, validation_loader, test_loader
