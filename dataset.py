from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size, augment=False):
    transform_list = [transforms.ToTensor()]

    # augmentation
    if augment:
        transform_list.insert(0, transforms.RandomRotation(10))

    # Normalizing dataset
    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
