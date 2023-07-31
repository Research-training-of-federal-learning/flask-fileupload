from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

def freestyledataset()
	transform_test = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.55206233,0.44260582,0.37644434),(0.2515312,0.22786127,0.22155665)),
            ])
    dataset = ImageFolder(".data/PUBFIG/pubfig83",transform = transform_test)
    train_dataset, test_dataset = random_split(dataset= dataset, lengths=[11070, 2733])
    return train_dataset, test_dataset