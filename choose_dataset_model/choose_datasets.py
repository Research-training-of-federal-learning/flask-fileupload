import torchvision
from torchvision.transforms import transforms

def choosedatasets(dataname,root):
    transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
    if(dataname=="CALTECH101"):
        dataset=torchvision.datasets.Caltech101(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="Caltech256"):
        dataset=torchvision.datasets.Caltech256(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="EUROSAT"):
        dataset=torchvision.datasets.EuroSAT(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="FAKEDATA"):
        dataset=torchvision.datasets.FakeData(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="FLICKR8K"):
        dataset=torchvision.datasets.Flickr8k(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="FLICKR30K"):
        dataset=torchvision.datasets.Flickr30k(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="INATURALIST"):
        dataset=torchvision.datasets.INaturalist(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="OMNIGLOT"):
        dataset=torchvision.datasets.Omniglot(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="SEMEION"):
        dataset=torchvision.datasets.SEMEION(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="SBU"):
        dataset=torchvision.datasets.SBU(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])
    elif(dataname=="SUN397"):
        dataset=torchvision.datasets.SUN397(
            root=root,
            download=True,
            transform=transform_train)
        train_ratio = 0.8
        test_ratio = 1 - train_ratio
        num_samples = len(dataset)
        num_train_samples = int(train_ratio * num_samples)
        train_dataset, test_dataset = random_split(dataset, [num_train_samples, num_samples - num_train_samples])



    elif(dataname=="CELEBA"):
        train_dataset = torchvision.datasets.CelebA(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.CelebA(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="COUNTRY211"):
        train_dataset = torchvision.datasets.Country211(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.Country211(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="DTD"):
        train_dataset = torchvision.datasets.DTD(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.DTD(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="FER2013"):
        train_dataset = torchvision.datasets.FER2013(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.FER2013(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="FGVCAIRCRAFT"):
        train_dataset = torchvision.datasets.FGVCAircraft(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.FGVCAircraft(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="FLOWERS102"):
        train_dataset = torchvision.datasets.Flowers102(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.Flowers102(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="FOOD101"):
        train_dataset = torchvision.datasets.Food101(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.Food101(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="IMAGENET"):
        train_dataset = torchvision.datasets.ImageNet(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.ImageNet(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="LSUN"):
        train_dataset = torchvision.datasets.LSUN(
            root=root,
            classes="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.LSUN(
            root=root,
            classes="test",
            download=True,
            transform=transform_train)
    elif(dataname=="OXFORDIIITPET"):
        train_dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.OxfordIIITPet(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="PLACES365"):
        train_dataset = torchvision.datasets.Places365(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.Places365(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="PCAM"):
        train_dataset = torchvision.datasets.PCAM(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.PCAM(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="RENDEREDSST2"):
        train_dataset = torchvision.datasets.RenderedSST2(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.RenderedSST2(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="STANFORDCARS"):
        train_dataset = torchvision.datasets.StanfordCars(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.StanfordCars(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="STL10"):
        train_dataset = torchvision.datasets.STL10(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.STL10(
            root=root,
            split="test",
            download=True,
            transform=transform_train)
    elif(dataname=="SVHN"):
        train_dataset = torchvision.datasets.SVHN(
            root=root,
            split="train",
            download=True,
            transform=transform_train)

        test_dataset = torchvision.datasets.SVHN(
            root=root,
            split="test",
            download=True,
            transform=transform_train)




    elif(dataname=="CIFAR10"):
        train_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="CIFAR100"):
        train_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="EMNIST"):
        train_dataset = torchvision.datasets.EMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.EMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="FASHIONMNIST"):
        train_dataset = torchvision.datasets.FashionMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.FashionMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="KMNIST"):
        train_dataset = torchvision.datasets.KMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.KMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="LFWPEOPLE"):
        train_dataset = torchvision.datasets.LFWPeople(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.LFWPeople(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="QMNIST"):
        train_dataset = torchvision.datasets.QMNIST(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.QMNIST(
            root=root,
            train=False,
            download=True,
            transform=transform_train)
    elif(dataname=="USPS"):
        train_dataset = torchvision.datasets.USPS(
            root=root,
            train=True,
            download=True,
            transform=transform_train)
        test_dataset = torchvision.datasets.USPS(
            root=root,
            train=False,
            download=True,
            transform=transform_train)

    elif(dataname=="freestyle"):
        pass



    return train_dataset,test_dataset