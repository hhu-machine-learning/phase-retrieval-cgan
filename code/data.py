import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from PIL import Image
import io
import h5py
import torchvision.transforms as transforms

# Custom Dataset for loading the CelebA H5 file containing all images as jpeg
class CelebAH5(torch.utils.data.Dataset):
    def __init__(self, h5file, transform):
        super(CelebAH5, self).__init__()
        self.h5file = h5file
        self.n_images = self.h5file['images'].shape[0]
        self.transform = transform

    def __getitem__(self, index):
        bin_data =  self.h5file['images'][index]
        return self.transform(Image.open(io.BytesIO(bin_data)))

    def __len__(self):
        return self.n_images

        
# Custom Dataset for loading the croped CelebA64 H5 file
class CelebA64H5(torch.utils.data.Dataset):
    def __init__(self, h5file, split, transforms):
        super(CelebA64H5, self).__init__()
        self.h5file = h5file
        self.split = split
        self.transforms = transforms

    def __getitem__(self, idx):
        return self.transforms(self.h5file[self.split][idx])

    def __len__(self):
        return len(self.h5file[self.split])

# Discards all meta information expect the image (As needed for MNIST)  
class ImageOnly(torch.utils.data.Dataset):

    def __init__(self, orig_dataset):
        self.orig_dataset = orig_dataset

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        return self.orig_dataset[idx][0]
        
# Linearly samples from a dataset without shuffle (Needed for linear validation)
class LinearSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)        

        
def load(name='MNIST', path="", batch_size=64, num_workers=1, in_memory=False, normalization=[]):
    
    if name[:6].lower() == 'celeba':

        if name[6:].lower() == '64':

            trans = transforms.Compose([
                transforms.ToTensor(),
            ]+normalization)
            
            if in_memory:
                h5file = h5file = h5py.File(path, 'r', driver='core')
            else:
                h5file = h5py.File(path, 'r')
            
            trainset = CelebA64H5(h5file, split='train', transforms=trans)
            valset = CelebA64H5(h5file, split='val', transforms=trans)
            testset = CelebA64H5(h5file, split='test', transforms=trans)
            
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True
            )
            
            validloader = torch.utils.data.DataLoader(
                valset, batch_size=batch_size,
                num_workers=num_workers, shuffle=False
            )
            
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size,
                num_workers=num_workers, shuffle=False
            )
            
            imsize = (3, 64, 64)

        else:
            
            if name[6:].lower() == 'pad':
                trans = transforms.Compose([
                    transforms.CenterCrop((108, 108)),
                    transforms.Resize(64),
                    transforms.Pad(32, 0),
                    transforms.ToTensor(), 
                ]+normalization)
            else: 
                trans = transforms.Compose([
                transforms.CenterCrop((108, 108)),
                transforms.Resize(64),
                transforms.ToTensor(),
            ]+normalization)

            if in_memory:
                h5file = h5file = h5py.File(path, 'r', driver='core')
            else:
                h5file = h5py.File(path, 'r')
            
            dataset = CelebAH5(h5file, transform=trans)

            indices = list(range(202589))
            train_idx, valid_idx, test_idx = indices[:162769], indices[162769:182636], indices[182636:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = LinearSampler(valid_idx)
            test_sampler = LinearSampler(test_idx)

            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=num_workers)
                
            validloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=valid_sampler,
                num_workers=num_workers, shuffle=False)

            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=test_sampler,
                num_workers=num_workers, shuffle=False)

            imsize = (3, 64, 64)
            
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize
        
    if name[:5].lower() == 'mnist':
        
        if name[5:].lower() == 'rnd':
            train_trans = transforms.Compose([
                transforms.RandomAffine((-15, 15), scale=(0.8, 1.1), shear=(-30, 30), resample=Image.BICUBIC),
                transforms.ToTensor(),
            ]+normalization)
            val_trans = transforms.ToTensor()
            test_trans = transforms.ToTensor()
        
        elif name[5:].lower() == 'rotate':
            train_trans = transforms.Compose([
                transforms.RandomChoice([transforms.RandomRotation((180, 180)), transforms.RandomRotation((0, 0))]),
                transforms.ToTensor(),
            ]+normalization)
            val_trans = transforms.ToTensor()
            test_trans = transforms.ToTensor()
        
        elif name[5:].lower() == 'pad':
            train_trans = val_trans = test_trans = transforms.Compose([
                transforms.Pad(14, 0),
                transforms.ToTensor(), 
            ]+normalization)
        
        elif name[5:].lower() == '64':
            train_trans = val_trans = test_trans = trans = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor()
            ]+normalization)
        
        else:
            train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
            val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
            test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = torchvision.datasets.MNIST(
            root=path, train=True,
            download=True,
            transform=train_trans
        )
        valset = torchvision.datasets.MNIST(
            root=path, train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = torchvision.datasets.MNIST(
            root=path, train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2000:], indices[:-2000] # Last 2000: val, First 8000 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            num_workers=num_workers, shuffle=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, shuffle=False
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, shuffle=False
        )

        imsize = (1, 28, 28)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize

    if name.lower() == 'fashion':
        print(name)
        
        train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = torchvision.datasets.FashionMNIST(
            root=path, train=True,
            download=True,
            transform=train_trans
        )
        valset = torchvision.datasets.FashionMNIST(
            root=path, train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = torchvision.datasets.FashionMNIST(
            root=path, train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2000:], indices[:-2000] # Last 2000: val, First 8000 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            num_workers=num_workers, shuffle=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, shuffle=False
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, shuffle=False
        )

        imsize = (1, 28, 28)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize
    
    if name.lower() == 'cifar':
        print(name)
        
        train_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        val_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        test_trans = transforms.Compose([transforms.ToTensor()]+normalization)
        
        trainset = torchvision.datasets.CIFAR10(
            root=path, train=True,
            download=True,
            transform=train_trans
        )
        valset = torchvision.datasets.CIFAR10(
            root=path, train=False,
            download=True,
            transform=val_trans
        )                                      
        testset = torchvision.datasets.CIFAR10(
            root=path, train=False,
            download=True,
            transform=test_trans
        )
        
        trainset = ImageOnly(trainset)
        valset = ImageOnly(valset)
        testset = ImageOnly(testset)
        
        indices = list(range(10000))
        valid_idx, test_idx = indices[-2000:], indices[:-2000] # Last 2000: val, First 8000 test
        valid_sampler = LinearSampler(valid_idx)
        test_sampler = LinearSampler(test_idx)
        
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size,
            num_workers=num_workers, shuffle=True
        )
            
        validloader = torch.utils.data.DataLoader(
            valset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, shuffle=False
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, sampler=test_sampler,
            num_workers=num_workers, shuffle=False
        )

        imsize = (3, 32, 32)
        
        return {'train': trainloader, 'val': validloader, 'test': testloader}, imsize

    print("{} did not match any known dataset".format(name))
    return None

