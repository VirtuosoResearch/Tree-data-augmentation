from torchvision import transforms

class MatchChannel(object):
    def __call__(self, pic):
        if pic.size()[0] == 1:
            assert len(pic.size()) == 3
            pic = pic.repeat(3,1,1)
        return pic

class BasicTransform():

    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, x):
        return self.transform(x)

class TrainTransform(BasicTransform):

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            MatchChannel(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
class TestTransform(BasicTransform):

    def __init__(self, size=224):
        self.transform =  transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            MatchChannel(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
class CIFARTrainTransform(BasicTransform):

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
        ])

class CIFARTestTransform(BasicTransform):

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
