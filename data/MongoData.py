from torch.utils.data import Dataset
from torchvision import transforms

class MongoData(Dataset):

    # RETURNS A HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        ret = {}
        ret['X'] = self.X.iloc[index].values
        ret['Y'] = self.Y.iloc[index].values

        #ret['X'] = self.tensorise_fn(ret['X'])
        #ret['Y'] = self.tensorise_fn(ret['Y'])
        return ret

    def __len__(self):
        return self.data_size

    def __init__(self, X, Y, base_model = None, transfer = False):
        self.data_size = X.shape[0]
        self.X = X
        self.Y = Y
        self.tensorise_fn = transforms.Compose([
            transforms.ToTensor()
        ])
        if transfer:
            self.base_model = base_model




