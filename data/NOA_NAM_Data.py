from torch.utils.data import Dataset
from torchvision import transforms
from data.mongo_datafetch_utils import fetch_training_data_MONGO, training_labels, target_labels
import pandas as pd

class NOAA_Data(Dataset):

    # RETURNS A HIGH RES VERSION OF THE IMAGE AT A GIVEN INDEX
    def __getitem__(self, index):
        ret = {}
        ret['X'] = self.X.iloc[index].values
        ret['Y'] = self.Y.iloc[index].values
        return ret

    def __len__(self):
        return self.data_size

    def __init__(self, geo_identifier):
        training_data = fetch_training_data_MONGO(geo_identifier)

        training_data = pd.DataFrame(training_data)

        X = training_data[training_labels]
        Y = training_data[target_labels]

        self.data_size = X.shape[0]
        self.X = X
        self.Y = Y
        self.tensorise_fn = transforms.Compose([transforms.ToTensor()])





