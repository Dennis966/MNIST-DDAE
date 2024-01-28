import pickle
from torch.utils.data import Dataset

### Training set
class NoisyCleanTrainSet(Dataset):
    def __init__(self):
        super().__init__()
        with open("train_clean.pickle", "rb") as train_clean_file:
            self.train_clean_dataset = pickle.load(train_clean_file)

        with open("train_noisy.pickle", "rb") as train_noisy_file:
            self.train_noisy_dataset = pickle.load(train_noisy_file)

    def __len__(self):
        assert len(self.train_clean_dataset) == len(self.train_noisy_dataset)
        return len(self.train_noisy_dataset)

    def __getitem__(self, item):
        return (self.train_noisy_dataset[item], self.train_clean_dataset[item])

### Testing set
class NoisyCleanTestSet(Dataset):
    def __init__(self):
        super().__init__()
        with open("test_clean.pickle", "rb") as test_clean_file:
            self.test_clean_dataset = pickle.load(test_clean_file)

        with open("test_noisy.pickle", "rb") as test_noisy_file:
            self.test_noisy_dataset = pickle.load(test_noisy_file)

    def __len__(self):
        assert len(self.test_clean_dataset) == len(self.test_noisy_dataset)
        return len(self.test_noisy_dataset)

    def __getitem__(self, item):
        return (self.test_noisy_dataset[item],self.test_clean_dataset[item])