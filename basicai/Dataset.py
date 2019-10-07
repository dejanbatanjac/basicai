class Dataset(object):
    'abstract class Dateset'

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        
    def tv(self):
        'train and validation split of indices in return'
        raise NotImplementedError    


D = Dataset  