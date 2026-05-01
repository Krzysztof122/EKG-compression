import kagglehub
import pandas as pd
import os
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.preprocessing import MinMaxScaler

#klasa do obslugi danych
#przy inicjalizacji danych pobiera dane
#wywolanie dataToLoader(batchSize) zwroci dwa dataloadery - train i test

class DataPreprocessor:
    def __init__(self):         #downloading data
        dataset_path = kagglehub.dataset_download("shayanfazeli/heartbeat")
        train_file_path = os.path.join(dataset_path, "mitbih_train.csv")
        test_file_path = os.path.join(dataset_path, "mitbih_test.csv")

        train = pd.read_csv(train_file_path, header=None)
        test = pd.read_csv(test_file_path, header=None)

        self.X_train_data = train.iloc[:, :-1].values   #biore wszytkie kolumny bez ostatniej bo labele mnie nie interesuja
        self.X_test_data = test.iloc[:, :-1].values

        self.scaler = MinMaxScaler()

    def dataToLoader(self, batchSize):
        self.X_train_data = self.scaler.fit_transform(self.X_train_data)
        self.X_test_data = self.scaler.transform(self.X_test_data)
        
        X_train_tensor = torch.from_numpy(self.X_train_data).float()
        X_test_tensor = torch.from_numpy(self.X_test_data).float()
        
        train_dataset = TensorDataset(X_train_tensor)
        test_dataset = TensorDataset(X_test_tensor)
        
        train_dataloader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batchSize)
        
        return train_dataloader, test_dataloader



    
