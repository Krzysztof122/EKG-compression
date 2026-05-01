from typing import List
import torch
import torch.nn as nn


#model dziedziczacy po nn.Module (standardowy sposob tworzenia sieci w pytorch)
class Net(nn.Module):
    #konstruktor przyjmuje dwie listy z warstwami nn.Module
    def __init__(self,
                 encoder_siec: List[nn.Module],
                 decoder_siec: List[nn.Module]):
        super().__init__()
        self.encoder = nn.Sequential(*encoder_siec)     #korzystajac z nn.Sequential tworze dwie sieci - encoder i decoder
        self.decoder = nn.Sequential(*decoder_siec)

    def forward(self,x: torch.Tensor) -> torch.Tensor:  #forward w moim modelu - przechodze przez encoder i decoder
        kompresja = self.encoder(x)
        rekonstrukcja = self.decoder(kompresja)
        return rekonstrukcja
    

#funkcja przeprowadza jedna epoke uczenia, zwraca srednia wartosc funkcji strat
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0

    for batch, data in enumerate(dataloader):
        X = data[0]
        
        # Przesyłamy X przez autoenkoder (kompresja -> dekompresja)
        pred = model(X)

        # Liczymy błąd: jak bardzo odtworzony sygnał różni się od oryginału X
        loss = loss_fn(pred, X)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch % 100 == 0:
            loss_val, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

    return total_loss / len(dataloader)


#funkcja testujaca - zwraca srednie MSE
def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data in dataloader:
            X = data[0]
            pred = model(X)
            test_loss += loss_fn(pred, X).item()

    test_loss /= num_batches
    print(f"Test Error (Avg MSE Loss): {test_loss:>8f} \n")
    return test_loss