import argparse
import torch
import torch.nn as nn
from typing import List, Tuple

from DataPreprocessor import DataPreprocessor
from Net import Net, train, test


#ta funkcja pozwala na proste tworzenie wielu architektur sieci "bottleneck"
#podaje argument z latent_dim, a funkcja tworzy plan encodera i decodera tak, ze schodzi 187 -> potegi dwojki -> latent_dim -> potegi dwojki -> 187

def stworz_warstwy(latent_dim: int) -> Tuple[List[nn.Module], List[nn.Module]]:

    # 1. walidacja argumentu
    if latent_dim >= 187:
        raise ValueError(f"Błąd: Zmienna ukryta ({latent_dim}) musi być mniejsza niż wejście (187), inaczej to nie kompresja!")
    if latent_dim < 2:
        raise ValueError(f"Błąd: Zbyt mała kompresja ({latent_dim}). Ustaw co najmniej 2.")


    # 2. LICZYMY SCHODKI (Potęgi dwójki)
    # Zaczynamy od 128 i dzielimy przez 2, dopóki schodek jest większy niż nasz cel (latent_dim)
    schodki = []
    obecny_rozmiar = 128
    
    while obecny_rozmiar > latent_dim:
        schodki.append(obecny_rozmiar)
        obecny_rozmiar = obecny_rozmiar // 2  # Dzielenie całkowite

    # Dla latent_dim = 50, schodki to: [128, 64]
    # Dla latent_dim = 16, schodki to: [128, 64, 32]

    # 3. BUDUJEMY ENKODER
    encoder_siec = []
    wejscie = 187
    
    for wyjscie in schodki:
        encoder_siec.extend([
            nn.Linear(wejscie, wyjscie),
            nn.BatchNorm1d(wyjscie),
            nn.GELU()
        ])
        wejscie = wyjscie # Wyjście poprzedniej warstwy staje się wejściem kolejnej
        
    # Ostatni skok z najniższego schodka do latent_dim
    encoder_siec.extend([nn.Linear(wejscie, latent_dim), nn.GELU()])

    # 4. BUDUJEMY DEKODER
    decoder_siec = []
    wejscie = latent_dim
    schodki_odwrotne = schodki[::-1] # Odwracamy schodki (np. z [128, 64] robimy [64, 128])
    
    for wyjscie in schodki_odwrotne:
        decoder_siec.extend([
            nn.Linear(wejscie, wyjscie),
            nn.GELU(),
            nn.BatchNorm1d(wyjscie) # W dekoderze zazwyczaj omijamy Dropout
        ])
        wejscie = wyjscie
        
    # Ostatni skok z najwyższego schodka do formatu oryginalnego (187)
    # Na samym końcu zostawiamy samo Linear (bez ReLU), bo znormalizowane dane EKG mogą mieć wartości ujemne
    decoder_siec.extend([nn.Linear(wejscie, 187),
                        nn.Sigmoid()])

    return encoder_siec, decoder_siec






# przyklad wywolania:
# python training.py 32 8
if __name__ == "__main__":

    #obsluga parametrow wywolania
    parser = argparse.ArgumentParser(description="Skrypt do testowania różnej siły kompresji EKG.")
    
    parser.add_argument(
        "latent", 
        type=int, 
        help="Rozmiar przestrzeni ukrytej (latent_dim), np. 16, 32, 50."
    )

    parser.add_argument(
        "epochs",
        type=int,
        help="Liczba epok do nauczenia",
        default=10
    )
    
    args = parser.parse_args()
    
    try:
        encoder_net, decoder_net = stworz_warstwy(args.latent)
        
        print("\n--- ENKODER ---")
        print(nn.Sequential(*encoder_net))
        
        print("\n--- DEKODER ---")
        print(nn.Sequential(*decoder_net))
        
        dp = DataPreprocessor()
        train_dataloader, test_dataloader = dp.dataToLoader(batchSize=32)

        model = Net(encoder_net, decoder_net)
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = args.epochs
        train_losses = []
        test_losses = []
        

        #zmienna do zapisu najlepszego wyniku
        best_test_loss = float('inf')
        nazwa_pliku = f"autoencoder_latent_{args.latent}.pth"

        for e in range(epochs):
            print(f"Epoch {e+1}\n-------------------------------")
            avg_train_loss = train(train_dataloader, model, loss_func, optimizer)
            avg_test_loss = test(test_dataloader, model, loss_func)

            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)

            #jesli po danej epoce test wyszedl najlepiej to zapisuje.
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                torch.save(model.state_dict(), nazwa_pliku)
                print("best one yet - saved")


        print("Uczenie zakończone!")
        print(f"Model zapisany jako: {nazwa_pliku}")



    except ValueError as e:
        print(f"\n[!!!] {e}\n")







