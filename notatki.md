# EKG Compression
1. korzystamy z pliku training.py - wywołujemy i podajemy rozmiar najmniejszej warstwy
    skrypt tworzy model i uczy go - generuje plik autoencoder_latent_<rozmiar>.pth

2. aby przetestować wytrenowaną wersję sieci, mozemy wywołać visualize.py, tylko w kodzie trzeba zmienić wartość latent
    w efekcie dostajemy przykładowe trzy zapisy EKG ze skompresowaną wersją.

## files
- Net.py - główna klasa modelu + dwie metody - train i test

- DataPreprocessor.py - pobiera dane i skaluje odpowiednio

## trained models
przykładowe 4 pliki z wytrenowanymi sieciami - 16, 40, 100 i 128 neuronów w środkowej warstwie
- autoencoder_latent_16.pth
- autoencoder_latent_40.pth
- autoencoder_latent_100.pth
- autoencoder_latent_128.pth

## changes with collab
- zmieniłem preprocessing danych, żeby były w zakresie od 0 do 1
- zmieniłem architekturę sieci tak, aby pasowała do nowego preprocessingu
- obudowałem w klasy
- teraz wydaje mi sie ze MSE wychodzi dużo niższe po tych zmianach głównie jeśli chodzi o preprocessing

# Usage
-------------jak używać kompresora--------------
Kompresja:

python compressor.py compress ekg.npy compressed.pkl 32 autoencoder_latent_32.pth

Dekompresja:

python compressor.py decompress compressed.pkl restored.npy 32 autoencoder_latent_32.pth

# TODO :
- [ ] dopisać coś w stylu pliku kompresor.py, taki "użytkowy", w którym można skompresować odczyt EKG lub zdekompresować odczyt EKG
- [ ] jakoś ten kod bardziej uporządkować
