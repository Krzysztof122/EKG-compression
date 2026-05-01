import torch
import matplotlib.pyplot as plt
from DataPreprocessor import DataPreprocessor
from Net import Net
from training import stworz_warstwy

def plot_ekg_reconstruction(model, dataloader, scaler, num_samples=3):
    model.eval() # Tryb ewaluacji
    
    # Pobieramy jedną paczkę danych
    dataiter = iter(dataloader)
    X_batch = next(dataiter)[0]
    
    with torch.no_grad():
        # Kompresja i dekompresja
        reconstructed = model(X_batch)
    
    # Zamieniamy tensory z powrotem na numpy arrays
    X_batch_np = X_batch.numpy()
    reconstructed_np = reconstructed.numpy()
    
    # ODWROTNIE SKALUJEMY
    X_original = scaler.inverse_transform(X_batch_np)
    X_reconstructed = scaler.inverse_transform(reconstructed_np)
    
    # Rysujemy wykresy
    fig, axes = plt.subplots(nrows=num_samples, ncols=1, figsize=(10, 3 * num_samples))
    
    if num_samples == 1:
        axes = [axes]
        
    for i in range(num_samples):
        axes[i].plot(X_original[i], label="Oryginał (X)", color="blue", alpha=0.7)
        axes[i].plot(X_reconstructed[i], label="Rekonstrukcja (Predykcja)", color="red", linestyle="--")
        axes[i].set_title(f"Próbka EKG nr {i+1}")
        axes[i].legend()
        axes[i].grid(True)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    latent_dim = 128
    
    # 1. Przygotowujemy dane
    dp = DataPreprocessor()
    _, test_loader = dp.dataToLoader(batchSize=32)
    
    # 2. Tworzymy architekturę i ładujemy wagi
    encoder_net, decoder_net = stworz_warstwy(latent_dim)
    model = Net(encoder_net, decoder_net)
    
    nazwa_pliku = f"autoencoder_latent_{latent_dim}.pth"
    try:
        model.load_state_dict(torch.load(nazwa_pliku, weights_only=True))
        print(f"Załadowano wagi z {nazwa_pliku}")
    except FileNotFoundError:
        print(f"Brak pliku {nazwa_pliku}. Najpierw wytrenuj model!")
        exit()
        
    # 3. Rysujemy
    plot_ekg_reconstruction(model, test_loader, dp.scaler, num_samples=3)