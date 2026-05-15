import argparse
import pickle
import numpy as np
import torch

from Net import Net
from training import stworz_warstwy


class EKGCompressor:
    def __init__(self, latent_dim: int, model_path: str):
        encoder_net, decoder_net = stworz_warstwy(latent_dim)

        self.model = Net(encoder_net, decoder_net)

        self.model.load_state_dict(
            torch.load(model_path, weights_only=True)
        )

        self.model.eval()

    def compress(self, ekg_signal: np.ndarray) -> np.ndarray:
        """
        Kompresja sygnału EKG do latent vector.
        """

        if ekg_signal.shape[0] != 187:
            raise ValueError(
                f"Sygnał musi mieć długość 187, otrzymano {ekg_signal.shape[0]}"
            )

        with torch.no_grad():
            x = torch.tensor(ekg_signal).float().unsqueeze(0)

            latent = self.model.encoder(x)

        return latent.squeeze(0).numpy()

    def decompress(self, latent_vector: np.ndarray) -> np.ndarray:
        """
        Dekompresja latent vector do sygnału EKG.
        """

        with torch.no_grad():
            z = torch.tensor(latent_vector).float().unsqueeze(0)

            reconstruction = self.model.decoder(z)

        return reconstruction.squeeze(0).numpy()


def save_compressed(latent_vector, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(latent_vector, f)


def load_compressed(input_path):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode")

    compress_parser = subparsers.add_parser("compress")
    compress_parser.add_argument("input")
    compress_parser.add_argument("output")
    compress_parser.add_argument("latent_dim", type=int)
    compress_parser.add_argument("model")

    decompress_parser = subparsers.add_parser("decompress")
    decompress_parser.add_argument("input")
    decompress_parser.add_argument("output")
    decompress_parser.add_argument("latent_dim", type=int)
    decompress_parser.add_argument("model")

    args = parser.parse_args()

    compressor = EKGCompressor(
        latent_dim=args.latent_dim,
        model_path=args.model
    )

    if args.mode == "compress":

        ekg = np.load(args.input)

        latent = compressor.compress(ekg)

        save_compressed(latent, args.output)

        print("Kompresja zakończona.")
        print(f"Rozmiar latent vector: {latent.shape}")

    elif args.mode == "decompress":

        latent = load_compressed(args.input)

        reconstructed = compressor.decompress(latent)

        np.save(args.output, reconstructed)

        print("Dekompresja zakończona.")
        print(f"Zrekonstruowany sygnał zapisano do {args.output}")


if __name__ == "__main__":
    main()