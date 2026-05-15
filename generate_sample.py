import numpy as np

# przykładowy sygnał EKG (187 próbek)
ekg_signal = np.array([
    0.725, 0.812, 0.901, 0.954, 0.991, 1.000, 0.965, 0.873, 0.742,
    0.611, 0.503, 0.442, 0.401, 0.377, 0.361, 0.350, 0.344, 0.338,
    0.332, 0.325, 0.318, 0.312, 0.308, 0.305, 0.303, 0.300, 0.298,
    0.295, 0.293, 0.290, 0.288, 0.286, 0.284, 0.282, 0.281, 0.280,
] * 5 + [
    0.72, 0.70, 0.68, 0.65, 0.60, 0.55, 0.50
])

# upewnienie się że mamy dokładnie 187 elementów
ekg_signal = ekg_signal[:187]

np.save("ekg.npy", ekg_signal)

print("Zapisano ekg.npy")
print("Shape:", ekg_signal.shape)