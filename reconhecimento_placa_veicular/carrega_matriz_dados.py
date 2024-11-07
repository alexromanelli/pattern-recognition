from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

def carregarMatrizes(proposito: str) -> tuple[np.ndarray, np.ndarray]:
    X_data = np.load(f'./data/X_{proposito}.csv.npy', 'r')
    Y_data = np.load(f'./data/Y_{proposito}.csv.npy', 'r')
    return (X_data, Y_data)

def main():
    X_data, Y_data = carregarMatrizes('training')
    print(X_data.shape)
    print(Y_data.shape)

if __name__ == "__main__":
    main()
