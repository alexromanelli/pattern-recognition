from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

def lerResultadosCorretos(file_content) -> tuple[list[tuple[int, int]],
                                                 list[tuple[int, int]],
                                                 list[tuple[int, int, int, int]]]:
    plate_corners_line = 0
    plate_corners = []
    char_corners = []
    for line in file_content.split('\n'):
        if line.startswith('plate: '):
            plate = line[7:]
        elif line.startswith('corners: '):
            plate_corners_line = line[9:]
            corners = plate_corners_line.split(' ')
            for corner in corners:
                x, y = corner.split(',')
                plate_corners.append((int(x), int(y)))
        # chars corners
        elif line.lstrip().startswith('char '):
            char_corners_str = line.lstrip()[8:].split(' ')
            char_corners.append((int(char_corners_str[0]), 
                                 int(char_corners_str[1]), 
                                 int(char_corners_str[2]), 
                                 int(char_corners_str[3])))
    return (plate, plate_corners, char_corners)

def criarMatrizes(proposito: str) -> tuple[np.ndarray, np.ndarray]:
    X_data = np.array([[]])         # all data of training images, the input
    Y_data = np.array([[]])         # all the correct output for the training images
    validation_images = np.array([])
    testing_images = np.array([])

    # X_data = np.load('./data/X_data.csv.npy', 'r')
    # print(X_data.shape)

    mypath = f"./data/UFPR-ALPR dataset/{proposito}/"
    tracks = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
    tracks.sort()
    i = 0
    j = 0
    for track in tracks:
        mypath2 = join(mypath, track)
        files = [f for f in listdir(mypath2) if isfile(join(mypath2, f))]
        files.sort()
        for file in files:
            if file.endswith(".png"):
                image = Image.open(join(mypath2, file))
                numpydata = np.array(image)
                if numpydata.shape != (1080, 1920, 3):
                    print(file)
                    exit(0)
                # img = Image.fromarray(numpydata)
                # img.save(join(mypath2, 'teste.png'), 'png')
                # print(numpydata.shape)
                if X_data.size == 0:
                    X_data = np.array([numpydata])
                else:
                    X_data = np.vstack((X_data, [numpydata]))
                i += 1
                # print(i)
                # print(X_data.shape)
            elif file.endswith(".txt"):
                file = open(join(mypath2, file), "r")
                plate, plate_corners, chars_corners = lerResultadosCorretos(file.read())
                corner_x1 = plate_corners[0][0]
                corner_y1 = plate_corners[0][1]
                corner_x2 = plate_corners[1][0]
                corner_y2 = plate_corners[1][1]
                corner_x3 = plate_corners[2][0]
                corner_y3 = plate_corners[2][1]
                corner_x4 = plate_corners[3][0]
                corner_y4 = plate_corners[3][1]
                if Y_data.size == 0:
                    Y_data = np.array([[corner_x1, corner_y1, 
                                        corner_x2, corner_y2, 
                                        corner_x3, corner_y3, 
                                        corner_x4, corner_y4]])
                else:
                    Y_data = np.append(Y_data, 
                                        np.array([[corner_x1, corner_y1, 
                                                   corner_x2, corner_y2, 
                                                   corner_x3, corner_y3, 
                                                   corner_x4, corner_y4]]),
                                        axis=0)
                j += 1
                # print(j)
                # print(Y_data.shape)

    np.save(f'./data/X_{proposito}.csv', X_data)
    np.save(f'./data/Y_{proposito}.csv', Y_data)
    print(X_data.shape)
    print(Y_data.shape)

if __name__ == "__main__":
    # criarMatrizes('training')
    # criarMatrizes('testing')
    criarMatrizes('validation')
