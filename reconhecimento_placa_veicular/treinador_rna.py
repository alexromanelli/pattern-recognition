from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD


optimizer = SGD()
loss='categorical_crossentropy'

def criarModeloRna():
    model = Sequential()
    return model

if __name__ == "__main__":
    m = criarModeloRna()
    m.compile(loss=loss,optimizer=optimizer)

    print(m.summary())
