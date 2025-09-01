from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from keras.layers import BatchNormalization, Dropout, ReLU
from keras.regularizers import l2

def make_embedding():
    inp = Input(shape=(105, 105, 3), name='input_image')

    x = Conv2D(64, (10, 10), padding='valid', kernel_regularizer=l2(2e-4))(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (7, 7), padding='valid', kernel_regularizer=l2(2e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (4, 4), padding='valid', kernel_regularizer=l2(2e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (4, 4), padding='valid', kernel_regularizer=l2(2e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)

    x = Dense(512, activation='relu', kernel_regularizer=l2(2e-4))(x)
    x = Dropout(0.4)(x)

    return Model(inputs=inp, outputs=x, name='EmbeddingNetwork')
