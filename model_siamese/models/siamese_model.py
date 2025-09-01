from keras.models import Model
from keras.layers import Input, Lambda, Dense
import tensorflow as tf
from models.embedding_network import make_embedding

def make_siamese_model():
    input_img = Input(name='input_img', shape=(105, 105, 3))
    validation_img = Input(name='validation_img', shape=(105, 105, 3))

    embedding = make_embedding()
    emb1 = embedding(input_img)
    emb2 = embedding(validation_img)

    l1_distance = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([emb1, emb2])
    output = Dense(1, activation='sigmoid')(l1_distance)

    return Model(inputs=[input_img, validation_img], outputs=output, name='SiameseNetwork')
