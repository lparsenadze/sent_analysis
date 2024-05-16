import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import logging

from src.serving import export_model
from src import config

logger = logging.getLogger(__name__)


def get_data():
    data = pd.read_json(config.DATA_FILE, lines=True)
    data.drop_duplicates(subset=['overall', 'reviewText'], inplace=True)
    data.dropna(subset=['overall', 'reviewText'], inplace=True)

    one_enc = OneHotEncoder()
    labels = one_enc.fit_transform(np.array(data['overall']).reshape((-1, 1))).todense()
    texts = np.array(data['reviewText'])

    return texts, labels


def get_vectorizer(texts, max_tokens):
    sequence_length = 50
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=sequence_length)

    vectorize_layer.adapt(texts)

    return vectorize_layer


def compile_train(vectorize_layer, max_tokens, texts, labels):
    model = tf.keras.models.Sequential([
        vectorize_layer,
        tf.keras.layers.Embedding(max_tokens, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 classes, so softmax activation
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Use categorical crossentropy for one-hot encoded labels
                  metrics=['accuracy'])

    # Train the model
    model.fit(texts, labels, epochs=50, batch_size=32)

    return model


def main():
    texts, labels = get_data()
    logger.info("Data processed.")

    vectorize_layer = get_vectorizer(texts, config.MAX_TOKENS)
    logger.info("Vectorization layer prepared.")

    model = compile_train(vectorize_layer, config.MAX_TOKENS, texts, labels)
    logger.info("Model compiled and trained.")

    export_model(model, config.MODELS_PATH)
    logger.info("Model saved.")


if __name__=="__main__":
    main()
