import pandas as pd
import tensorflow as tf

BATCH_SIZE = 10
FILE_NAME = "pima-indians-diabetes.data.txt"

# Categorical requires a little more care, I guess
NUMERICAL_FEATURE_NAMES = [
    "Times Pregnant",
    "Plasma Glucose Concentration",
    "Diastolic Blood Pressure",
    "Triceps Skin Fold Thickness",
    "Two-hour serum insulin",
    "BMI",
    "Diabetes Pedigree Function"
    "Age"
]

# this is a binary classification problem, so probably something like Softmax should work
# as a good first pass
LABEL_NAMES = [
    "Diabetic"
]

def heart_disease_classifier():
    df = pd.read_csv(
        tf.keras.utils.get_file(
            "heart.csv", 
            "https://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
        )
    )

    labels = df.pop("target")

    # whatever we can call tf.convert_to_tensor on can be passed into tf tensor things
    numeric_features = df[["age", "thalach", "trestbps", "chol", "oldpeak"]]
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(numeric_features) # let's see if we can port this over to the diabetes code
    """
    adapt() is meant only as a single machine utility to compute layer state.
    To analyze a dataset that cannot fit on a single machine, see Tensorflow
    Transform for a multi-machine, map-reduce solution.
    """

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(
        numeric_features,
        labels,
        epochs=150,
        batch_size=BATCH_SIZE
    )

def main():
    # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    # https://www.tensorflow.org/tutorials/load_data/csv
    column_names = []
    column_names.extend(NUMERICAL_FEATURE_NAMES)
    column_names.extend(LABEL_NAMES)

    diabetes_df = pd.read_csv(
        FILE_NAME, 
        header=None, 
        sep=',', 
        names=column_names
    )

    features_df = diabetes_df[NUMERICAL_FEATURE_NAMES]
    labels_df = diabetes_df[LABEL_NAMES]
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "./checkpoints"
    )

    
    model = tf.keras.models.Sequential()

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(features_df)

    model.add(normalizer)

    # I mean technically this first layer is two layers...
    # activation, again, being a non-linear 
    model.add(tf.keras.layers.Dense(12, activation='relu')) # There are the linear weights to turn the 8 numerical feature values to 12 hidden, then 
    model.add(tf.keras.layers.Dense(8, activation ='relu'))

    # one output at the end of the day (shouldn't this be a softmax regression hmm)
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  # I believe this was mentioned for softmax regression in particular
        optimizer='adam',
        metrics=['accuracy'],
    )


    # if tf can invoke tf.convert_to_tensor() on it, then it can be used
    # anywhere a tensor can
    model.fit(
        features_df, 
        labels_df, 
        epochs=150, 
        batch_size=2,
        callbacks=[checkpoint_cb]
    )

    model.predict([[8,125,96,0,0,0.0,0.232,54]])

if __name__ == "__main__":
    # main()
    heart_disease_classifier()
