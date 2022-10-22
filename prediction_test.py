import cv2
import tensorflow as tf
import os

values = ["Demented", "NonDemented"]

IMG_SIZE = 36


def transform_test_data(filepath):
    global IMG_SIZE
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


def test_prediction(model):
    d_correct = 0
    d_total = 0
    n_correct = 0
    n_total = 0

    for img in os.listdir("mri/Alzheimer_s Dataset/test/Demented"):
        if d_total > 200:
            break
        d_total += 1
        prediction = model.predict([transform_test_data("mri/Alzheimer_s Dataset/test/Demented/" + img)])[0][0]
        if prediction < 0.5:
            d_correct += 1

    for img in os.listdir("mri/Alzheimer_s Dataset/test/NonDemented"):
        if n_total > 200:
            break
        n_total += 1
        prediction = model.predict([transform_test_data("mri/Alzheimer_s Dataset/test/NonDemented/" + img)])[0][0]
        if prediction > 0.5:
            n_correct += 1

    return d_correct, d_total, n_correct, n_total


def get_accuracy(d_c, d_t, n_c, n_t):
    print("Demented correctness: " + str(d_c), str(d_t), str(d_c / d_t))
    print("Non-demented correctness: " + str(n_c), str(n_t), str(n_c / n_t))
    print("Accuracy: " + str((d_c + n_c) / (d_t + n_t)))


d_c, d_t, n_c, n_t = test_prediction(tf.keras.models.load_model("CNN76%size36.model"))
get_accuracy(d_c, d_t, n_c, n_t)
