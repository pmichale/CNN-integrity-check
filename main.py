# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    return img


# load an image and predict the class
def run_example():
    # load the image
    path = './test/bad/bad68.png'
    # load model
    model = load_model('models/not_trainable/EfficientNetB0.h5')
    # predict the class

    prediction = load_image(path)
    result = model.predict(prediction)
    if result[0] <= 0.5:
        print("Poskozeno:", 0)
    else:
        print("Neposkozeno:", 1)
    from PIL import Image
    image = Image.open(path)
    image.show()


# entry point, run the example
run_example()
