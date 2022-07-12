import cv2 as cv
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Subtract, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import load_model

import util


class ReflectionRemovalEncoderDecoder():

    def __init__(self, input_shape, load=False, model_address="./model"):
        print("V1 initializing")
        self.input_shape = input_shape
        if load:
            self.model = load_model(model_address)
            return
        self.model: Model = self.build_model()

    @staticmethod
    def load(model_address="./model"):
        return load_model(model_address)

    def save_model(self, model_address="./model"):
        self.model.save(model_address)

    def evaluate(self, files, refs):
        files = files[672:-1]
        images = [cv.resize(cv.imread(file) / 255,
                            (self.input_shape[0], self.input_shape[1])) for file
                  in files]
        x = []
        y = []
        for image in images:
            tx, ty = util.generate_batch(image, refs, 4)
            x.append(tx)
            y.append(ty)
        x = np.array(x)
        y = np.array(y)
        return self.model.evaluate(x, y)

    def test(self, image):
        image = cv.resize(image, (self.input_shape[0], self.input_shape[1]))
        batch = np.expand_dims(image, axis=0)
        return self.model.predict(batch)[0]

    def train(self, all_image_files, reflection_files, save_address, epochs=5,
              batch_size=32):
        image_files = all_image_files[0:672]  # last 29 images for test and eval
        refs = [cv.imread(ref_file) / 255 for ref_file in reflection_files]
        for epoch in range(epochs // 5):
            for i in range(0, 4, 2):
                print(f"i is {i} at epoch {epoch}")
                I0 = cv.resize(cv.imread(image_files[i]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                I1 = cv.resize(cv.imread(image_files[i + 1]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x1, y1 = util.generate_batch(I0, refs, batch_size // 2)
                x2, y2 = util.generate_batch(I1, refs, batch_size // 2)
                x = np.array(x1 + x2, "float32").reshape((-1,
                                                          self.input_shape[1],
                                                          self.input_shape[0],
                                                          self.input_shape[2]))
                y = np.array(y1 + y2, "float32").reshape((-1,
                                                          self.input_shape[1],
                                                          self.input_shape[0],
                                                          self.input_shape[2]))
                self.model.fit(x, y, epochs=5, batch_size=32,)
            self.save_model(save_address + "/model" + str(epoch))
        self.save_model(save_address + "/model")
        # print(self.evaluate(all_image_files, refs))

    import cv2 as cv
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1],
                           flags=cv.INTER_LINEAR)
    return result


def random_flip(I):
    if np.random.random() >= 0.5:
        I = cv.flip(I, 1)
    if np.random.random() >= 0.7:
        I = cv.flip(I, 0)

    return I


def random_rotate(I):
    return rotate_image(I, np.round(np.random.random() * 45))


def random_rotate_and_flip(I):
    return random_rotate(random_flip(I))


def replace_image(I, mask, loc=(0, 0), w=0.1):
    mask *= w
    end = (np.minimum(I.shape[0], loc[0] + mask.shape[0]),
           np.minimum(I.shape[1], loc[1] + mask.shape[1]))
    I[loc[0]:end[0], loc[1]:end[1]] = I[loc[0]:end[0], loc[1]:end[1]] * (
            1 - w) + mask[:np.minimum(mask.shape[0], I.shape[0] - loc[0]),
                     :np.minimum(mask.shape[1], I.shape[1] - loc[1])]


def add_reflection(I, refs, ref_range=(0.5, 1.5), max_ref_count=6,
                   weight_range=(0.01, 0.4)):
    I = I.copy()
    start = (0, 0)
    end = (I.shape[0], I.shape[1])
    ref_indices = []
    for i in range(max_ref_count):
        ref_indices.append(
            np.floor(np.random.random() * len(refs)).astype("int"))
    at_least_one = False
    for ind in ref_indices:
        if at_least_one and np.random.random() > 0.7:
            continue
        at_least_one = True
        refl = refs[ind]
        scl = (ref_range[1] - ref_range[0]) * np.random.random() + ref_range[0]
        refl = resize(refl, scale=scl)
        refl = random_rotate_and_flip(refl)
        refl = blur(refl)
        randX = np.floor((end[0] - start[0]) * np.random.random() + start[0]).astype("int")
        randY = np.floor((end[1] - start[1]) * np.random.random() + start[1]).astype("int")
        w = (weight_range[1] - weight_range[0]) * np.random.random() + \
            weight_range[0]
        replace_image(I, refl, (randX, randY), w)
    return I


def generate_batch(I, refs, size=16):
    I1 = cv.flip(I.copy(), 1)  # other side of the road
    x = []
    y = []
    for i in range(size // 2):
        x.append(I)
        y.append(add_reflection(I, refs))
    for i in range(size // 2):
        x.append(I1)
        y.append(add_reflection(I1, refs))
    return x, y


def blur(img, k=5):
    return cv.GaussianBlur(img, (k, k), 0)


def resize(I, scale):
    return cv.resize(I, (
        np.round(I.shape[1] * scale).astype("int"), np.round(I.shape[0] * scale).astype("int")))


import cv2 as cv
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Subtract
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import load_model


class ReflectionRemovalEncoderDecoder():

    def __init__(self, input_shape, load=False, model_address="./model"):
        self.input_shape = input_shape
        if load:
            self.model = load_model(model_address)
            return
        self.model: Model = self.build_model()

    @staticmethod
    def load(model_address="./model"):
        return load_model(model_address)

    def save_model(self, model_address="./model"):
        self.model.save(model_address)

    def evaluate(self, files, refs):
        files = files[672:-1]
        images = [cv.resize(cv.imread(file) / 255,
                            (self.input_shape[0], self.input_shape[1])) for file
                  in files]
        x = []
        y = []
        for image in images:
            tx, ty = generate_batch(image, refs, 4)
            x.append(tx)
            y.append(ty)
        x = np.array(x)
        y = np.array(y)
        return self.model.evaluate(x, y)

    def test(self, image):
        image = cv.resize(image, (self.input_shape[0], self.input_shape[1]))
        batch = np.expand_dims(image, axis=0)
        return self.model.predict(batch)[0]

    def train(self, all_image_files, reflection_files, save_address, epochs=100,
              batch_size=32):
        print("in train")
        image_files = all_image_files[0:672]  # last 29 images for test and eval
        refs = [cv.imread(ref_file) / 255 for ref_file in reflection_files]
        self.save_model(save_address + "/model-bk")
        print(len(image_files))
        for epoch in range(epochs // 10):
            for i in range(0, len(image_files), 2):
                print(f"i is {i} at epoch {epoch}")
                I0 = cv.resize(cv.imread(image_files[i]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                I1 = cv.resize(cv.imread(image_files[i + 1]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x1, y1 = generate_batch(I0, refs, batch_size // 2)
                x2, y2 = generate_batch(I1, refs, batch_size // 2)
                x = np.array(x1 + x2, "float32").reshape((-1,
                                                          self.input_shape[1],
                                                          self.input_shape[0],
                                                          self.input_shape[2]))
                y = np.array(y1 + y2, "float32").reshape((-1,
                                                          self.input_shape[1],
                                                          self.input_shape[0],
                                                          self.input_shape[2]))
                self.model.fit(x, y, epochs=10, batch_size=32)
            self.save_model(save_address + "/model" + str(epoch))
        self.save_model(save_address + "/model")
        print(self.evaluate(all_image_files, refs))

    def build_model(self):
        inputs = Input(
            (self.input_shape[1], self.input_shape[0], self.input_shape[2]))
        x = Conv2D(filters=32, kernel_size=9, padding="same", name="c11")(
            inputs)
        x = Activation("relu", name="a11")(x)
        x = Conv2D(filters=32, kernel_size=9, padding="same", name="c12")(x)
        x = Activation("relu", name="a12")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same", name="c13")(x)
        x = Activation("relu", name="a13")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same", name="c14")(x)
        x = Activation("relu", name="a14")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same", name="c15")(x)
        x = Activation("relu", name="a15")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same", name="c16")(x)
        x = Activation("relu", name="a16")(x)

        # down sampling

        l1 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd1")(x)
        l1 = Activation("relu", name="ad1")(l1)
        l1 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd2")(l1)
        l1 = Activation("relu", name="ad2")(l1)
        l2 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd3")(l1)
        l2 = Activation("relu", name="ad3")(l2)
        l2 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd4")(l2)
        l2 = Activation("relu", name="ad4")(l2)
        l3 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd5")(l2)
        l3 = Activation("relu", name="ad5")(l3)
        l3 = Conv2D(filters=32, kernel_size=5, padding="same", name="cd6")(l3)
        l3 = Activation("relu", name="ad6")(l3)
        # up sampling
        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu1")(l3)
        up = Activation("relu", name="au1")(up)
        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu2")(up)
        up = Activation("relu", name="au2")(up)

        up = Add(name="add1")([up, l2])

        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu3")(up)
        up = Activation("relu", name="au3")(up)
        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu4")(up)
        up = Activation("relu", name="au4")(up)

        up = Add(name="add2")([up, l1])

        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu5")(up)
        up = Activation("relu", name="au5")(up)
        up = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="cu6")(up)
        up = Subtract(name="sub1")([x, up])
        up = Activation("relu", name="au6")(up)

        # Transmission Layer Perfection
        out = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="ct1")(up)
        out = Activation("relu", name="at1")(out)
        out = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="ct2")(out)
        out = Activation("relu", name="at2")(out)

        out = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="ct3")(out)
        out = Activation("relu", name="at3")(out)
        out = Conv2DTranspose(filters=32, kernel_size=5, padding="same", name="ct4")(out)
        out = Activation("relu", name="at4")(out)

        out = Conv2DTranspose(filters=32, kernel_size=9, padding="same", name="ct5")(out)
        out = Activation("relu", name="at5")(out)
        out = Conv2DTranspose(filters=3, kernel_size=9, padding="same", name="ct6")(out)
        out = Activation("relu", name="at6out")(out)
        model = Model(inputs, out, name="EncoderDecoderReflectionRemover")
        model.summary()
        # update?
        model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

        return model
