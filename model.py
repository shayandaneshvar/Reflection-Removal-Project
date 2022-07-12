import cv2 as cv
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Subtract
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import load_model

import util


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
            tx, ty = util.generate_batch(image, refs, 4)
            x.append(tx)
            y.append(ty)
        x = np.array(x)
        y = np.array(y)
        return self.model.evaluate(x, y)

    def test(self, image):
        image = cv.resize(image, (self.input_shape[0],self.input_shape[1]))
        batch = np.expand_dims(image, axis=0)
        return self.model.predict(batch)[0]

    def train(self, all_image_files, reflection_files, epochs=100,
              batch_size=32):
        image_files = all_image_files[0:672]  # last 29 images for test and eval
        refs = [cv.imread(ref_file) / 255 for ref_file in reflection_files]
        for epoch in range(epochs // 5):
            for i in range(0, len(image_files), 2):
                print(f"i is {i} at epoch {epoch}")
                I0 = cv.resize(cv.imread(image_files[i]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                I1 = cv.resize(cv.imread(image_files[i + 1]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x1, y1 = util.generate_batch(I0, refs, batch_size // 2)
                x2, y2 = util.generate_batch(I1, refs, batch_size // 2)
                x = np.array(x1 + x2, "float32").reshape((-1,
                                                          self.input_shape[0],
                                                          self.input_shape[1],
                                                          self.input_shape[2]))
                y = np.array(y1 + y2, "float32").reshape((-1,
                                                          self.input_shape[0],
                                                          self.input_shape[1],
                                                          self.input_shape[2]))
                self.model.fit(x, y, epochs=5,batch_size=32)
            self.save_model("./model" + str(epoch))
        self.save_model("./model")
        print(self.evaluate(all_image_files, refs))

    def build_model(self):
        inputs = Input(self.input_shape)
        x = Conv2D(filters=32, kernel_size=9, padding="same", name="c11")(
            inputs)
        x = Conv2D(filters=32, kernel_size=9, padding="same", name="c12")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same", name="c13")(x)
        x = Activation("relu", name="a11")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same",name="c14")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same",name="c15")(x)
        x = Conv2D(filters=32, kernel_size=5, padding="same",name="c16")(x)
        x = Activation("relu",name="a12")(x)

        # down sampling

        l1 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd1")(x)
        l1 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd2")(l1)
        l1 = Activation("relu",name="ad1")(l1)
        l2 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd3")(l1)
        l2 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd4")(l2)
        l2 = Activation("relu",name="ad2")(l2)
        l3 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd5")(l2)
        l3 = Conv2D(filters=32, kernel_size=5, padding="same",name="cd6")(l3)
        l3 = Activation("relu",name="ad3")(l3)
        # up sampling
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu1")(l3)
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu2")(up)
        up = Activation("relu",name="au1")(up)

        up = Add(name="add1")([up, l2])
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu3")(up)
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu4")(up)
        up = Activation("relu",name="au2")(up)

        up = Add(name="add2")([up, l1])
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu5")(up)
        up = Conv2D(filters=32, kernel_size=5, padding="same",name="cu6")(up)
        up = Subtract(name="sub1")([x, up])
        up = Activation("relu",name="au3")(up)

        # Transmission Layer Perfection
        out = Conv2D(filters=32, kernel_size=5, padding="same",name="ct1")(up)
        out = Conv2D(filters=32, kernel_size=5, padding="same",name="ct2")(out)
        out = Activation("relu",name="at1")(out)

        out = Conv2D(filters=32, kernel_size=5, padding="same",name="ct3")(out)
        out = Conv2D(filters=32, kernel_size=5, padding="same",name="ct4")(out)
        out = Activation("relu",name="at2")(out)

        out = Conv2D(filters=32, kernel_size=5, padding="same",name="ct5")(out)
        out = Conv2D(filters=3, kernel_size=5, padding="same",name="ct6")(out)
        out = Activation("relu",name="at3out")(out)
        model = Model(inputs, out, name="EncoderDecoderReflectionRemover")
        model.summary()
        # update?
        model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

        return model
