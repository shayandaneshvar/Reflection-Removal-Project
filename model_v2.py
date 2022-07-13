from tensorflow.keras.layers import Conv2DTranspose, MaxPool2D, Concatenate, BatchNormalization, Conv2D

import util
import cv2 as cv
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Activation
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
              batch_size=64):
        image_files = all_image_files[0:672]  # last 29 images for test and eval
        refs = [cv.imread(ref_file) / 255 for ref_file in reflection_files]
        for epoch in range(0, epochs // 10):
            j = 0
            # if epoch == 2 and j == 0:
            #     j = 200

            for i in range(0, len(image_files), 4):
                print(f"i is {i} at epoch {epoch}")
                I0 = cv.resize(cv.imread(image_files[i]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                I1 = cv.resize(cv.imread(image_files[i + 1]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x1, y1 = util.generate_batch(I0, refs, batch_size // 4)
                x2, y2 = util.generate_batch(I1, refs, batch_size // 4)
                I2 = cv.resize(cv.imread(image_files[i + 2]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x3, y3 = util.generate_batch(I2, refs, batch_size // 4)
                I3 = cv.resize(cv.imread(image_files[i + 2]),
                               (self.input_shape[0], self.input_shape[1])) / 255
                x4, y4 = util.generate_batch(I3, refs, batch_size // 4)

                x = np.array(x1 + x2 + x3 + x4, "float32").reshape(
                    (-1,
                     self.input_shape[1],
                     self.input_shape[0],
                     self.input_shape[2]))
                y = np.array(y1 + y2 + y3 + y4, "float32").reshape(
                    (-1,
                     self.input_shape[1],
                     self.input_shape[0],
                     self.input_shape[2]))
                self.model.fit(x, y, epochs=20, batch_size=32)
                if i != 0 and i % 100 == 0:
                    self.save_model(save_address + f"/temp{epoch}-{i}")

            self.save_model(save_address + "/model" + str(epoch))
        self.save_model(save_address + "/model")
        # print(self.evaluate(all_image_files, refs))

    def conv_block(self, inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x


    def encoder_block(self, input, num_filters):
        x = self.conv_block(input, num_filters)
        p = MaxPool2D((2, 2))(x)
        return x, p

    def decoder_block(self, input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self):
        inputs = Input(
            (self.input_shape[1], self.input_shape[0], self.input_shape[2]))
        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        b1 = self.conv_block(p4, 1024)

        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        out = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs, out, name="EncoderDecoderReflectionRemover")
        model.summary()
        # update?
        model.compile(optimizer='adam', loss="mse", metrics=['accuracy'])

        return model
