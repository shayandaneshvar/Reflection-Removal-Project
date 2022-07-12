from model import ReflectionRemovalEncoderDecoder


class ReflectionRemovalEncoderDecoderV2(ReflectionRemovalEncoderDecoder):

    def __init__(self, input_shape, load=False, model_address="./model"):
        super(ReflectionRemovalEncoderDecoderV2, self).__init__(input_shape,
                                                                load,
                                                                model_address)
        print("V2 Initializing")
