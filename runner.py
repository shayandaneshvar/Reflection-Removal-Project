import cv2 as cv
from glob import glob

from sklearn.model_selection import train_test_split

from model import ReflectionRemovalEncoderDecoder

if __name__ == '__main__':
    model = ReflectionRemovalEncoderDecoder((320, 240, 3), load=False,
                                            model_address="./model")
    # model.save_model("./raw_model")
    # print(model.evaluate(sorted(glob("./camvid/*.png")),
    #                      [cv.imread(f) / 255 for f in sorted(glob("./ref/*.png"))]))
    model.train(sorted(glob("./camvid/*.png")), sorted(glob("./ref/*.png")),".")
    pr = model.test(cv.imread("./test/test 0.png") / 255)
    cv.imshow("sd", pr * 255)
    cv.waitKey()
    cv.imshow("sd", pr )
    cv.waitKey()
    train_test_split()
