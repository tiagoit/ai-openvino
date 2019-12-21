import cv2
import numpy as np


def pose_estimation(input_image):
    """
    Given some input image, preprocess the image so that
    it can be used with the related pose estimation model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    """
    height = 256
    width = 456
    preprocessed_image = np.copy(input_image)
    preprocessed_image = cv2.resize(preprocessed_image, (width, height))
    preprocessed_image = preprocessed_image.transpose((2, 0, 1))
    preprocessed_image = preprocessed_image.reshape(1, 3, height, width)
    return preprocessed_image


def text_detection(input_image):
    """
    Given some input image, preprocess the image so that
    it can be used with the related text detection model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    """
    preprocessed_image = np.copy(input_image)

    return preprocessed_image


def car_meta(input_image):
    """
    Given some input image, preprocess the image so that
    it can be used with the related car metadata model
    you downloaded previously. You can use cv2.resize()
    to resize the image.
    """
    preprocessed_image = np.copy(input_image)

    return preprocessed_image
