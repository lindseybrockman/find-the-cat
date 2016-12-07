import random
from StringIO import StringIO

import cv2
from flask import (
    Flask,
    render_template,
    request
)
import numpy
from PIL import Image
import requests


CAT_CASCADE = cv2.CascadeClassifier('/Users/lindseybrockman/git-repos/find-the-cat/haarcascade_frontalcatface.xml')
CAT_CASCADE_EXTENDED = cv2.CascadeClassifier('~/git-repos/find-the-cat/haarcascade_frontalcatface_extended.xml')
HUMAN_FACE_CASCADE = cv2.CascadeClassifier('/Users/lindseybrockman/git-repos/find-the-cat/haarcascade_frontalface_default.xml')
MAX_IMAGE_SIZE = 700.0

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    params = request.params
    if params:
        message, image = find_the_cat(params)
        return render_template(
            'index.html',
            message=message,
            image=image
        )
    else:
        return render_template('index.html')


def get_image_from_url(url):
    response = requests.get(url)
    return Image.open(StringIO(response.content))


def resize_image(image):
    if image.height > MAX_IMAGE_SIZE or image.width > MAX_IMAGE_SIZE:
        h = MAX_IMAGE_SIZE / image.height
        w = MAX_IMAGE_SIZE / image.width
        resize_factor = min(h, w)

        new_width = int(image.width * resize_factor)
        new_height = int(image.height * resize_factor)
        return image.resize((new_width, new_height))

    return image


def find_the_cat(image):
    image_array = numpy.array(image)
    cats = run_detection(image_array)
    if not len(cats):
        cats = run_detection(image, cascade=CAT_CASCADE_EXTENDED)

    if len(cats):
        message = 'Found cats! Yay!'
        for (x, y, w, h) in cats:
            cv2.rectangle(
                image_array,
                (x, y),
                (x + w, y + h),
                (0, 0, 255),
                2
            )
            cv2.putText(
                image_array,
                get_random_text(),
                (x, y - 10),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.55,
                (0, 0, 255),
                2
            )

    else:
        message = 'Aw, didn\'t find any cats :('

    return message, image_array


def run_detection(image, cascade=CAT_CASCADE):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cats = cascade.detectMultiScale(
        grayscale_image,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(2, 2),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    return cats


def get_random_text():
    choices = (
        "aaaw",
        "adorbs",
        "kitty!",
        "omg",
        "ily",
        "qt pie",
        "cat!",
        "lil bb",
        "so cute!",
        "I can't even",
        "so precious"
    )

    return random.choice(choices)


if __name__ == '__main__':
    image_url = 'http://www.wildcatconservation.org/wp-content/uploads/2013/03/2016-sand-cat-group.jpg'
    image = get_image_from_url(image_url)
    image = resize_image(image)
    message, image = find_the_cat(image)
    cv2.imshow("Cats!", image)
    cv2.waitKey(0)
