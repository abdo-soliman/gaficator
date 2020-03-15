import os
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from argparse import ArgumentParser
from deeplabv3 import DeepLabModel, label_to_color_image

TEMPLATE_IMAGE_PATH = os.path.join(os.getcwd(), "models", "template.jpg")
MODEL_PATH = os.path.join(os.getcwd(), "models", "deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz")
SEGMENT_BACKGROUND = False
GAMAL_X_START = 85
GAMAL_X_END = 255
GAMAL_Y_START = 80
GAMAL_Y_END = 250
GAMAL_SIZE = 170
PADDING_X = 0
PADDING_Y = 0


def segmentation(pil_input_img, input_img):
    segmenter = DeepLabModel(MODEL_PATH)
    _, seg_map = segmenter.run(pil_input_img)
    mask = np.array(label_to_color_image(seg_map).astype(np.uint8))
    mask = cv2.resize(mask, (pil_input_img.size[0], pil_input_img.size[1]), interpolation=cv2.INTER_AREA)

    result = input_img.copy()
    idx = (mask==0)
    result[idx] = 0
    return result


def get_faces(img):
    detector = MTCNN()
    results = detector.detect_faces(img)
    faces = []
    for box in results:
        x, y, w, h = box['box']

        start_x = x - PADDING_X
        end_x = x + w + PADDING_X

        start_y = y - PADDING_Y
        end_y = y + h + PADDING_Y

        face = img[start_y:end_y, start_x:end_x, :]
        face = cv2.resize(face, (GAMAL_SIZE, GAMAL_SIZE), interpolation=cv2.INTER_AREA)
        faces.append(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY))

    return faces

def gaficate(input_file, output_dir):
    template = cv2.imread(TEMPLATE_IMAGE_PATH)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    input_img = cv2.imread(input_file)

    if SEGMENT_BACKGROUND:
        pil_input_img = Image.open(input_file)
        segmented_image = segmentation(pil_input_img, input_img)
    else:
        segmented_image = input_img

    count = 1
    for face in get_faces(segmented_image):
        result = template.copy()
        mask = face.copy()
        mask[mask != 0] = 1
        mask[mask == 0] = 255
        mask[mask != 255] = 0
        blending_mask = result[GAMAL_Y_START:GAMAL_Y_END, GAMAL_X_START:GAMAL_X_END] * mask
        result[GAMAL_Y_START:GAMAL_Y_END, GAMAL_X_START:GAMAL_X_END] = cv2.blur(face + blending_mask, (3, 3))
        cv2.imwrite(os.path.join(output_dir, "output_" + str(count) + ".jpg"), result)
        count += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', default="input.jpg", type=str, help='input image to be gaficated')
    parser.add_argument('-o', '--output_dir', default="output", type=str, help='gafication output filename')
    parser.add_argument('-es', '--enable_segmentation', default=False, type=bool, help='enable background segmentation')
    parser.add_argument('-px', '--padding_x', default=PADDING_X, type=str, help='face width padding in pixels')
    parser.add_argument('-py', '--padding_y', default=PADDING_Y, type=str, help='face height padding in pixels')

    args = parser.parse_args()
    input_file = args.input
    output_dir = args.output_dir
    SEGMENT_BACKGROUND = args.enable_segmentation
    PADDING_X = args.padding_x
    PADDING_Y = args.padding_y

    gaficate(input_file, output_dir)
