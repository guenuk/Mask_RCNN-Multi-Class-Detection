# import colorsys
import cv2
import numpy as np
from numpy.lib.type_check import imag


def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    colors = random_colors(n_instances)
    height, width = image.shape[:2]
    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        mask = masks[::, i]
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )
    return image


if __name__ == '__main__':
    import os
    import sys
    import random
    import math
    import time
    import screw as screw
    from mrcnn import utils
    import mrcnn.model as modellib

    ROOT_DIR = os.path.abspath("/content/Mask_RCNN-Multi-Class-Detection")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    SCREW_WEIGHTS_PATH = "../gdrive/MyDrive/maskRCNN/model/mask_rcnn_screw_0030.h5"
    config = screw.ScrewConfig()
    SCREW_DIR = "dataset"
    print(SCREW_WEIGHTS_PATH)


    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        name = 'screw'
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1


    infer_config = InferenceConfig()
    infer_config.display()
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=infer_config)
    weights_path = "../gdrive/MyDrive/maskRCNN/model/mask_rcnn_screw_0030.h5"  # TODO: update this path
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    class_names = ['BG', 'M', 'I', 'N', 'K', 'B']
    capture = cv2.videoCapture(0)
    while True:
        ret, frame = capture.read()
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
