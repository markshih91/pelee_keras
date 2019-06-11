import numpy as np
import colorsys
from models.pelee import pelee
import cv2
import time
import tensorflow as tf
from tensorflow.python.ops import gen_image_ops
tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2


image_size = (300, 300, 3)
n_classes = 80
mode = 'inference_fast'
l2_regularization = 0.0005
min_scale = 0.1
max_scale = 0.9
scales = None
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]
two_boxes_for_ar1 = True
steps = None
offsets = None
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.01
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False
CATEGORIES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    hsv = [(i / len(labels), 1, 0.7) for i in range(len(labels))]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [(color[0]*255, color[1]*255, color[2]*255) for color in colors]

    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
    """

    labels = predictions[:, 0]
    boxes = predictions[:, 2:]

    if len(labels) > 0:
        colors = compute_colors_for_labels(labels)

        for box, color in zip(boxes, colors):
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple([int(top_left[0]), int(top_left[1])]),
                                  tuple([int(bottom_right[0]), int(bottom_right[1])]), tuple(color), 1)

    return image


def overlay_mask(image, predictions, alpha=0.5):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
    """
    masks = predictions['masks']
    labels = predictions['class_ids']

    if len(labels) > 0:
        masks = masks.transpose(2, 0, 1).astype(np.uint8)
        colors = compute_colors_for_labels(labels)

        for mask, color in zip(masks, colors):
            for c in range(3):
                image[:, :, c] = np.where(mask == 1,
                                          image[:, :, c] *
                                          (1 - alpha) + alpha * color[c],
                                          image[:, :, c])

    composite = image

    return composite


def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    scores = predictions[:, 1]
    labels = predictions[:, 0]
    labels = [CATEGORIES[int(i)] for i in labels]
    boxes = predictions[:, 2:]

    template = "{}: {:.2f}"
    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        s = template.format(label, score)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return image


def run_on_opencv_image(model, image, threshold):
    """
    Arguments:
        image (np.ndarray): an image as returned by OpenCV

    Returns:
        prediction (BoxList): the detected objects. Additional information of the detection properties can be found in
        the fields of the BoxList via `prediction.fields()`
    """
    input_image = image[:, :, [2, 1, 0]]
    input_image = cv2.resize(input_image, (image_size[0], image_size[1]))
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)[0]

    predictions = predictions[predictions[:, 1] > threshold]

    if len(predictions) > 0:
        predictions[:, 2] = predictions[:, 2] * image.shape[1] / image_size[0]
        predictions[:, 3] = predictions[:, 3] * image.shape[0] / image_size[1]
        predictions[:, 4] = predictions[:, 4] * image.shape[1] / image_size[1]
        predictions[:, 5] = predictions[:, 5] * image.shape[0] / image_size[0]

    result = image.copy()
    result = overlay_boxes(result, predictions)
    result = overlay_class_names(result, predictions)

    return result


def main():

    model = pelee(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                             aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                             offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                             divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                             nms_max_output_size, return_predictor_sizes)

    # Load weights trained on MS-COCO
    model.load_weights('pretrained_weights/pelee_coco_44_loss-5.9207_val_loss-5.2225.h5', by_name=True)

    # prepare object that handles inference plus adds predictions on top of image
    cam = cv2.VideoCapture(0)
    while True:
        # start_time = time.time()

        ret_val, img = cam.read()
        composite = run_on_opencv_image(model, img, 0.5)
        cv2.imshow("COCO detections", composite)

        # t = time.time() - start_time
        # print("Time0: {:.6f} s / img, {:.2f} fps".format(t, 1 / t))

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
