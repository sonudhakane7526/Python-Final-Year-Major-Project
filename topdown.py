# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras import Model
# from tensorflow.keras.layers import (
#     Add,
#     Concatenate,
#     Conv2D,
#     Input,
#     Lambda,
#     LeakyReLU,
#     UpSampling2D,
#     ZeroPadding2D,
#     BatchNormalization
# )
# from tensorflow.keras.regularizers import l2
# import wget

# # Define the YOLOv3 model functions and layers
# def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
#     if strides == 1:
#         padding = 'same'
#     else:
#         x = ZeroPadding2D(((1, 0), (1, 0)))(x)
#         padding = 'valid'
#     x = Conv2D(filters=filters, kernel_size=kernel_size,
#                strides=strides, padding=padding,
#                use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
#     if batch_norm:
#         x = BatchNormalization()(x)
#         x = LeakyReLU(alpha=0.1)(x)
#     return x

# def DarknetResidual(x, filters):
#     prev = x
#     x = DarknetConv(x, filters // 2, 1)
#     x = DarknetConv(x, filters, 3)
#     x = Add()([prev, x])
#     return x

# def DarknetBlock(x, filters, blocks):
#     x = DarknetConv(x, filters, 3, strides=2)
#     for _ in range(blocks):
#         x = DarknetResidual(x, filters)
#     return x

# def Darknet(name=None):
#     x = inputs = Input([None, None, 3])
#     x = DarknetConv(x, 32, 3)
#     x = DarknetBlock(x, 64, 1)
#     x = DarknetBlock(x, 128, 2)
#     x = x_36 = DarknetBlock(x, 256, 8)
#     x = x_61 = DarknetBlock(x, 512, 8)
#     x = DarknetBlock(x, 1024, 4)
#     return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

# def YoloConv(filters, name=None):
#     def yolo_conv(x_in):
#         if isinstance(x_in, tuple):
#             inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
#             x, x_skip = inputs
#             x = DarknetConv(x, filters, 1)
#             x = UpSampling2D(2)(x)
#             x = Concatenate()([x, x_skip])
#         else:
#             x = inputs = Input(x_in.shape[1:])
#         x = DarknetConv(x, filters, 1)
#         x = DarknetConv(x, filters * 2, 3)
#         x = DarknetConv(x, filters, 1)
#         x = DarknetConv(x, filters * 2, 3)
#         x = DarknetConv(x, filters, 1)
#         return Model(inputs, x, name=name)(x_in)
#     return yolo_conv

# def YoloOutput(filters, anchors, classes, name=None):
#     def yolo_output(x_in):
#         x = inputs = Input(x_in.shape[1:])
#         x = DarknetConv(x, filters * 2, 3)
#         x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
#         x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
#                                             anchors, classes + 5)))(x)
#         return tf.keras.Model(inputs, x, name=name)(x_in)
#     return yolo_output

# def yolo_boxes(pred, anchors, classes):
#     grid_size = tf.shape(pred)[1]
#     box_xy, box_wh, objectness, class_probs = tf.split(
#         pred, (2, 2, 1, classes), axis=-1)
#     box_xy = tf.sigmoid(box_xy)
#     objectness = tf.sigmoid(objectness)
#     class_probs = tf.sigmoid(class_probs)
#     pred_box = tf.concat((box_xy, box_wh), axis=-1)
#     grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
#     grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
#     box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
#         tf.cast(grid_size, tf.float32)
#     box_wh = tf.exp(box_wh) * anchors
#     box_x1y1 = box_xy - box_wh / 2
#     box_x2y2 = box_xy + box_wh / 2
#     bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
#     return bbox, objectness, class_probs, pred_box

# def yolo_nms(outputs, anchors, masks, classes):
#     b, c, t = [], [], []
#     for o in outputs:
#         b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
#         c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
#         t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))
#     bbox = tf.concat(b, axis=1)
#     confidence = tf.concat(c, axis=1)
#     class_probs = tf.concat(t, axis=1)
#     scores = confidence * class_probs
#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#         boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
#         scores=tf.reshape(
#         scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
#         max_output_size_per_class=50,
#         max_total_size=50,
#         iou_threshold=0.4,
#         score_threshold=0.3
#     )
#     return boxes, scores, classes, valid_detections

# def YoloV3(size=416, channels=3, anchors=yolo_anchors,
#            masks=yolo_anchor_masks, classes=80):
#     x = inputs = Input([size, size, channels], name='input')

#     x_36, x_61, x = Darknet(name='yolo_darknet')(x)

#     x = YoloConv(512, name='yolo_conv_0')(x)
#     output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

#     x = YoloConv(256, name='yolo_conv_1')((x, x_61))
#     output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

#     x = YoloConv(128, name='yolo_conv_2')((x, x_36))
#     output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

#     boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
#                      name='yolo_boxes_0')(output_0)
#     boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
#                      name='yolo_boxes_1')(output_1)
#     boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
#                      name='yolo_boxes_2')(output_2)

#     outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
#                      name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

#     return Model(inputs, outputs, name='yolov3')

# def load_darknet_weights(model, weights_file):
#     wf = open(weights_file, 'rb')
#     major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
#     layers = ['yolo_darknet', 'yolo_conv_0', 'yolo_output_0', 'yolo_conv_1', 'yolo_output_1', 'yolo_conv_2', 'yolo_output_2']
#     for layer_name in layers:
#         sub_model = model.get_layer(layer_name)
#         for i, layer in enumerate(sub_model.layers):
#             if not layer.name.startswith('conv2d'):
#                 continue
#             batch_norm = None
#             if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
#                 batch_norm = sub_model.layers[i + 1]
#             filters = layer.filters
#             size = layer.kernel_size[0]
#             in_dim = layer.input_shape[-1]
#             if batch_norm is None:
#                 conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
#             else:
#                 bn_weights = np.fromfile(wf, dtype=np.float32, count=filters * 4)
#                 conv_bias = bn_weights[3 * filters:]
#             conv_weights = np.fromfile(wf, dtype=np.float32, count=np.prod(layer.kernel_size) * in_dim * filters)
#             conv_weights = conv_weights.reshape(layer.kernel_size[0], layer.kernel_size[1], in_dim, filters)
#             conv_weights = np.transpose(conv_weights, (2, 0, 1, 3))
#             layer.set_weights([conv_weights, conv_bias])

# def weights_download(out='models/yolov3.weights'):
#     _ = wget.download('https://pjreddie.com/media/files/yolov3.weights', out=out)
    
# def draw_outputs(img, outputs, class_names):
#     boxes, scores, classes, nums = outputs
#     h, w, _ = img.shape
#     for i in range(nums[0]):
#         box = boxes[0][i]
#         class_id = int(classes[0][i])
#         score = scores[0][i]
#         x1, y1, x2, y2 = box
#         x1, x2 = int(x1 * w), int(x2 * w)
#         y1, y2 = int(y1 * h), int(y2 * h)
#         color = (0, 255, 0) # Green color
#         label = f"{class_names[class_id]}: {score:.2f}"
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#     return img

# def main():
#     weights_path = 'models/yolov3.weights'
#     classes_path = 'models/classes.txt'
    
#     # Load YOLO model
#     yolo = YoloV3()
#     load_darknet_weights(yolo, weights_path)
    
#     # Load class names
#     with open(classes_path, 'r') as file:
#         class_names = file.read().splitlines()

#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, image = cap.read()
#         if not ret:
#             break
        
#         img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (416, 416))
#         img = img.astype(np.float32) / 255.0
#         img = np.expand_dims(img, 0)
        
#         # Run model inference
#         preds = yolo(img)
        
#         # Post-process outputs
#         bbox, objectness, class_probs, pred_box = yolo_boxes(preds[0], yolo_anchors, len(class_names))
#         boxes, scores, classes, valid_detections = yolo_nms([bbox, objectness, class_probs], yolo_anchors, yolo_anchor_masks, len(class_names))
        
#         # Draw outputs
#         img_with_boxes = draw_outputs(image, (boxes, scores, classes, valid_detections), class_names)
        
#         # Display result
#         cv2.imshow('YOLOv3 Detection', img_with_boxes)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add, Concatenate, Conv2D, Input, Lambda, LeakyReLU, UpSampling2D,
    ZeroPadding2D, BatchNormalization
)
from tensorflow.keras.regularizers import l2
import wget

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
               padding=padding, use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x

def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x

def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x

# Define function for loading YOLOv3 weights
def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = ['yolo_darknet', 'yolo_conv_0', 'yolo_output_0',
              'yolo_conv_1', 'yolo_output_1', 'yolo_conv_2', 'yolo_output_2']
    
    for layer_name in layers:
        try:
            sub_model = model.get_layer(layer_name)
            # Only access layers if sub_model is a Model
            if isinstance(sub_model, tf.keras.Model):
                for layer in sub_model.layers:
                    if not layer.name.startswith('conv2d'):
                        continue
                    
                    batch_norm = None
                    if (sub_model.layers.index(layer) + 1 < len(sub_model.layers) and 
                        sub_model.layers[sub_model.layers.index(layer) + 1].name.startswith('batch_norm')):
                        batch_norm = sub_model.layers[sub_model.layers.index(layer) + 1]
                        
                    filters = layer.filters
                    size = layer.kernel_size[0]
                    in_dim = layer.input_shape[-1]

                    if batch_norm is None:
                        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                    else:
                        bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                        
                    conv_shape = (filters, in_dim, size, size)
                    conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
                    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
                    
                    if batch_norm is None:
                        layer.set_weights([conv_weights, conv_bias])
                    else:
                        layer.set_weights([conv_weights])
                        batch_norm.set_weights(bn_weights)
                    
        except Exception as e:
            print(f"Error loading weights for {layer_name}: {e}")

    wf.close()


# Draw output boxes on the image
def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img

# Load YOLOv3 model
def YoloV3(size=None, channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, classes=80):
    x = inputs = Input([size, size, channels], name='input')
    x_36, x_61, x = Darknet(name='yolo_darknet')(x)
    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)
    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)
    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)
    return Model(inputs, [output_0, output_1, output_2], name='yolov3')

def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv

def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = x_in
        x = Conv2D(filters=anchors * (5 + classes), kernel_size=1, padding='same', name=name)(x)
        x = Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], anchors, 5 + classes)))(x)
        return x

    return yolo_output

# # Initialize and load pre-trained weights
# yolo = YoloV3()
# load_darknet_weights(yolo, 'models/yolov3.weights')

# # Capture video from the camera placed above
# cap = cv2.VideoCapture(0)

# while(True):
#     ret, image = cap.read()
#     if ret == False:
#         break
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (416, 416))
#     img = img.astype(np.float32)
#     img = np.expand_dims(img, 0)
#     img = img / 255.0
#     class_names = [c.strip() for c in open("models/classes.txt").readlines()]

#     # Run object detection
#     # boxes, scores, classes, nums = yolo(img)
#     # Run object detection

#     outputs = yolo(img)

# # Initialize lists for concatenation
# boxes, scores, classes, nums = [], [], [], []

# for output in outputs:
#     box = output[..., :4]  # Bounding boxes
#     objectness = output[..., 4:5]  # Objectness score
#     class_probs = output[..., 5:]  # Class probabilities

#     boxes.append(box)
#     scores.append(objectness)
#     classes.append(class_probs)

#     # Count detections
#     nums.append(tf.reduce_sum(tf.cast(objectness > 0.5, tf.int32), axis=-1))

#     # Convert lists to tensors
#     boxes = tf.concat(boxes, axis=1)
#     scores = tf.concat(scores, axis=1)
#     classes = tf.concat(classes, axis=1)

 
#     # Person and phone detection logic
#     person_count = 0
#     phone_detected = False
#     for i in range(nums[0]):
#         if int(classes[0][i] == 0):  # Person
#             person_count += 1
#         if int(classes[0][i] == 67):  # Phone
#             phone_detected = True
#             print('Cheating detected: Phone detected')

#     if person_count == 0:
#         print('No person detected')
#     elif person_count > 1:
#         print('Cheating detected: More than one person detected')
        
#     if phone_detected:
#         print('Phone detected!')

#     # Display results
#     image = draw_outputs(image, (boxes, scores, classes, nums), class_names)
#     cv2.imshow('Prediction', image)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
