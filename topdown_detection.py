import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization
)
from tensorflow.keras.regularizers import l2
import wget

from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks

def load_darknet_weights(model, weights_file):

    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    layers = ['yolo_darknet',
            'yolo_conv_0',
            'yolo_output_0',
            'yolo_conv_1',
            'yolo_output_1',
            'yolo_conv_2',
            'yolo_output_2']

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
          
            
            if not layer.name.startswith('conv2d'):
                continue
                
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
            
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
    
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])

    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        
        # Check if the detected class is a person (class ID = 0)
        if int(classes[i]) == 0:
            # Calculate the center and radius for the circle
            center_x = (x1y1[0] + x2y2[0]) // 2
            center_y = (x1y1[1] + x2y2[1]) // 2
            radius = min((x2y2[0] - x1y1[0]) // 2, (x2y2[1] - x1y1[1]) // 2)

            # Draw the circle for person detection
            img = cv2.circle(img, (center_x, center_y), radius, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                (x1y1[0], x1y1[1] - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        else:
            # For other classes, continue using the rectangle
            img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
            img = cv2.putText(img, '{} {:.4f}'.format(
                class_names[int(classes[i])], objectness[i]),
                x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    
    return img

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


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
    
def DarknetConv(x, filters, kernel_size, strides=1, batch_norm=True):
    # Image padding
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  
        padding = 'valid'
        
    # Defining the Conv layer
    x = Conv2D(filters=filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    
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

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  
    x = x_36 = DarknetBlock(x, 256, 8)  
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

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
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                            anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def yolo_boxes(pred, anchors, classes):

    grid_size = tf.shape(pred)[1]

    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1) 

    grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box

def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
        scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=0.5,
        score_threshold=0.6
    )

    return boxes, scores, classes, valid_detections


def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80):
  
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')

def weights_download(out='models/yolov3.weights'):
    _ = wget.download('https://pjreddie.com/media/files/yolov3.weights', out='models/yolov3.weights')
    
yolo = YoloV3()
load_darknet_weights(yolo,'models/yolov3.weights') 

cap = cv2.VideoCapture(0)


while(True):
    ret, image = cap.read()
    if ret == False:
        break
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    img = img / 255
    class_names = [c.strip() for c in open("models/classes.TXT").readlines()]
    boxes, scores, classes, nums = yolo(img)
    count=0

    for i in range(nums[0]):
        if int(classes[0][i]) == 0:   
           count += 1
        if int(classes[0][i]) == 67:  
           print('Cell Phone detected')
        if int(classes[0][i]) == 70:  
           print('Calculator detected')
        if int(classes[0][i]) == 62:   
           print('Book detected')

        if count == 0:
           print('No person\'s head detected')
        elif count > 1: 
           print('More than one person\'s heads are detected')

        image = draw_outputs(image, (boxes, scores, classes, nums), class_names)
        cv2.imshow('Detection', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

# while True:
#     ret, image = cap.read()
#     if not ret:
#         break
    
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (320, 320))
#     img = img.astype(np.float32)
#     img = np.expand_dims(img, 0)
#     img = img / 255

#     # Load class names
#     class_names = [c.strip() for c in open("models/classes.TXT").readlines()]

#     # Get detections
#     boxes, scores, classes, nums = yolo(img)

#     count = 0  # Initialize count for detected persons

#     for i in range(nums[0]):
#         if int(classes[0][i]) == 0:   # Assuming 0 is the class index for persons
#             count += 1
#         if int(classes[0][i]) == 67:  
#             print('Cell Phone detected')
#         if int(classes[0][i]) == 70:  
#             print('Calculator detected')
#         if int(classes[0][i]) == 62:   
#             print('Book detected')

#     # Print detection summary
#     if count == 0:
#         print("No person's head detected")
#     elif count == 1:
#         print('1 person\'s head detected')
#     else: 
#         print(f'{count} persons\' heads detected')

#     # Draw outputs on the image
#     image = draw_outputs(image, (boxes, scores, classes, nums), class_names)

#     # Display the count of detected persons on the image
#     cv2.putText(image, f'Detected Heads: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Show the image with detections
#     cv2.imshow('Detection', image)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
        
cap.release()
cv2.destroyAllWindows()



#     # for i in range(nums[0]):
#     #     if int(classes[0][i] == 0):
#     #         count +=1
#     #     if int(classes[0][i] == 67):
#     #         print('Mobile Phone detected')
#     # if count == 0:
#     #     print('No person head detected')
#     # elif count > 1: 
#     #     print('More than one persons heads are detected')
        
#     # image = draw_outputs(image, (boxes, scores, classes, nums), class_names)

#     # cv2.imshow('Prediction', image)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     break