from PIL import Image
import numpy as np
from builtins import list
import cv2
import os
import cv2
import tensorflow as tf
import numpy as np
import colorsys
import random
import json

from PIL import Image
import matplotlib.pyplot as plt

### General ###
def create_folder(path):    
    '''
        Create folder for path

        INPUTS:
        path: path of the folder that should be created
    '''

    if not os.path.exists(path):
        os.makedirs(path)
        # resp_path = path
        if os.path.exists(path):
            print("Folder created. (Path: {})".format(path))
        else:
            print("Folder can not be created!!! (Path: {})".format(path))
            return None
    else:
        print("Folder already exists. (Path: {})".format(path))
        # resp_path = path

    return path

def create_json_file_from_dict(path, dictionary):
    '''
        Create json file from a dictionary

        INPUTS:
        path: path of the json file
        dictionary: dictionary that is saved in json

        OUTPUT:
        json file: file with dictionary informations in it
    '''
    try:
        with open(path, "w") as outfile:
            json.dump(dictionary, outfile, indent=4)
    except Exception as ex:
        print(ex)
        print(f"Could not create json file. (Path: {path})")
        return False
    else:
        print(f"Created json file. (Path: {path})")
        return True

def del_key_recursive(keyname: str, to_clean_dict: dict):
    '''
        Deletes key from a dictionary in all sub-levels

        INPUTS:
        keyname: name of the key which will be deleted
        to_clean_dict: dictionary where the key will be searched in

        OUTPUT:
        copy_to_clean_dict: dictionary without key inside
    '''
    copy_to_clean_dict = to_clean_dict.copy()
    for key, val in to_clean_dict.items():
        if type(val) == type({}):
            copy_to_clean_dict[key] = del_key_recursive(keyname, val)
        if key == keyname:
            del copy_to_clean_dict[keyname]
    return copy_to_clean_dict

### Tensorflow ###
def load_model(model_path):
    '''
        Load model in model_path

        INPUT:
        model_path: path of the model

        OUTPUT:
        model: model loaded in tensorflow
    '''

    print("Loading model. (Path: {})".format(model_path))
    model = tf.keras.models.load_model(model_path)
    print("Model loaded. (Path: {})".format(model_path))

    return model

def generate_predictions(model, image):
    '''
        Generate predictions for image with model
        
        model: nn-model used
        image: image used for prediction

        OUTPUT:
        predictions: predictions of the nn for the specific image
    '''

    print("Starting prediction.")
    predictions = model.predict(image)
    print("Prediction finished.")

    return predictions

def non_max_suppression(prediction, IoU_threshold=0.6, prediction_threshold=0.5):
    '''
        Using non-maximum-suppression for reducing the amount of bounding boxes

        INPUTS:
        prediction: array with all predictions
        IoU_threshold: Threshold for the intersection over union between prediction bboxes
        prediction_threshold: Threshold for the minimum reached probability of an object

        OUTPUT:
        selected_boxes: predictions that succesfully went through nms filtering
        selected_indices: indices of the selected boxes in the previous prediction array
    '''
    boxes = prediction[0,:,:4]
    scores = prediction[0,:,4:].max(axis=1)
    selected_indices = tf.image.non_max_suppression(boxes, scores, max_output_size=scores.shape[0], iou_threshold=IoU_threshold, score_threshold=prediction_threshold)
    selected_boxes = tf.gather(boxes, selected_indices)

    return selected_boxes.numpy(), selected_indices.numpy()

### Images ###
def load_image(image_path):
    '''
        Loading image from image_path

        INPUT:
        image_path: path to the image file

        OUTPUT:
        image: image data in cv2-format
    '''

    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def adapt_image_to_yolov4(image, input_w=960, input_h=608):
    '''
        Adaption of the image for yolov4 input

        INPUTS:
        image: image data in cv2-format
        input_w: width of the yolov4-nn-input layer
        input_h: height of the yolov4-nn-input layer

        OUTPUT:
        yolov4_image: resized image to yolov4-input layer
    '''



    yolov4_image = cv2.resize(image, (input_w, input_h))
    yolov4_image = [yolov4_image / 255.]
    yolov4_image = np.asarray(yolov4_image).astype(np.float32)

    return tf.constant(yolov4_image)

def draw_bbox_on_image(predictions, image, target_path, prediction_threshold=0.0):
    '''
        Generates a png-image of the inference

        INPUTS:
        predictions: array with predicted bboxes
        image: image in cv2-format
        target_path: path for saving the image
        prediction_threshold: Threshold which predictions should be drawn

        OUTPUT:
        Image file with bboxes
    '''
    result_image = np.copy(image)
    bbox_attributes = predictions.shape[-1] # number of bbox attributes in model
    
    hsv_tuples = [(1.0 * x / (bbox_attributes), 0.6, 1.) for x in range(bbox_attributes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for classIdx in range(4, bbox_attributes):
        relevant_predictions = [predictions[idx, :] for idx in range(predictions.shape[0]) if predictions[idx, classIdx] > prediction_threshold]

        for relevant_prediction in relevant_predictions:
            boxes = relevant_prediction[0:4]
            pred_conf = relevant_prediction[classIdx]

            coord=[]
            coord.append(int(boxes[0] * image.shape[0]))
            coord.append(int(boxes[1] * image.shape[1]))
            coord.append(int(boxes[2] * image.shape[0]))
            coord.append(int(boxes[3] * image.shape[1]))

            fontScale = 0.5

            bbox_color = colors[classIdx]
            bbox_thick = 4
            bbox_mess = '{}: {:.2f}'.format(classIdx, float(pred_conf))
            c1, c2 = (coord[1], coord[0]), (coord[3], coord[2])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

            #Original + Detection
            result_image = cv2.rectangle(result_image, c1, c2, (bbox_color[0], bbox_color[1], bbox_color[2]), bbox_thick)
            result_image = cv2.rectangle(result_image, c1, c3, (bbox_color[0], bbox_color[1], bbox_color[2]), -1) #filled
            result_image = cv2.putText(result_image, bbox_mess, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    save_image(result_image, target_path)

def draw_bbox_on_image_with_text(image, target_path, bbox, text):
    '''
        Generates a png-image of the inference

        INPUTS:
        image: image in cv2-format
        target_path: path for saving the image
        bbox: bbox to be drawn on image
        text: Text to be added on image

        OUTPUT:
        Image file with bbox and text
        result_image: array of the merged result
    '''
    result_image = np.copy(image)
    
    coord=[]
    coord.append(int(bbox[0] * image.shape[0]))
    coord.append(int(bbox[1] * image.shape[1]))
    coord.append(int(bbox[2] * image.shape[0]))
    coord.append(int(bbox[3] * image.shape[1]))

    fontScale = 0.5

    bbox_color = (0, 0, 255)
    bbox_thick = 2

    c1, c2 = (coord[1], coord[0]), (coord[3], coord[2])
    t_size = cv2.getTextSize(text, 0, fontScale, thickness=bbox_thick // 2)[0]
    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

    #Original + Detection
    result_image = cv2.rectangle(result_image, c1, c2, (bbox_color[0], bbox_color[1], bbox_color[2]), bbox_thick)
    result_image = cv2.rectangle(result_image, c1, c3, (bbox_color[0], bbox_color[1], bbox_color[2]), -1) #filled
    result_image = cv2.putText(result_image, text, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    save_image(result_image, target_path)

    return result_image

def add_bbox_on_image_with_text(image, bbox, text, color=(0, 0, 255)):
    '''
        Function adds bounding box to image

        INPUTS:
        image: image in cv2-format
        bbox: bbox to be drawn on image
        text: Text to be added on image
        color: color of the bbox

        OUTPUT:
        result_image: array of the merged result
    '''
    result_image = np.copy(image)
    
    coord=[]
    coord.append(int(bbox[0] * image.shape[0]))
    coord.append(int(bbox[1] * image.shape[1]))
    coord.append(int(bbox[2] * image.shape[0]))
    coord.append(int(bbox[3] * image.shape[1]))

    fontScale = 0.5

    bbox_color = color
    bbox_thick = 2

    c1, c2 = (coord[1], coord[0]), (coord[3], coord[2])
    t_size = cv2.getTextSize(text, 0, fontScale, thickness=bbox_thick // 2)[0]
    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

    #Original + Detection
    result_image = cv2.rectangle(result_image, c1, c2, (bbox_color[0], bbox_color[1], bbox_color[2]), bbox_thick)
    result_image = cv2.rectangle(result_image, c1, c3, (bbox_color[0], bbox_color[1], bbox_color[2]), -1) #filled
    result_image = cv2.putText(result_image, text, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return result_image
    
def add_single_prediction_to_image(prediction, image):
    '''
        Adds a bbox to image

        INPUT:
        prediction: Prediction of a single object (shape: 1 x n)
        image: image for detection

        OUTPUT:
        result_image: bbox and original image merged together
        target_confidence: confidenc of the prediction added
    '''

    result_image = np.copy(image)
    bbox_attributes = prediction[0:4] # number of bbox attributes in model
    class_attributes = prediction[4:]
    
    target_confidence = 0
    target_index = 0
    for index, class_confidence in enumerate(class_attributes):
        if class_confidence > target_confidence:
            target_confidence = class_confidence
            target_index = index + 4

    coord=[]
    coord.append(int(bbox_attributes[0] * image.size[1]))
    coord.append(int(bbox_attributes[1] * image.size[0]))
    coord.append(int(bbox_attributes[2] * image.size[1]))
    coord.append(int(bbox_attributes[3] * image.size[0]))

    fontScale = 0.5

    bbox_color = [255, 255, 255]
    bbox_thick = 4
    bbox_mess = '{}: {:.2f}'.format(target_index, float(target_confidence))
    c1, c2 = (coord[1], coord[0]), (coord[3], coord[2])
    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)

    #Original + Detection
    result_image = cv2.rectangle(result_image, c1, c2, (bbox_color[0], bbox_color[1], bbox_color[2]), bbox_thick)
    result_image = cv2.rectangle(result_image, c1, c3, (bbox_color[0], bbox_color[1], bbox_color[2]), -1) #filled
    result_image = cv2.putText(result_image, bbox_mess, (c1[0], (c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return result_image, target_confidence

def save_image(image, target_path):
    '''
        Save image to target_path

        INPUTS:
        image: image in cv2-format
        target_path: path for saving the image

        OUTPUT:
        png file of the image 
    '''

    try:
        # print("Saving image. (Path: {})".format(target_path))
        cv2.imwrite(target_path + ".png", image)
        np.save(target_path, image)
        print(f"Image saved. (Path: {target_path})")
    except Exception as ex:
        print(f"Image could not be saved. (Path: {target_path})")
        print(ex)

def create_compare_image(images, scale):
    '''
        Create an image merged from different images

        INPUT:
        images: array of images
        scale: scale for the sizing of the images

        OUTPUT:
        final_image_data: array of the merged result
    '''

    max_width = 0
    max_height = [0]
    for img in images:
        max_height.append(img.shape[0])
        max_width += img.shape[1]
    h = np.max(max_height)
    w = max_width
    final_image = np.zeros((h, w, 3), dtype=np.uint8)

    current_x = 0 # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[:image.shape[0],current_x:image.shape[1]+current_x,:] = image
        current_x += image.shape[1]

    final_image_data = cv2.resize(final_image, (int(final_image.shape[1]/(scale*len(images))), int(final_image.shape[0]/(scale*len(images)))))

    return final_image_data

### Plotting ###
def save_figure(fig, target_path):
    '''
        Save figure to path

        INPUT:
        fig: matplotlib figure to be saved
        target_path: path for saving the figure

        OUTPUT:
        as png file saved figure
    '''

    fig.savefig(target_path)
    print("Figure saved. (Path: {})".format(target_path))

def plot_image(image, ax, title="", aspect="equal"):
    '''
        Configures a plot for an image
        
        INPUTS:
        image: image to be plotted
        ax: axis of the subplot the image will be plotted in
        title: title of the subplot
        aspect: sizing of the subplot
    '''
    
    ax.set_title(title)
    ax.matshow(image)
    ax.set_aspect(aspect)

    # return ax

def plot_graph(xdata, ydata, ax, grid=True, title="", xlabel="", ylabel="", xlim0=0, xlim1=1, ylim0=0, ylim1=1.05, aspect="equal", marker="", color='tab:blue', linestyle='-'):
    '''
        Configures a plot for a graph

        INPUTS:
        xdata: array with all x-values
        ydata: array with all y-values
        ax: axis of the subplot the graph will be plotted in
        grid: bool if grid should be activated
        title: title of the subplot
        xlabel: label for the x-axis of the subplot
        ylabel: label for the y-axis of the subplot
        xlim0: minimum of the x-axis
        xlim1: maximimum of the x-axis
        ylim0: minimum of the y-axis
        ylim1: maximimum of the y-axis
        aspect: sizing of the subplot
        marker: marker style used for the plotted data
        color: color of marker and line of the plotted data
        linestyle: style of the line of the plotted data
    '''

    if grid:
        ax.grid()
    ax.set_title(title)
    ax.set_xlim(xlim0, xlim1)
    ax.set_ylim(ylim0, ylim1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(xdata, ydata, marker=marker, color=color, linestyle=linestyle)
    ax.set_aspect(aspect)

    # return ax

def plot_text(text, ax, xpos=0.5, ypos=0.5, ha="central", va="central", fontsize=16, color='k'):
    '''
        adds text to a plot

        INPUTS:
        text: text to be added
        ax: axis of the subplot the text will be plotted in
        xpos: x-position of the text in the subplot
        ypos: y-position of the text in the subplot
        ha: horizontal alignment
        va: vertical alignment
        fontsize: size of the font of the plotted text
        color: color of the plotted text
    '''
    ax.text(xpos, ypos, text, ha=ha, va=va, fontsize=fontsize, color=color)
    # return ax

def plot_vertical_line(ax, xpos=0.5, ymin=0, ymax=1, color='r', linestyle='--'):
    '''
        adds vertical line to a plot

        INPUTS:
        ax: axis of the subplot the text will be plotted in
        xpos: x-position of the vertical line in the subplot
        y_min: start of the vertical line
        y_max: end of the vertical line
        color: color of the plotted line
        linestyle: style of the plotted line
    '''
    ax.axvline(x=xpos, ymin=ymin, ymax=ymax, color=color, linestyle=linestyle)
    
    # return ax

def plot_line(xdata, ydata, ax, marker="", color='r', linestyle='--'):
    '''
        adds line to a plot

        INPUTS:
        xdata: array with all x-values
        ydata: array with all y-values
        ax: axis of the subplot the graph will be plotted in
        marker: marker style used for the plotted data
        color: color of marker and line of the plotted data
        linestyle: style of the line of the plotted data
    '''
    ax.plot(xdata, ydata, marker=marker, color=color, linestyle=linestyle)

    # return ax
    

def visualize_boxes_and_labels_on_image(image:Image.Image , Image_Info:list , color = (0, 255, 0))->Image.Image:
    """
    class method that draw bounding boxes and info on image

    Input:
        image: Image object
        Image_Info : dict containg the info bounding boxes
        color :   RGB
        text_color : RGB

    Output:
        image: output Image object

    """

    open_cv_image = np.array(image)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for ele in Image_Info:
        [x1, y1, x2, y2]  = ele.get('box')
        className = ele.get('class').lower()

        if className == "car":
            color = (0, 255, 0)
        elif className == "pedestrian":
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        confidence = ele.get('confidence')
        if confidence is not None:
            className = className + "%" + str(int(float(confidence) * 100))

        cv2.rectangle(open_cv_image, (int(x1),int( y1)), (int(x2), int(y2)), color, 3)
        font_scale = 0.8
        font_thickness = 1
        cv2.putText(open_cv_image, className, (int(x1), int(y1) - 5), font, font_scale, color, font_thickness, cv2.LINE_AA)


    image = Image.fromarray(open_cv_image)

    return image