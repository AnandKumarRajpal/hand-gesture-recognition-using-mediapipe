
# import necessary modules
import csv
import cv2 as cv
import pandas as pd
import numpy as np
import mediapipe as mp
import tensorflow as tf

def flatten_landmark(landmark):
    """
        Returns a flatened array of all 21 points (x, y) in format [x1, y1, x2, y2, ...]
        
        Args:
            landmark: array of landmark objects
    """
    flat_list = []
    for lm in landmark:
            flat_list.extend([lm.x, lm.y])
    return flat_list

def select_mode(key, mode):
    """
        Helper to set the mode (add data to dataset / predict) and extract number for adding class to dataset

        Args:
            key: The ASCII code for the pressed key
    """
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode

def log_to_csv(number, mode, landmarks, path):
    """
        Helper to add new data to dataset

        Args:
            number: The true label for class
            mode: Currently selected mode (add/predict)
            landmarks: NormalizedLandmarkList
            path: Path to dataset file
    """
    if mode == 1 and (0 <= number <= 9):
        with open(path, 'a') as f:
            writer = csv.writer(f)
            flat_list = flatten_landmark(landmarks.landmark)
            writer.writerow([number, *flat_list])
    return
    
# path to dataset
DATASET_PATH = './keypoint_classifier/keypoint.csv'
# path to the classification model file
MODEL_PATH_TFLITE = './keypoint_classifier/model/keypoint_classifier.tflite'

# load the classifier labels
keypoint_classifier_labels_df = pd.read_csv('./keypoint_classifier/keypoint_classifier_label.csv', header=None)
keypoint_classifier_labels = dict(list(enumerate(keypoint_classifier_labels_df.iloc[:, 0])))

# load the mediapipe hands helper class
mp_hands = mp.solutions.hands
# load the mediapipe drawing helper class
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# initialize hands helper
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# load the model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH_TFLITE)
interpreter.allocate_tensors()

# Get I / O tensor
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
  
# define a video capture object
vid = cv.VideoCapture(0)

# set the default mode to prediction mode
mode = 0

# start detection  
while(True):

    # wait for key detection
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break
    # set the mode and number based on key
    number, mode = select_mode(key, mode)
      
    # Capture the video frame
    ret, frame = vid.read()

    # mirror the output
    image = cv.flip(frame, 1)

    image.flags.writeable = False

    # start points detection
    results = hands.process(image)

    image.flags.writeable = True

    # if the hands points were detected
    if results.multi_hand_landmarks is not None:
        # for each handpoint in the handpoint detection list
        for hand_landmark in results.multi_hand_landmarks:

            # draw the detected points
            mp_drawing.draw_landmarks(
                    image,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # get the flattened array of landmark points that can be passed to classification model
            lm = flatten_landmark(hand_landmark.landmark)
            
            # if prediction mode
            if (mode == 0):
                # set the input tensor
                interpreter.set_tensor(input_details[0]['index'], np.float32([lm]))

                # infer
                interpreter.invoke()
                tflite_results = interpreter.get_tensor(output_details[0]['index'])
                
                # get the label with highest probability
                hand_sign_id = np.argmax(np.squeeze(tflite_results))

                # show the predicted class on the screen
                cv.putText(image, "Prediction: " + keypoint_classifier_labels[hand_sign_id], (10, 90),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
                    cv.LINE_AA)
            else:
                # add data to dataset
                log_to_csv(number, mode, hand_landmark, DATASET_PATH)

    # if add mdoe
    if mode == 1:
        cv.putText(image, "Logging into dataset", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "Number: " + str(number), (10, 180),
                        cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3,
                        cv.LINE_AA)

    # Display the resulting frame
    cv.imshow('Hand Gesture Recognition', image)
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv.destroyAllWindows()