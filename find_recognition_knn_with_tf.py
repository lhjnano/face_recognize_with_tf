import math
from sklearn import neighbors
import cv2
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from threading import Thread
import matplotlib.pyplot as plt
import importlib.util
import numpy as np
import time 


pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter
    
TF_MODEL_DIRECTORY = 'tflite_graph'
TF_GRAPH_NAME ='face.tflite'
TF_GRAPH_PATH =  os.path.join('face_recognize_with_tf', TF_MODEL_DIRECTORY, TF_GRAPH_NAME)
FACE_THRESHOLD = 0.7
interpreter = Interpreter(model_path=TF_GRAPH_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

ZOOM = 4

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            class_sample_image = image
            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                known_face = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes, model='large')[0]
                X.append(known_face)
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance', )
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def load_model(model_path=None) :
    if model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    with open(model_path, 'rb') as f:
        knn_clf = pickle.load(f)
    return knn_clf 
    
    
def reco_faces(image, imgW, imgH) :
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    
    input_data = np.expand_dims(frame_resized, axis=0)
    
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std
    
    
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    
    locations = []
    if( len(scores) == 0 ) : 
        return []
    for i in range(len(scores)):
        if ((scores[i] > FACE_THRESHOLD) and (scores[i] <= 1.0)):
            
            top = int(max(1,(boxes[i][0] * imgH))/ZOOM)
            left  = int(max(1,(boxes[i][1] * imgW))/ZOOM)
            bottom= int(min(imgH,(boxes[i][2] * imgH))/ZOOM)
            right = int(min(imgW,(boxes[i][3] * imgW))/ZOOM)

            locations.append( (top, right, bottom, left) )
    return locations


def predict_faces(knn_clf, image, X_face_locations) :
    #print(f'Find Faces : {X_face_locations}')
    distance_threshold = 0.95
    if len(X_face_locations) == 0:
        return []
        
    timestamp = time.time()
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(image, known_face_locations=X_face_locations, model='large')
    print(f'faces_encodings : {time.time()-timestamp}s')
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction(image, predictions, is_showing=False) :

    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for name, (top, right, bottom, left) in predictions:
    
        top = top * ZOOM
        right = right * ZOOM
        bottom = bottom *ZOOM
        left = left * ZOOM 
        
        cv2.rectangle(image, (left,top), (right,bottom), (10, 255, 0), 4)
            
        # Draw label
        #object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
        label = name
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
        label_top = max(top, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(image, (left, label_top-labelSize[1]-10), (left+labelSize[0], label_top+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
        cv2.putText(image, label, (left, label_top-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        
    face_landmarks_list = face_recognition.face_landmarks(frame_rgb, model="large")

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            #print(f'point : {face_landmarks[facial_feature][0]}, {face_landmarks[facial_feature][1]}')
            if len(face_landmarks[facial_feature]) == 1 :
                cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][0], (255, 255, 255), 2)
            else :
                cv2.line(image, face_landmarks[facial_feature][0], face_landmarks[facial_feature][1], (255, 255, 255), 2)
    if is_showing :
        cv2.imshow('Object detector', image)
        
    return image


if __name__ == "__main__":
    # STEP 1: Train the KNN classifier and save it to disk
    # Once the model is trained and saved, you can skip this step next time.
    print("Training KNN classifier...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Training complete!")

    # STEP 2: Using the trained classifier, make predictions for unknown images
    fnn_clf = load_model("trained_knn_model.clf")
    CWD_PATH = os.getcwd()
    VIDEO_NAME = "test.mp4"
    VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)
    video = cv2.VideoCapture(VIDEO_PATH)
    imgW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    imgH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    
    OUTPUT_FILE_NAME = "output.mp4"
    OUTPUT_FILE_PATH = os.path.join(CWD_PATH, OUTPUT_FILE_NAME)
    output_file = cv2.VideoWriter(
            filename=OUTPUT_FILE_NAME,
            fourcc=cv2.VideoWriter_fourcc(*"x264"),
            fps=float(frames_per_second),
            frameSize=(int(imgW), int(imgH)),
            isColor=True)
    
    while(video.isOpened()) :
        
        ret, frame =  video.read()
        if not ret :
            print('Reached the end of the video!')
            break
            
        locations = reco_faces(frame, imgW, imgH)
        frame_resized = cv2.resize(frame, (0, 0), fx=float(1/ZOOM), fy=float(1/ZOOM))
        predictions = predict_faces(fnn_clf, frame_resized, locations)

        prediction_frame = show_prediction(frame, predictions)
        output_file.write(prediction_frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
    video.release()
    output_file.release()
    cv2.destroyAllWindows()


