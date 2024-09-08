import numpy as np
import time
import cv2
import math
import imutils
import pyttsx3

# Load labels and model
labelsPath = r"C:\Users\samee\OneDrive\Desktop\Exb social distancing\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = r"C:\Users\samee\OneDrive\Desktop\Exb social distancing\yolov3.weights"
configPath = r"C:\Users\samee\OneDrive\Desktop\Exb social distancing\yolov3.cfg"

# Initialize colors and TTS engine
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
engine = pyttsx3.init()

# Load model
print("Loading Machine Learning Model ...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Start camera
print("Starting Camera ...")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, image = cap.read()
    image = imutils.resize(image, width=800)
    (H, W) = image.shape[:2]

    # Prepare input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Perform prediction
    start = time.time()
    layerOutputs = net.forward(net.getUnconnectedOutLayersNames())
    end = time.time()
    print(f"Prediction time/frame: {end - start:.6f} seconds")

    # Process detections
    boxes, confidences, classIDs = [], [], []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.1 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    color_far, color_close = (0, 255, 0), (0, 0, 255)

    # Draw bounding boxes and labels
    if len(idxs) > 0:
        idxs = idxs.flatten()
        for i in idxs:
            for j in idxs:
                if i < j:
                    x_dist, y_dist = boxes[j][0] - boxes[i][0], boxes[j][1] - boxes[i][1]
                    distance_between_objects = math.sqrt(x_dist**2 + y_dist**2)
                    color, label = (color_far, "Normal")

                    if distance_between_objects < 220:
                        color, label = (color_close, "Red Alert: MOVE AWAY")
                        engine.say("Please maintain social distancing")
                        engine.runAndWait()

                    for box in [boxes[i], boxes[j]]:
                        cv2.rectangle(image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
                        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Social Distancing Detector", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
