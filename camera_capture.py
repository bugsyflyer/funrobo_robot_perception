""" 
Simple script to display and write frames from the selected camera using OpenCV.

You save frames by simply pressing "q" on your keyboard.
"""
import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv

video_id = 1 # this is setup-dependent and would need to change. Ranges from 0-10+
cap = cv2.VideoCapture(video_id) 
i = 0

#roboflow model stuff
rf = Roboflow(api_key="RmDkhaQDgIJQgd76nVTR")
project = rf.workspace().project("block-segmentation-ofboy")
model = project.version(4).model
label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator(opacity=0.3)
bounding_box_annotator = sv.BoxAnnotator()
dot_annotator = sv.DotAnnotator(radius=12)

def CA1(img):
    
    #mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # saturation
    _, otsu_mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu_mask
    
    #morphology
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #duplicate image for drawing
    img_contours = img.copy()

    #for each contour
    for cnt in contours:
        area = cv2.contourArea(cnt)

        #get rid of noise
        if area > 500:
            #bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_contours, (x,y), (x+w, y+h), (0,255,0), 2)

            #actual contours
            cv2.drawContours(img_contours, [cnt], -1, (255,0,0), 2)

    #segment
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow(mask, cmap='gray', vmin=0, vmax=255)
    cv2.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    cv2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def CA2(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    confidence_threshold = 90

    result = model.predict(img, confidence=confidence_threshold).json()
    
    predictions = result["predictions"]
    
    labels = [item["class"] for item in result["predictions"]]
    
    #red cube
    red_predictions = [pred for pred in result["predictions"] if pred["class"] == "red"]
    
    red_result = dict()
    red_result["image"] = result["image"]
    red_result["predictions"] = red_predictions
    
    detections = sv.Detections.from_inference(red_result)

    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator(opacity=0.3)
    bounding_box_annotator = sv.BoxAnnotator()
    dot_annotator =sv.DotAnnotator(radius=12)
    
    #annotate the image
    annotated_image = mask_annotator.annotate(
        scene=img_rgb, detections=detections)
    annotated_image = bounding_box_annotator.annotate(
        scene=annotated_image, detections=detections)
    annotated_image = dot_annotator.annotate(
        scene=annotated_image, detections=detections)
    
    cv2.imshow("annotated", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.show()
    

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'): # saves the current frame each time "q" is pressed
            print("Saving to frame.png")
            frame_str = './img/img'+str(i)+'.png'
            cv2.imwrite(frame_str, frame)
            #call a func
            CA2(frame)
            i+= 1

    else:
        print("Failed to capture frame")
        break

cap.release()