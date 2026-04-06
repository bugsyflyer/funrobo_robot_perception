""" 
Simple script to display and write frames from the selected camera using OpenCV.

You save frames by simply pressing "q" on your keyboard.
"""
import cv2
import numpy as np

video_id = 1 # this is setup-dependent and would need to change. Ranges from 0-10+
cap = cv2.VideoCapture(video_id) 
i = 0


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
            # frame_str = './img/calibration_imgs/calibration_img'+str(i)+'.png'
            cv2.imwrite(frame_str, frame)
            i+= 1

    else:
        print("Failed to capture frame")
        break

cap.release()