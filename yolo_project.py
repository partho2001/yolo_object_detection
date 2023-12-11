!wget "https://pjreddie.com/media/files/yolov3.weights"
!wget "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
!wget "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

import cv2
import numpy as np
from google.colab.patches import cv2_imshow

def process_video_with_yolo(input_video_path, output_video_path):
    # Load YOLOv3 COCO model
    yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

    # Load COCO class names
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Open the video file
    video_capture = cv2.VideoCapture(input_video_path)

    # Get the video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the video codec
    output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), isColor=True)

    # Define YOLO configuration
    output_layers = yolo_net.getUnconnectedOutLayersNames()

    # Process each frame in the video
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if not ret:
            break

        # Detect objects in the frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo_net.setInput(blob)
        outs = yolo_net.forward(output_layers)

        # Initialize lists for storing detections
        class_ids = []
        confidences = []
        boxes = []

        # Extract information from detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    w = int(detection[2] * frame_width)
                    h = int(detection[3] * frame_height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Draw bounding boxes and labels on the frame
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the output frame
        cv2_imshow(frame)

        # Write the processed frame to the output video file
        output_video.write(frame)

        # Exit if the 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture and output video objects
    video_capture.release()
    output_video.release()

    # Close all windows
    cv2.destroyAllWindows()

# Example usage
input_video_path = "/content/drive/MyDrive/test_video2.mp4"
output_video_path = "/content/drive/MyDrive/output_yolo.mp4"

process_video_with_yolo(input_video_path, output_video_path)
