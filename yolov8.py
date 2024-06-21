import ultralytics
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

# Function to downsample video
def downsample_video(input_path, output_path, scale_factor):
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Get the video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_width / scale_factor), int(frame_height / scale_factor)))

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Downsample the frame
            downsized_frame = cv2.resize(frame, (int(frame_width / scale_factor), int(frame_height / scale_factor)))
            # Write the downsized frame to the output video file
            out.write(downsized_frame)
        else:
            break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Input and output file paths
input_video_path = "C:/Users/Pratiksha/Downloads/Cam_2_Cap_870_compressed.mp4"
output_video_path = 'D:/Results/downsampled_video.mp4'

# Define the downsampling factor
downsampling_factor = 2  # Adjust this according to your needs

# Call the function to downsample the video
downsample_video(input_video_path, output_video_path, downsampling_factor)

print("Video downsampling complete.")

# Load YOLO model
model = YOLO('"D:/New_day_dataset_toll_plaza_vehicle_detection/vehicle_detection_weights/content/runs/detect/train/weights/best.pt"')

# Define callback function to be used in video processing
def callback(frame: np.ndarray, index: int) -> np.ndarray:
    # Model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # Annotate the frame with all detected classes
    labels = [f"{model.model.names[class_id]} {confidence:0.2f}"
              for confidence, class_id in zip(detections.confidence, detections.class_id)]
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
    return annotated_frame

# Process the whole video
SOURCE_VIDEO_PATH = "D:/Results/downsampled_video.mp4"
TARGET_VIDEO_PATH = "D:/Results/Cam_2_Cap_870_compressed_result.mp4"

box_annotator = sv.BoxAnnotator(thickness=3, text_thickness=3, text_scale=2)

sv.process_video(
    source_path=SOURCE_VIDEO_PATH,
    target_path=TARGET_VIDEO_PATH,
    callback=callback
)

print("Video processing complete.")
