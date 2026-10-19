import argparse
from datetime import datetime
import cv2
from pymongo import MongoClient
from ultralytics import YOLO

# Parse command-line arguments for video source and database management
parser = argparse.ArgumentParser(
    description="YOLO26 Real-Time Detection, Tracking and MongoDB Logging"
)
parser.add_argument(
    "--source",
    type=str,
    default="0",
    help=(
        "Camera index (e.g., 0), IP stream URL, or path to a video file"
        " (e.g., data/video.mp4)"
    ),
)
parser.add_argument(
    "--reset-db",
    action="store_true",
    help="Clear existing MongoDB collection logs before starting detection",
)
args = parser.parse_args()

# Determine source type (int for webcam, string for path/URL)
source = int(args.source) if args.source.isdigit() else args.source

# Connect to the local MongoDB instance and collection
client = MongoClient("mongodb://localhost:27017/")
db = client["yolo_mongo_db"]
collection = db["person_logs"]

# If --reset-db flag is provided, clear the collection prior to tracking
if args.reset_db:
  collection.delete_many({})
  print("[MongoDB] Cleared existing documents from 'person_logs' collection.")

# Load the YOLO26 model
model = YOLO("yolo26n.pt")

# Open the selected video source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
  print(f"Error: Could not open video source {args.source}")
  exit()

print(
    f"Starting YOLO26 detection and MongoDB logging on source"
    f" {args.source}... Press 'q' to exit."
)

# Dictionary to keep track of active tracks and calculate average confidence
active_tracks = {}

while cap.isOpened():
  success, frame = cap.read()
  if not success:
    print("End of video stream or failed to grab frame.")
    break

  # Run detection and tracking (class=0 represents persons in COCO)
  results = model.track(frame, persist=True, classes=[0])

  # Process detected boxes if tracking IDs are available
  if results[0].boxes and results[0].boxes.id is not None:
    boxes = results[0].boxes
    for box in boxes:
      if box.id is None:
        continue

      track_id = int(box.id.item())
      cls_id = int(box.cls.item())
      class_name = model.names[cls_id]
      confidence = float(box.conf.item())

      # Bounding box coordinates [xmin, ymin, xmax, ymax]
      coords = box.xyxy[0].tolist()
      xmin, ymin, xmax, ymax = coords

      # Bounding box characterization: top-right and bottom-left points
      top_right = {"x": round(xmax, 2), "y": round(ymin, 2)}
      bottom_left = {"x": round(xmin, 2), "y": round(ymax, 2)}

      # If this track ID is new, insert a new document to MongoDB
      if track_id not in active_tracks:
        entry_time = datetime.now()

        log_document = {
            "track_id": track_id,
            "bounding_box": {
                "top_right": top_right,
                "bottom_left": bottom_left,
            },
            "class_name": class_name,
            "average_confidence": round(confidence, 4),
            "entry_time": entry_time,
        }

        # Insert document and capture MongoDB's default _id
        result = collection.insert_one(log_document)
        mongo_default_id = result.inserted_id

        active_tracks[track_id] = {
            "mongo_id": mongo_default_id,
            "conf_sum": confidence,
            "conf_count": 1,
        }

        print(
            f"[MongoDB] Inserted new person - YOLO ID: {track_id}, Default"
            f" _id: {mongo_default_id}, Time: {entry_time}"
        )
      else:
        # Update existing track's confidence sum and count
        track_data = active_tracks[track_id]
        track_data["conf_sum"] += confidence
        track_data["conf_count"] += 1
        avg_confidence = track_data["conf_sum"] / track_data["conf_count"]

        # Update MongoDB document with latest bounding box and updated average confidence
        collection.update_one(
            {"_id": track_data["mongo_id"]},
            {
                "$set": {
                    "bounding_box": {
                        "top_right": top_right,
                        "bottom_left": bottom_left,
                    },
                    "average_confidence": round(avg_confidence, 4),
                }
            },
        )

  # Plot the annotated results on the frame
  annotated_frame = results[0].plot()

  # Display the frame in a window
  cv2.imshow(f"YOLO26 Detection - Source {args.source}", annotated_frame)

  # Exit loop on pressing 'q'
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release the capture device and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()