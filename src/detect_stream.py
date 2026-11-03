import argparse
from datetime import datetime
import cv2
from pymongo import MongoClient
from ultralytics import YOLO


def run_detection(source_input, reset_db=False, mongo_uri="mongodb://localhost:27017/"):
    """
    Handles real-time person detection, tracking via YOLOv10,
    and logs bounding boxes & confidence stats to MongoDB.
    """
    # 1. Connect to MongoDB and set up collection
    client = MongoClient(mongo_uri)
    db = client["yolo_mongo_db"]
    collection = db["person_logs"]

    # Clear existing documents if the reset flag is passed
    if reset_db:
        collection.delete_many({})
        print("[MongoDB] Cleared existing documents from 'person_logs' collection.")

    # 2. Load pretrained YOLOv10 model
    model = YOLO("yolov10n.pt")

    # 3. Resolve video source (Camera index, local video file path, or IP Stream URL)
    source = int(source_input) if str(source_input).isdigit() else source_input
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source_input}")
        return

    print(f"[Streaming] Started YOLOv10 tracking on source: {source_input}. Press 'q' to exit.")

    # Dictionary storing track-level aggregated stats (running confidence sum & count)
    active_tracks = {}

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("[Streaming] End of stream or failed to grab frame.")
            break

        # 4. Run detection and object tracking (classes=[0] filters persons only)
        results = model.track(frame, persist=True, classes=[0])

        # 5. Process detected bounding boxes when tracking IDs are present
        if results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.item())
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                confidence = float(box.conf.item())

                # Extract coordinates [xmin, ymin, xmax, ymax]
                coords = box.xyxy[0].tolist()
                xmin, ymin, xmax, ymax = coords

                # Define requested corner points: Top-Right and Bottom-Left
                top_right = {"x": round(xmax, 2), "y": round(ymin, 2)}
                bottom_left = {"x": round(xmin, 2), "y": round(ymax, 2)}

                # 6. Insert new document into MongoDB if object is observed for the first time
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

                    result = collection.insert_one(log_document)
                    mongo_id = result.inserted_id

                    active_tracks[track_id] = {
                        "mongo_id": mongo_id,
                        "conf_sum": confidence,
                        "conf_count": 1,
                    }

                    print(f"[MongoDB] Inserted track_id={track_id} (Mongo _id: {mongo_id}) at {entry_time}")
                else:
                    # 7. Update existing track's bounding box and calculate running average confidence
                    track_data = active_tracks[track_id]
                    track_data["conf_sum"] += confidence
                    track_data["conf_count"] += 1
                    avg_confidence = track_data["conf_sum"] / track_data["conf_count"]

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

        # 8. Render visualization overlay on video frame
        annotated_frame = results[0].plot()
        cv2.imshow(f"YOLOv10 Person Tracking - Source: {source_input}", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step A, B, C: YOLOv10 Person Detection & Tracking")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0), IP URL, or video path")
    parser.add_argument("--reset-db", action="store_true", help="Reset 'person_logs' collection before running")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB URI")

    args = parser.parse_args()
    run_detection(source_input=args.source, reset_db=args.reset_db, mongo_uri=args.mongo_uri)