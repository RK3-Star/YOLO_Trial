"""
Detect people in a video stream, keep an id on each person, and store the track.

This is the detection half of the project. train_model.py is the training half.
Both use the same nano model menu and the same database, yolo_mongo_db, but
they write different collections:

  person_logs        one document per tracked person in a video
  training_results   one document per training run

A source is one of three things:
  1. A webcam index, such as 0.
  2. A network URL starting with rtsp://, rtmp://, http://, or https://.
  3. A video file, or a folder of video files.
Still images are rejected. This script is for a stream, not a single photo.

The pretrained weights know the 80 COCO classes. classes=[0] keeps only
"person", which is class 0 in that list. A car or a dog in the same frame
is ignored. The box drawn on screen is in the original video pixels, even
though the network itself resizes the frame to imgsz (640 by default).

Runtime flags, all optional:
  --device           cuda:0, 0, 1, or cpu. auto uses the GPU when it exists.
  --half             FP16 inference on an NVIDIA GPU. CPU stays FP32.
  --cudnn-benchmark  time the conv algorithms once for this imgsz.
  --tensorrt         build or reuse a local .engine and run that.
  --visualize        draw the boxes. With --tensorrt the window shows the
                     TensorRT engine output and the frames per second.
                     --no-visualize still writes person_logs, with no window.
"""

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
import time
import cv2
from pymongo import MongoClient
from ultralytics import YOLO

from inference_runtime import configure_runtime

# Detection loads a .pt file from the project root, the same folder train_model.py
# uses. A version that is not on disk is not offered, so nothing is downloaded.
PROJECT_DIR = Path(__file__).resolve().parent.parent
# TensorRT engines are built for one image size and one precision, and they
# stay in this project. The name records both, so 640 and 320 do not share a file.
ENGINES_DIR = PROJECT_DIR / "engines"

# A folder passed as --source is searched for these extensions only.
# A .jpg or .png in that folder is not a stream, so it is left out.
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def is_stream_url(source):
    """
    Determines if the source string represents a live network video stream.
    Checks for RTSP, RTMP, and HTTP(S). OpenCV can open those URLs the same
    way it opens a file. A path that merely contains "http" in the middle of
    the name is not treated as a stream; the scheme has to be at the start.
    """
    source_str = str(source).lower()
    return any(
        source_str.startswith(proto)
        for proto in ["rtsp://", "rtmp://", "http://", "https://"]
    )


def resolve_source(source_input):
    """
    Validates and resolves the input source into one of 3 supported stream types:
      1. Webcam / Local Camera Index (e.g., 0, 1)
      2. Network Video Stream URL (RTSP / RTMP / HTTP)
      3. Valid Video File path or Directory containing supported video files

    If the path is missing, or the file is a still image, the script stops
    with a message instead of letting OpenCV fail later. A folder with one
    video is used automatically. A folder with several videos prints a
    numbered menu and waits, the same pattern as the model menu.
    """
    source_str = str(source_input).strip()

    # "0" is the default camera. OpenCV wants this as an int, not the string "0".
    if source_str.isdigit():
        print(f"[Source Setup] Using Camera Device Index: {source_str}")
        return int(source_str)

    # Category 2: Live network streaming URL
    if is_stream_url(source_str):
        print(f"[Source Setup] Using Network Stream URL: {source_str}")
        return source_str

    path = Path(source_str)

    # Category 3: Local file or directory containing video files
    if path.exists():
        # Directory resolution: scan for supported video formats
        if path.is_dir():
            video_files = [
                f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
            ]

            if not video_files:
                print(f"\n[Error] Directory '{source_str}' contains no supported video files.")
                print(f"Allowed video formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
                sys.exit(1)

            # Auto-select if only one video file is found
            if len(video_files) == 1:
                selected = str(video_files[0])
                print(f"[Source Setup] Detected video file: {selected}")
                return selected

            # Interactive selection menu for directories with multiple videos
            print(f"\n--- Found Multiple Videos in '{source_str}' ---")
            for idx, f in enumerate(video_files, 1):
                print(f"  {idx}. {f.name}")
            print("---------------------------------------------")

            while True:
                choice = input(f"Select a video number (1-{len(video_files)}): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(video_files):
                    selected = str(video_files[int(choice) - 1])
                    print(f"[Source Setup] Selected video: {selected}\n")
                    return selected
                print(f"Error: Enter a valid number between 1 and {len(video_files)}.")

        # Single file resolution: verify against allowed video extensions
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                print(f"[Source Setup] Using Video File: {path}")
                return str(path)
            else:
                print(f"\n[Error] Invalid file type: '{path.name}' ({path.suffix})")
                print("Static images and non-video files are strictly disallowed.")
                print(f"Allowed video formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
                sys.exit(1)

    # Rejection fallback for non-existent files or invalid network targets
    print(f"\n[Error] Invalid or unreachable video source: '{source_input}'")
    print("Source must be one of the 3 supported types:")
    print("  1. Webcam / Camera Index (e.g., --source 0)")
    print("  2. Network Video Stream (e.g., rtsp://..., rtmp://..., http://...)")
    print(f"  3. Valid Video File / Directory ({', '.join(sorted(VIDEO_EXTENSIONS))})")
    sys.exit(1)


def _version_digits(path):
    """'yolov10n.pt' -> '10'. The size letter after the digits is ignored."""
    stem = path.stem.lower().replace("yolo", "").replace("v", "")
    digits = ""
    for char in stem:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return digits


def _local_weight_files():
    """`.pt` checkpoints that actually sit in the project root."""
    return sorted(path.resolve() for path in PROJECT_DIR.glob("*.pt") if path.is_file())


def _match_local_model(text, files):
    """Resolve a version or file name to one .pt file in the project root."""
    if text is None or not str(text).strip():
        return None
    raw = str(text).strip().strip('"')
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    if candidate.is_file() and candidate.suffix.lower() == ".pt":
        resolved = candidate.resolve()
        if resolved.parent == PROJECT_DIR:
            return resolved
    lowered = raw.lower()
    by_name = [path for path in files if lowered in {path.name.lower(), path.stem.lower()}]
    if len(by_name) == 1:
        return by_name[0]
    cleaned = lowered.replace("yolo", "").replace("v", "").replace(".pt", "")
    by_version = [path for path in files if cleaned and _version_digits(path) == cleaned]
    if len(by_version) == 1:
        return by_version[0]
    return None


def select_nano_model(input_value=None):
    """
    Ask which local checkpoint to run.

    The menu is the .pt files in the project root. The returned path is
    absolute, so the detector opens that file instead of downloading weights.
    """
    files = _local_weight_files()
    if not files:
        raise SystemExit(f"No .pt weights found in {PROJECT_DIR}")

    matched = _match_local_model(input_value, files)
    if matched:
        print(f"[Model Setup] Using local weights -> {matched}")
        return str(matched)
    if input_value:
        print(f"\n[Warning] '{input_value}' is not a model file in {PROJECT_DIR}.")

    print("\n--- Models in this project ---")
    for index, path in enumerate(files, 1):
        print(f"  {index}. {path.name}  (version {_version_digits(path)})")
    print("------------------------------")

    while True:
        choice = input(f"Select a model (1-{len(files)}, a version, or a file name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            selected = files[int(choice) - 1]
            print(f"[Model Selected] {selected}\n")
            return str(selected)
        matched = _match_local_model(choice, files)
        if matched:
            print(f"[Model Selected] {matched}\n")
            return str(matched)
        print(f"Error: '{choice}' is not in this folder. Enter a number from 1 to {len(files)}.")


def _align_imgsz(imgsz):
    """TensorRT needs a side length that is a multiple of the YOLO stride, 32."""
    aligned = max(32, int(round(int(imgsz) / 32.0) * 32))
    if aligned != int(imgsz):
        print(f"[TensorRT] imgsz {imgsz} adjusted to {aligned} so it matches the model stride of 32.")
    return aligned


def _export_tensorrt(weights, imgsz, device, half):
    """
    Load a local TensorRT engine, building it from the .pt file when needed.

    The engine is serialized GPU code. It cannot be built or executed on CPU.
    A later run with the same weights, imgsz, and precision reuses the file
    under engines/ instead of exporting again. The first export is slow.
    """
    if str(device).lower() == "cpu":
        raise SystemExit(
            "[TensorRT] An NVIDIA GPU is required to export and run a .engine file. "
            "Leave --tensorrt off to run the .pt model on cpu."
        )

    precision = "fp16" if half else "fp32"
    destination = ENGINES_DIR / f"{Path(weights).stem}_imgsz{imgsz}_{precision}.engine"
    if destination.is_file():
        print(f"[TensorRT] Reusing local engine: {destination}")
        return destination

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[TensorRT] Exporting {Path(weights).name} -> {destination.name} "
        f"(imgsz={imgsz}, {precision}). The first export can take several minutes."
    )
    onnx_path = Path(weights).with_suffix(".onnx")
    onnx_existed = onnx_path.is_file()
    model = YOLO(str(weights))
    try:
        exported = Path(model.export(format="engine", imgsz=imgsz, device=device, half=half))
    except Exception as exc:
        raise SystemExit(
            "[TensorRT] Export failed. TensorRT has to be installed for this Python, "
            f"and the GPU has to support it.\n{exc}"
        ) from exc

    exported = exported.resolve()
    if exported != destination.resolve():
        shutil.move(str(exported), destination)
    if onnx_path.is_file() and not onnx_existed:
        onnx_path.unlink()
    print(f"[TensorRT] Saved engine: {destination}")
    return destination


def _sync_box_to_db(box, track_id, model, imgsz, active_tracks, collection, model_name):
    """
    Synchronizes bounding box data and tracking metrics with MongoDB.
    The tracker assigns an integer id that stays with one person across frames.
    The first time that id appears, this inserts a document. Every later frame
    updates the same document. active_tracks remembers the Mongo _id in memory
    so the update does not have to search the collection by track_id.

    Stored fields, and what they mean:
      track_id            the tracker's id for this person in this run
      model_used          which nano weights produced the box
      inference_imgsz     the square size the frame was resized to
      class_name          always "person" while classes=[0] is set
      bounding_box        top-right and bottom-left corners, in video pixels
      centroid            center of the box, useful for following motion
      average_confidence  running mean of the per-frame confidence, 0 to 1
      entry_time          when this id was first seen
      last_seen           the most recent frame that still had this id
      duration_seconds    last_seen minus entry_time
      total_frames        how many frames this id has been written for
    """
    # cls is the class index, conf is how sure the model is about this box.
    cls_id = int(box.cls.item())
    class_name = model.names[cls_id]
    confidence = float(box.conf.item())

    # xyxy is the box in the original frame: left, top, right, bottom.
    # Those numbers are pixels, not the 0-1 fractions used in training labels.
    coords = box.xyxy[0].tolist()
    xmin, ymin, xmax, ymax = coords

    # Two opposite corners are enough to redraw the box. The centroid is the
    # middle of the box, which is the point to track if you care where the
    # person is rather than how wide the box is.
    top_right = {"x": round(xmax, 2), "y": round(ymin, 2)}
    bottom_left = {"x": round(xmin, 2), "y": round(ymax, 2)}
    center_x = round((xmin + xmax) / 2, 2)
    center_y = round((ymin + ymax) / 2, 2)
    now = datetime.now()

    # First frame of this id: insert the whole document.
    if track_id not in active_tracks:
        log_document = {
            "track_id": track_id,
            "model_used": model_name,
            "inference_imgsz": imgsz,
            "class_name": class_name,
            "bounding_box": {"top_right": top_right, "bottom_left": bottom_left},
            "centroid": {"x": center_x, "y": center_y},
            "average_confidence": round(confidence, 4),
            "entry_time": now,
            "last_seen": now,
            "duration_seconds": 0.0,
            "total_frames": 1,
        }
        result = collection.insert_one(log_document)

        # conf_sum / conf_count is the running average. Keeping the sum avoids
        # rereading every previous confidence from Mongo.
        active_tracks[track_id] = {
            "mongo_id": result.inserted_id,
            "conf_sum": confidence,
            "conf_count": 1,
            "entry_time": now,
        }
        print(
            f"[MongoDB] Inserted track_id={track_id} (Model: {model_name}, imgsz: {imgsz}) at {now.strftime('%H:%M:%S')}"
        )

    # Later frame of the same id: move the box, extend the duration, add a frame.
    else:
        track_data = active_tracks[track_id]
        track_data["conf_sum"] += confidence
        track_data["conf_count"] += 1

        # Average confidence is the mean of every frame so far, not just this one.
        # Duration is wall-clock time since the person entered, in seconds.
        avg_conf = track_data["conf_sum"] / track_data["conf_count"]
        duration = round((now - track_data["entry_time"]).total_seconds(), 2)

        # $set replaces those fields. $inc adds 1 to total_frames inside Mongo,
        # so two updates cannot lose a count by writing a stale number.
        collection.update_one(
            {"_id": track_data["mongo_id"]},
            {
                "$set": {
                    "bounding_box": {"top_right": top_right, "bottom_left": bottom_left},
                    "centroid": {"x": center_x, "y": center_y},
                    "average_confidence": round(avg_conf, 4),
                    "last_seen": now,
                    "duration_seconds": duration,
                },
                "$inc": {"total_frames": 1},
            },
        )


def run_detection(
    source_input,
    model_name,
    imgsz_input="640",
    reset_db=False,
    clear_on_exit=False,
    mongo_uri="mongodb://localhost:27017/",
    device="auto",
    half=False,
    cudnn_benchmark=False,
    tensorrt=False,
    visualize=True,
):
    """
    Open the video, run person tracking, draw the boxes, and keep Mongo in sync.

    reset_db empties person_logs before the first frame, which is useful when
    a previous run should not mix with this one. clear_on_exit empties it
    again when the window closes. Neither flag touches training_results.

    device, half, and cudnn_benchmark are resolved before the first frame.
    tensorrt replaces the .pt file with a local .engine for this imgsz.
    visualize draws that engine's boxes when tensorrt is on, and the PyTorch
    boxes otherwise. --no-visualize keeps the Mongo writes and skips the window.
    """
    resolved_src = resolve_source(source_input)
    runtime_device, runtime_half, _cudnn_on = configure_runtime(device, half, cudnn_benchmark)

    client = MongoClient(mongo_uri)
    db = client["yolo_mongo_db"]
    collection = db["person_logs"]

    # Optional: start from an empty person_logs collection.
    if reset_db:
        result = collection.delete_many({})
        print(f"[MongoDB Startup] Cleared {result.deleted_count} existing documents from 'person_logs'.")

    # Maps a tracker id to the Mongo _id and the confidence totals for that person.
    active_tracks = {}

    # VideoCapture reads files, webcams, and network URLs through OpenCV.
    cap = cv2.VideoCapture(resolved_src)

    if not cap.isOpened():
        print(f"\n[Error] Could not open video stream source: {resolved_src}")
        sys.exit(1)

    # The header size is the real frame. 640x480 is only a fallback when the
    # camera has not reported a size yet.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    # YOLO always resizes the frame to a square before predicting. 640 is the
    # size these nano models were trained at, so it is the default. "native"
    # uses the longer side of this video instead, which keeps more detail on
    # a large file and costs more time. The drawn boxes are scaled back to
    # the original frame either way.
    str_imgsz = str(imgsz_input).strip().lower()
    imgsz = max(width, height) if str_imgsz in ["native", "0", "orig", "none"] else int(imgsz_input)

    # The engine is compiled for one square size, so native video sizes are
    # rounded to the stride before export. Inference then uses that same size.
    if tensorrt:
        imgsz = _align_imgsz(imgsz)
        try:
            model_name = str(_export_tensorrt(model_name, imgsz, runtime_device, runtime_half))
        except SystemExit:
            cap.release()
            raise

    # These are the pretrained COCO weights, not the fine-tuned hardhat run,
    # unless --tensorrt replaced them with an engine built from those weights.
    print(f"[YOLO] Initializing model weights: {model_name}")
    print(
        f"[YOLO] device={runtime_device} half={runtime_half} "
        f"tensorrt={tensorrt} visualize={visualize}"
    )
    model = YOLO(model_name)

    # Initialize resizable OpenCV display window
    runtime_label = "TensorRT" if tensorrt else "PyTorch"
    window_name = (
        f"YOLO Person Tracking [{runtime_label}] ({Path(model_name).name} @ imgsz={imgsz}) "
        f"- {Path(str(resolved_src)).name}"
    )
    show_window = visualize
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("[Control] Click display window and press 'q' or Ctrl+C in terminal to stop.\n")
    else:
        print("[Control] No window. Press Ctrl+C in the terminal to stop.\n")

    frame_count = 0
    fps_total = 0.0
    try:
        # Main stream ingestion and inference loop
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                print("\n[Streaming] Video playback finished or stream disconnected.")
                break

            # track() is detect() plus an id. persist=True tells the tracker to
            # remember ids from the previous frame, so the same person keeps
            # the same track_id. Without it, every frame would look like a new
            # person and Mongo would fill with one-frame documents.
            # classes=[0] drops every COCO class except person.
            # device and half are the resolved runtime, not the raw CLI text.
            started = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                classes=[0],
                imgsz=imgsz,
                device=runtime_device,
                half=runtime_half,
                verbose=False,
            )
            elapsed = time.perf_counter() - started
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            frame_count += 1
            fps_total += fps

            # boxes.id is None on a frame where the detector found people but
            # the tracker has not assigned ids yet. Those boxes are drawn and
            # not written.
            if results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for box in boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    _sync_box_to_db(box, track_id, model, imgsz, active_tracks, collection, model_name)

            # plot() draws the box, the class name, and the track id on a copy
            # of the frame. The original frame is not modified. The FPS line is
            # the TensorRT view when --tensorrt is set, and the PyTorch view
            # otherwise. --no-visualize skips this and still writes Mongo.
            if show_window:
                annotated_frame = results[0].plot(line_width=1, font_size=0.5)
                cv2.putText(
                    annotated_frame,
                    f"{runtime_label} {runtime_device} {'FP16' if runtime_half else 'FP32'}  {fps:.1f} FPS",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                try:
                    cv2.imshow(window_name, annotated_frame)
                except cv2.error as exc:
                    print(f"[Visualize] Could not open a window ({exc}). Continuing without one.")
                    show_window = False

            # waitKey(1) gives the window 1 ms to refresh and reads the keyboard.
            # The mask keeps the low 8 bits, which is the key code for "q".
            if show_window and cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[Exit] Stopped by user ('q').")
                break

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user (Ctrl+C).")
    finally:
        # finally runs on "q", on Ctrl+C, and when the video ends, so the
        # camera and the window are always released.
        if clear_on_exit:
            result = collection.delete_many({})
            print(f"[MongoDB Shutdown] Cleared {result.deleted_count} documents on exit.")

        if frame_count:
            print(f"[Speed] Average {fps_total / frame_count:.1f} FPS over {frame_count} frames ({runtime_label}).")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Command-line argument configuration
    parser = argparse.ArgumentParser(description="YOLO Person Stream Tracking Pipeline")
    parser.add_argument(
        "--source",
        type=str,
        default="data",
        help="Video file path, video directory ('data'), webcam index ('0'), or RTSP URL",
    )
    parser.add_argument("--version", type=str, help="Version digits of a .pt file in the project root, such as 10, 11, or 26")
    parser.add_argument("--model", type=str, help="File name of a .pt checkpoint in the project root")
    parser.add_argument(
        "--imgsz",
        type=str,
        default="640",
        help="Inference resolution matrix (e.g., 640, 1280, 1920) or 'native'",
    )
    parser.add_argument("--reset-db", action="store_true", help="Clear MongoDB collection BEFORE running")
    parser.add_argument("--clear-on-exit", action="store_true", help="Clear MongoDB collection AFTER script exits")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB connection URI")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="GPU index (0, 1, cuda:0) or cpu. auto uses cuda:0 when a GPU exists, otherwise cpu.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 inference on an NVIDIA GPU. Halves activation memory and raises throughput. CPU stays FP32.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Let cuDNN time convolution algorithms and keep the fastest for this imgsz. GPU only.",
    )
    parser.add_argument(
        "--tensorrt",
        action="store_true",
        help="Export the chosen .pt to a local TensorRT .engine, or reuse one, and run inference with it. Requires an NVIDIA GPU.",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the annotated video. With --tensorrt the window is the TensorRT output plus FPS. --no-visualize runs without a window.",
    )

    args = parser.parse_args()

    # No --version and no --model means the script prints the nano list and waits.
    # An unknown version is rejected and the question is asked again.
    requested_input = args.version or args.model
    chosen_model = select_nano_model(requested_input)

    run_detection(
        source_input=args.source,
        model_name=chosen_model,
        imgsz_input=args.imgsz,
        reset_db=args.reset_db,
        clear_on_exit=args.clear_on_exit,
        mongo_uri=args.mongo_uri,
        device=args.device,
        half=args.half,
        cudnn_benchmark=args.cudnn_benchmark,
        tensorrt=args.tensorrt,
        visualize=args.visualize,
    )