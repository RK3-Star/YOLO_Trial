"""
Stage D: fine-tune one YOLO nano model on one object, then save the run in MongoDB.

YOLO is an object detector. It does not return a single class for the whole
picture. For every object it finds, it returns a box and a confidence. Training
here starts from weights that already detect the 80 COCO classes (person, car,
and so on) and adjusts them so the model detects one chosen object, such as
Hardhat, from the Roboflow PPE export.

Order of work:
  1. Ask which nano model and which object to detect. A missing or unknown
     answer is rejected and the question is asked again. --imgsz still overrides
     the image size; 640 is only the default.
  2. Build a one-class copy of the dataset under data/single_class/<object>.
     Boxes for every other class are removed. Pictures with no box of the
     chosen object stay in the set as background.
  3. Train on the training split. During training, Ultralytics also watches the
     validation split and keeps the checkpoint that does best there. The test
     split is not used for that choice.
  4. Score that checkpoint on the test split. The four required scores are
     accuracy (mAP@0.5), precision, recall, and F1.
  5. Insert one document into yolo_mongo_db.training_results. Person tracks
     from detect_stream.py stay in a different collection, person_logs.

The three data splits, and why each exists:
  train       pictures whose boxes update the weights
  validation  pictures used to pick the best checkpoint while training
  test        pictures never used to update or pick weights; the reported score
"""

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from ultralytics import YOLO
import yaml

from inference_runtime import configure_runtime

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# This file lives in src/, so the project root is one level up. Weights the
# user can pick are the .pt files in that root, not a download.
PROJECT_DIR = Path(__file__).resolve().parent.parent
# Plots and the Ultralytics checkpoint go here. An absolute path is required:
# Ultralytics otherwise uses a machine-wide runs folder, which on this
# computer points at a different project.
RUNS_DIR = PROJECT_DIR / "runs" / "detect"
# A stable copy of the best weights, named for the object and the base model.
MODELS_DIR = PROJECT_DIR / "models"

# data.yaml names the same folder in more than one way. "val", "valid", and
# "validation" are one held-out split. "test" is a second held-out split and
# must be counted on its own. Folding test into validation under-counts the
# dataset (the PPE export would be logged as 39,578 images instead of 44,001).
SPLIT_KEYS = {
    "train": ("train",),
    "valid": ("val", "valid", "validation"),
    "test": ("test",),
}


def _class_names(data_config):
    """
    Return class names in id order.

    Roboflow writes names as a list, where position 0 is class id 0.
    Some YOLO files write names as a dict, {0: "Hardhat", 1: "Person"}.
    The id is the first number on each label line, so the order matters.
    """
    names = data_config.get("names") or []
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda k: int(k))]
    return list(names)


def _count_images(directory):
    """
    Count image files under a split directory, including nested folders.

    Label files (.txt) are not images, so they are not part of the dataset
    size. rglob walks subfolders because some exports nest images.
    """
    if not directory.is_dir():
        return 0
    return sum(
        1
        for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _find_split_dir(raw_value, yaml_dir):
    """
    Find the image folder for one data.yaml split entry.

    Roboflow writes paths like ``../train/images`` even when data.yaml already
    sits inside the dataset folder. From data/hardhat/data.yaml that path
    points at data/train/images, which does not exist. The real folder is
    data/hardhat/train/images. Ultralytics drops the leading ``../`` in that
    case. This function does the same thing, so the counted files are the
    files training will read.
    """
    entries = [raw_value] if isinstance(raw_value, str) else list(raw_value or [])
    for entry in entries:
        raw = Path(str(entry))
        candidates = [raw] if raw.is_absolute() else []
        candidates.append(yaml_dir / raw)

        raw_text = str(entry).replace("\\", "/")
        if raw_text.startswith("../"):
            candidates.append(yaml_dir / raw_text[3:])

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_dir() and any(
                f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS for f in resolved.rglob("*")
            ):
                return resolved
    return None


def _count_split(raw_value, yaml_dir):
    """Count images for one data.yaml split entry."""
    directory = _find_split_dir(raw_value, yaml_dir)
    return _count_images(directory) if directory else 0


def get_dataset_info(data_yaml_path):
    """
    Read data.yaml and describe the dataset that will be trained.

    Train, validation, and test are counted separately, then turned into
    percentages of the total. Those percentages are the assignment's
    "how the data was split". On this export they are about 69.92% train,
    20.03% validation, and 10.05% test.

    The count includes background pictures that have no box of the chosen
    object. Those pictures are still part of the training set. images_with_object,
    stored later, is the smaller count of pictures that actually contain the object.

    Returns zeros when the yaml file is missing.
    """
    empty = {
        "dataset_size": 0,
        "counts": {"train": 0, "valid": 0, "test": 0},
        "split": {"train_percentage": 0.0, "valid_percentage": 0.0, "test_percentage": 0.0},
        "classes": [],
        "dataset_url": None,
    }
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        return empty

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f) or {}

    yaml_dir = yaml_path.parent
    counts = {}
    for split_name, keys in SPLIT_KEYS.items():
        found = 0
        for key in keys:
            raw_value = data_config.get(key)
            if not raw_value:
                continue
            found = _count_split(raw_value, yaml_dir)
            if found:
                break
        counts[split_name] = found

    total = sum(counts.values())

    def percent(count):
        return round((count / total) * 100, 2) if total else 0.0

    roboflow = data_config.get("roboflow") or {}
    return {
        "dataset_size": total,
        "counts": counts,
        "split": {
            "train_percentage": percent(counts["train"]),
            "valid_percentage": percent(counts["valid"]),
            "test_percentage": percent(counts["test"]),
        },
        "classes": _class_names(data_config),
        "dataset_url": roboflow.get("url") if isinstance(roboflow, dict) else None,
    }


def _match_class(classes, object_name):
    """
    Resolve a class name to (id, canonical name).

    Returns None when the name is empty or is not one of the given classes.
    The caller then prints the legal options and asks again, instead of
    training on a class the dataset does not have.

    "hardhat" matches "Hardhat" when only one class folds to that spelling.
    The returned id is the position in this list. select_object looks the
    name up again in the full dataset list, because the label files still
    use the original id (Hardhat is 3 in the 14-class export).
    """
    if object_name is None or not str(object_name).strip():
        return None

    wanted = str(object_name).strip()
    exact = [i for i, name in enumerate(classes) if name == wanted]
    if len(exact) == 1:
        return exact[0], classes[exact[0]]

    folded = [i for i, name in enumerate(classes) if name.lower() == wanted.lower()]
    if len(folded) == 1:
        return folded[0], classes[folded[0]]
    return None


def _local_weight_files():
    """`.pt` checkpoints that actually sit in the project root."""
    return sorted(path.resolve() for path in PROJECT_DIR.glob("*.pt") if path.is_file())


def _version_digits(path):
    """'yolov10n.pt' -> '10', 'yolo11n.pt' -> '11'. The size letter is ignored."""
    stem = path.stem.lower().replace("yolo", "").replace("v", "")
    digits = ""
    for char in stem:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return digits


def _match_local_model(text, files):
    """
    Resolve a user answer to one .pt file in the project root.

    Accepts the file name, the stem, or the version digits. A path outside
    this project is rejected so training cannot start from a downloaded or
    unrelated checkpoint.
    """
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
    Ask which local checkpoint to fine-tune.

    The menu is the .pt files in the project root. A version or file name
    that is not one of those files is rejected and asked again. The returned
    path is absolute, so Ultralytics loads that file instead of downloading.
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


def _is_detection_object(name):
    """
    Keep labels that name a thing to detect.

    The PPE export also labels the absence of gear (NO-Hardhat, NO-Gloves,
    and the rest) and a Fall-Detected event. Those are not objects to find.
    The menu offers the physical objects only: Gloves, Goggles, Hardhat,
    Ladder, Mask, Person, Safety Cone, and Safety Vest. Typing NO-Hardhat
    or Fall-Detected is rejected.
    """
    lowered = name.strip().lower()
    if lowered.startswith("no-") or lowered.startswith("no "):
        return False
    if lowered.replace("_", "-") in {"fall-detected", "fall detected"}:
        return False
    return True


def _detection_objects(classes):
    """
    Class names the user can train the detector on.

    If a dataset had no names left after the filter, the full list is used
    so the menu is never empty.
    """
    objects = [name for name in classes if _is_detection_object(name)]
    return objects or list(classes)


def select_object(data_yaml_path, requested=None):
    """
    Ask which object the detector should learn.

    The menu lists detectable objects only. A number or a name both work.
    Hardhat is option 3 on this export, and it is also class id 3 in the
    original 14-class label files. A name that is not on the menu, including
    an absence label, is rejected and the question is asked again.
    --object Hardhat skips the menu when that name is legal.
    """
    source_yaml = Path(data_yaml_path)
    if not source_yaml.exists():
        raise SystemExit(f"Dataset yaml not found: {source_yaml}")

    with open(source_yaml, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f) or {}

    classes = _class_names(data_config)
    if not classes:
        raise SystemExit(f"No class names found in {source_yaml}")

    objects = _detection_objects(classes)
    matched = _match_class(objects, requested)
    if matched:
        original_id = classes.index(matched[1])
        print(f"[Object] Detect {matched[1]} (class id {original_id})")
        return matched[1]

    if requested:
        print(f"\n[Warning] '{requested}' is not an object to detect.")

    print("\n--- Objects to detect ---")
    for index, name in enumerate(objects, 1):
        print(f"  {index}. {name}")
    print("-------------------------")

    while True:
        choice = input(f"Select an object to detect (1-{len(objects)}, or a name): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(objects):
            selected = objects[int(choice) - 1]
            print(f"[Object Selected] Detect {selected}\n")
            return selected
        matched = _match_class(objects, choice)
        if matched:
            print(f"[Object Selected] Detect {matched[1]}\n")
            return matched[1]
        print(f"Error: '{choice}' is not an object to detect. Enter a number from 1 to {len(objects)}.")


def _label_fingerprint(label_dir):
    """
    Change marker for a labels folder: how many .txt files, and the newest
    modification time. prepare_single_class_dataset compares this with
    stamp.json so a second run does not rewrite tens of thousands of labels
    when the source export has not changed.
    """
    files = [p for p in label_dir.glob("*.txt")]
    newest = max((p.stat().st_mtime_ns for p in files), default=0)
    return {"count": len(files), "mtime_ns": newest}


def _link_image_tree(src_dir, dst_dir):
    """
    Add a second directory entry for every image.

    Hard links share the file bytes with the original export, so the one-class
    view does not duplicate the pictures on disk. Ultralytics resolves the
    dataset path. A directory junction would resolve back to data/hardhat and
    the trainer would read the original 14-class labels. A hard link stays
    inside data/single_class, so the filtered labels beside it are the ones
    that get read. Deleting a hard link does not delete the original file.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    for src in src_dir.rglob("*"):
        if not src.is_file() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
        linked += 1
    return linked


def _write_class_labels(src_dir, dst_dir, class_id):
    """
    Write labels that contain only one class, remapped to class 0.

    Images with no box of that class get no label file. Ultralytics treats a
    missing label as background, so those pictures still teach the model what
    is not the object. Dropping them would show the object in every training
    picture and the detector would fire more false alarms.

    Old filtered files are deleted first. Otherwise a previous object, such as
    Gloves, could leave boxes behind in a folder that is now supposed to be
    Hardhat-only. Each object has its own folder, so this only clears a rebuild
    of the same object.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for old in dst_dir.glob("*.txt"):
        old.unlink()

    images_with_object = 0
    boxes = 0
    for src in src_dir.glob("*.txt"):
        kept = []
        for line in src.read_text(encoding="utf-8").splitlines():
            # A YOLO box line is: class_id  center_x  center_y  width  height.
            # All four box numbers are fractions of the image size, between 0 and 1.
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if cls_id == class_id:
                # The one-class dataset has a single name, so this box becomes class 0.
                kept.append("0 " + " ".join(parts[1:5]))
        if not kept:
            continue
        (dst_dir / src.name).write_text("\n".join(kept) + "\n", encoding="utf-8")
        images_with_object += 1
        boxes += len(kept)
    return images_with_object, boxes


def prepare_single_class_dataset(data_yaml_path, object_name):
    """
    Build a one-class YOLO dataset for ``object_name`` next to the source export.

    The source labels in data/hardhat stay untouched. Training reads
    data/single_class/<object>/data.yaml, which declares nc: 1 and a single
    name. The original class id (3 for Hardhat) is rewritten to 0, because a
    one-class model only has class 0.
    """
    source_yaml = Path(data_yaml_path)
    if not source_yaml.exists():
        raise SystemExit(f"Dataset yaml not found: {source_yaml}")

    with open(source_yaml, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f) or {}

    classes = _class_names(data_config)
    matched = _match_class(classes, object_name)
    if matched is None:
        available = ", ".join(classes) if classes else "(none)"
        raise SystemExit(f"Object '{object_name}' is not an option. Available classes: {available}")
    class_id, class_name = matched
    yaml_dir = source_yaml.parent

    split_dirs = {}
    for split_name, keys in SPLIT_KEYS.items():
        for key in keys:
            raw_value = data_config.get(key)
            if not raw_value:
                continue
            image_dir = _find_split_dir(raw_value, yaml_dir)
            if image_dir is not None:
                split_dirs[split_name] = image_dir
                break

    if "train" not in split_dirs:
        raise SystemExit(f"No training images found from {source_yaml}")

    out_dir = yaml_dir.parent / "single_class" / class_name
    stamp_path = out_dir / "stamp.json"
    fingerprint = {
        "source": str(source_yaml.resolve()),
        "object": class_name,
        "class_id": class_id,
        "labels": {},
    }
    for split_name, image_dir in split_dirs.items():
        label_dir = image_dir.parent / "labels"
        if not label_dir.is_dir():
            raise SystemExit(f"No labels folder beside {image_dir}")
        fingerprint["labels"][split_name] = _label_fingerprint(label_dir)

    # Skip the rebuild when the source labels have not changed since last time.
    if stamp_path.exists():
        try:
            saved = json.loads(stamp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            saved = {}
        if saved.get("fingerprint") == fingerprint and (out_dir / "data.yaml").exists():
            print(f"[Object] Reusing one-class dataset at {out_dir}")
            return {
                "yaml": str(out_dir / "data.yaml"),
                "object": class_name,
                "class_id": class_id,
                "images_with_object": saved.get("images_with_object", {}),
                "object_boxes": saved.get("object_boxes", {}),
            }

    print(f"[Object] Building a one-class dataset for '{class_name}' (source class id {class_id})")
    images_with_object = {}
    object_boxes = {}
    yaml_paths = {}
    folder_names = {"train": "train", "valid": "valid", "test": "test"}
    yaml_keys = {"train": "train", "valid": "val", "test": "test"}

    for split_name, image_dir in split_dirs.items():
        folder = folder_names[split_name]
        linked = _link_image_tree(image_dir, out_dir / folder / "images")
        with_object, boxes = _write_class_labels(image_dir.parent / "labels", out_dir / folder / "labels", class_id)
        images_with_object[split_name] = with_object
        object_boxes[split_name] = boxes
        yaml_paths[yaml_keys[split_name]] = f"{folder}/images"
        print(f"[Object] {split_name}: {with_object} images contain {class_name} ({boxes} boxes), {linked} images linked")

    roboflow = data_config.get("roboflow") or {}
    dataset_yaml = {
        **yaml_paths,
        "nc": 1,  # number of classes the trainer should learn
        "names": [class_name],
    }
    if isinstance(roboflow, dict) and roboflow.get("url"):
        dataset_yaml["roboflow"] = {"url": roboflow["url"]}

    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    stamp_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "images_with_object": images_with_object,
                "object_boxes": object_boxes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "yaml": str(out_dir / "data.yaml"),
        "object": class_name,
        "class_id": class_id,
        "images_with_object": images_with_object,
        "object_boxes": object_boxes,
    }


def _f1(precision, recall):
    """
    Harmonic mean of precision and recall.

    F1 = 2 * precision * recall / (precision + recall).
    An average would still look decent if the model predicted almost nothing
    but the few boxes it did predict were right (high precision, tiny recall).
    The harmonic mean stays low unless both numbers are high. If both are
    zero the denominator would be zero, so F1 is defined as 0.
    """
    if precision + recall <= 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _save_trained_weights(model, object_name, base_model):
    """
    Copy the best checkpoint into models/ inside this project.

    Ultralytics also keeps its own copy under runs/detect. The models/ file
    is the stable path recorded in Mongo, named for the object and the
    starting checkpoint, for example models/Hardhat_yolov10n.pt.
    """
    best = Path(getattr(model.trainer, "best", ""))
    if not best.is_file():
        best = Path(getattr(model.trainer, "last", ""))
    if not best.is_file():
        raise SystemExit("Training finished but no best.pt or last.pt was written.")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    destination = MODELS_DIR / f"{object_name}_{Path(base_model).stem}.pt"
    shutil.copy2(best, destination)
    return best, destination


def run_training(
    data_yaml,
    epochs=3,
    imgsz=640,
    fraction=1.0,
    base_model="yolov10n.pt",
    mongo_uri="mongodb://localhost:27017/",
    object_name="Hardhat",
    reset_db=False,
    clear_on_exit=False,
    device="auto",
    half=False,
    cudnn_benchmark=False,
):
    """
    Fine-tune a YOLO checkpoint on one class, score it on the held-out test
    split, and insert one document into ``yolo_mongo_db.training_results``.

    Boxes for every other class are removed before training. Images that do
    not contain ``object_name`` stay in the set as background.

    Detection has no single accuracy number. The value stored as ``accuracy``
    is mAP@0.5 (mean average precision at IoU 0.5), which is the standard
    detection score. ``accuracy_metric`` records that name.

    ``fraction`` matches Ultralytics: only the training split is subsampled,
    with ``round(train_count * fraction)`` images when fraction is below 1.
    Python rounds a .5 to the nearest even integer, so 0.1 of 30,765 training
    images is 3,076, not 3,077. Validation and test images are left whole.
    ``dataset_size`` stays the full image count. ``train_images_used`` records
    how many training images this run actually used.

    ``imgsz`` is the square side, in pixels, that every picture is resized to.
    The default 640 matches the Roboflow export and the size these nano models
    were originally trained at. It must be a multiple of 32, the model stride.
    --imgsz 320 is still legal: it is faster and uses less memory, and it
    throws away detail. A size above 640 only upscales pictures that are
    already 640x640.
    """
    # Same database as detection. This script writes a separate collection.
    client = MongoClient(mongo_uri)
    try:
        db = client["yolo_mongo_db"]
        collection = db["training_results"]

        # Optional. Empties training_results before this run starts.
        # person_logs, written by detect_stream.py, is left alone.
        if reset_db:
            deleted = collection.delete_many({})
            print(f"[MongoDB Startup] Cleared {deleted.deleted_count} existing documents from 'training_results'.")

        # Step 1. Keep only the chosen object's boxes.
        prepared = prepare_single_class_dataset(data_yaml, object_name)
        train_yaml = prepared["yaml"]
        # Step 2. Count every image, including pictures with no box of this object.
        info = get_dataset_info(train_yaml)
        counts = info["counts"]
        # Same cutoff Ultralytics uses in data/base.py: fraction applies to train only.
        train_images_used = counts["train"] if fraction >= 1 else round(counts["train"] * fraction)
        eval_split = "test" if counts["test"] else "val"
        images_with_object = prepared["images_with_object"]
        object_boxes = prepared["object_boxes"]

        # Step 3. Start from the nano weights chosen in the menu, then fine-tune them.
        # device "auto" is cuda:0 when a GPU exists, otherwise cpu.
        # --half turns on FP16 for the test score. On a GPU, training uses AMP,
        # which is the training form of 16-bit math. On CPU both stay FP32.
        # TensorRT is inference-only, so it is a detect_stream.py flag.
        runtime_device, runtime_half, cudnn_on = configure_runtime(device, half, cudnn_benchmark)
        model = YOLO(base_model)

        # Duration covers model.train() only. The test pass below is not included.
        start_time = datetime.now()
        print(f"[Training] Started training '{prepared['object']}' on '{train_yaml}' at {start_time}")
        print(
            f"[Dataset] {info['dataset_size']} images | "
            f"train {info['split']['train_percentage']}% | "
            f"valid {info['split']['valid_percentage']}% | "
            f"test {info['split']['test_percentage']}%"
        )
        print(
            f"[Object] {prepared['object']}: "
            f"{sum(images_with_object.values())} images with a box, "
            f"{sum(object_boxes.values())} boxes. Other classes are not trained."
        )
        print(f"[Dataset] Training uses {train_images_used} of {counts['train']} train images (fraction={fraction})")

        train_kwargs = dict(
            data=train_yaml,
            epochs=epochs,
            imgsz=imgsz,  # 640 unless --imgsz says otherwise
            fraction=fraction,  # shrinks the training split only
            project=str(RUNS_DIR),
            device=runtime_device,
            verbose=True,
        )
        # AMP is the training form of 16-bit math. Set it only when --half
        # resolved to a GPU. Otherwise Ultralytics keeps its own default.
        if runtime_half:
            train_kwargs["amp"] = True
            print("[Half] Training uses automatic mixed precision on this GPU. The test score uses FP16.")
        model.train(**train_kwargs)

        end_time = datetime.now()
        duration_seconds = round((end_time - start_time).total_seconds(), 2)
        run_weights, saved_weights = _save_trained_weights(model, prepared["object"], base_model)
        print(f"[Weights] Best checkpoint: {run_weights}")
        print(f"[Weights] Local copy: {saved_weights}")

        # Score the split that was not used to update weights. Training already
        # validates on `val`; this pass is the held-out test result.
        # name= puts the pictures in runs/detect/test. Ultralytics would
        # otherwise call that folder val, because val() is the mode name.
        print(f"[Evaluation] Scoring the trained weights on the {eval_split} split...")
        metrics = model.val(
            data=train_yaml,
            split=eval_split,
            imgsz=imgsz,
            project=str(RUNS_DIR),
            name=eval_split,
            device=runtime_device,
            half=runtime_half,
        )
        # A second run finds runs/detect/test already there and writes test2.
        picture_dirs = [path for path in RUNS_DIR.glob(f"{eval_split}*") if path.is_dir()]
        picture_dir = max(picture_dirs, key=lambda path: path.stat().st_mtime) if picture_dirs else RUNS_DIR / eval_split
        print(f"[Evaluation] Test pictures: {picture_dir}")

        # All four numbers come from the test split, not from the validation
        # pass that ran inside model.train().
        #
        # Precision: of the boxes the model predicted, the fraction that
        # covered a real object. True positives / (true positives + false positives).
        # Recall: of the real objects, the fraction the model found.
        # True positives / (true positives + false negatives).
        # mp and mr are those values averaged over the classes. Here there is
        # one class, so they are the Hardhat (or chosen object) scores.
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        # Accuracy is stored as mAP@0.5. A predicted box counts as correct
        # when its overlap with the labeled box (intersection over union) is
        # at least 0.5. Average precision summarizes the precision-recall
        # curve, and mAP is that number for the class. Classic accuracy,
        # (TP + TN) / everything, does not apply: a picture has no finite
        # count of regions that correctly contain nothing.
        accuracy = float(metrics.box.map50)
        f1_score = _f1(precision, recall)

        # Step 4. One document per run. The assignment fields are:
        #   start time, duration, dataset size, split percentages,
        #   and accuracy / precision / recall / F1.
        # The other keys make the document readable without this source file.
        log_document = {
            "start_time": start_time,  # clock time just before model.train()
            "duration_seconds": duration_seconds,  # seconds inside model.train() only
            "object": prepared["object"],  # the single class that was learned
            "dataset_size": info["dataset_size"],  # train + valid + test images
            "images_with_object": images_with_object,  # pictures that contain a box
            "object_boxes": object_boxes,  # how many boxes of that class
            "fraction": fraction,  # 1.0 means the whole training split
            "train_images_used": train_images_used,
            "split": {
                "train_images": counts["train"],
                "valid_images": counts["valid"],
                "test_images": counts["test"],
                "train_percentage": info["split"]["train_percentage"],
                "valid_percentage": info["split"]["valid_percentage"],
                "test_percentage": info["split"]["test_percentage"],
            },
            "evaluated_on": eval_split,  # "test" when that folder exists
            "model": base_model,  # absolute path of the .pt file in the project root
            "run_dir": str(run_weights.parent.parent),  # runs/detect/train* inside this project
            "weights_path": str(saved_weights),  # stable copy under models/
            "epochs": epochs,  # full passes over the training images
            "imgsz": imgsz,  # square side used for train and for the test score
            "device": runtime_device,  # "cpu" or the CUDA index that ran the job
            "half": runtime_half,  # True when the test score ran in FP16
            "cudnn_benchmark": cudnn_on,  # True when cuDNN timed conv algorithms
            "classes": info["classes"],  # ["Hardhat"] after the one-class filter
            "dataset_url": info["dataset_url"],  # Roboflow page the export came from
            "training_results": {
                "accuracy": round(accuracy, 4),  # mAP@0.5 on the test split
                "accuracy_metric": "mAP50",  # says what the accuracy number is
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1_score, 4),
            },
        }

        inserted = collection.insert_one(log_document)

        print("\n" + "=" * 50)
        print(f"[MongoDB] Saved training result with _id: {inserted.inserted_id}")
        print(f"Duration: {duration_seconds}s | Dataset size: {info['dataset_size']} | Train images used: {train_images_used}")
        print(
            f"Split: {info['split']['train_percentage']}% train / "
            f"{info['split']['valid_percentage']}% valid / "
            f"{info['split']['test_percentage']}% test"
        )
        print(
            f"Test results -> Accuracy (mAP50): {accuracy:.4f}, "
            f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}"
        )
        print(f"Weights -> {saved_weights}")
        print("=" * 50 + "\n")

        # Optional. The document is inserted and printed first, then removed.
        # person_logs, written by detect_stream.py, is left alone.
        if clear_on_exit:
            deleted = collection.delete_many({})
            print(f"[MongoDB] Deleted {deleted.deleted_count} documents from 'training_results'.")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step D: YOLO training, test-split evaluation, MongoDB log")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training and evaluation. 640 matches this dataset and the YOLO nano models.")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use (validation and test stay whole). Example: 0.1",
    )
    parser.add_argument("--version", type=str, help="Version digits of a .pt file in the project root, such as 10, 11, or 26.")
    parser.add_argument("--model", type=str, help="File name of a .pt checkpoint in the project root. Asks if omitted or missing.")
    parser.add_argument("--object", type=str, help="Class to train. Asks if omitted, and asks again when the name is not in the dataset.")
    parser.add_argument("--reset-db", action="store_true", help="Delete every document in training_results before training starts.")
    parser.add_argument("--clear-on-exit", action="store_true", help="Delete every document in training_results after the run is saved and printed.")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB URI")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="GPU index (0, 1, cuda:0) or cpu. auto uses cuda:0 when a GPU exists, otherwise cpu.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="FP16 for the test score on an NVIDIA GPU, and AMP during training. CPU stays FP32.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Let cuDNN time convolution algorithms and keep the fastest for this imgsz. GPU only.",
    )

    args = parser.parse_args()
    # Menus run first. A valid --model / --object skips its question.
    # A value that is not on the list is rejected and asked again.
    chosen_model = select_nano_model(args.version or args.model)
    chosen_object = select_object(args.data, args.object)
    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        fraction=args.fraction,
        base_model=chosen_model,
        mongo_uri=args.mongo_uri,
        object_name=chosen_object,
        reset_db=args.reset_db,
        clear_on_exit=args.clear_on_exit,
        device=args.device,
        half=args.half,
        cudnn_benchmark=args.cudnn_benchmark,
    )
