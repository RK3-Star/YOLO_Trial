import argparse
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from ultralytics import YOLO
import yaml


def get_dataset_info(data_yaml_path):
    """
    Parses data.yaml, computes total image count, and determines
    the train/val split percentages while resolving relative paths.
    """
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    yaml_path = Path(data_yaml_path)

    if not yaml_path.exists():
        return 0, {"train_percentage": 0.0, "test_val_percentage": 0.0}

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f) or {}

    base_dir = yaml_path.parent
    root_setting = data_config.get("path", "")

    # Candidate root directories where image subfolders may reside
    candidate_roots = [
        base_dir,
        (base_dir / root_setting).resolve() if root_setting else None,
        (Path.cwd() / root_setting).resolve() if root_setting else None,
        Path.cwd(),
    ]
    candidate_roots = [p for p in candidate_roots if p and p.exists()]

    def count_images_for_keys(keys):
        for key in keys:
            raw_val = data_config.get(key)
            if not raw_val:
                continue

            paths_to_check = [raw_val] if isinstance(raw_val, str) else raw_val
            for path_str in paths_to_check:
                p = Path(path_str)
                # Strip leading parent references ('..') commonly found in Roboflow exports
                clean_parts = [part for part in p.parts if part != ".."]
                clean_rel = Path(*clean_parts) if clean_parts else Path()

                search_dirs = []
                if p.is_absolute():
                    search_dirs.append(p)

                for root in candidate_roots:
                    search_dirs.extend([root / clean_rel, root / p, root / key])

                for s_dir in search_dirs:
                    if s_dir and s_dir.exists() and s_dir.is_dir():
                        imgs = [f for f in s_dir.rglob("*") if f.suffix.lower() in valid_extensions]
                        if imgs:
                            return len(imgs)
        return 0

    train_count = count_images_for_keys(["train"])
    val_count = count_images_for_keys(["val", "valid", "test"])
    total_size = train_count + val_count

    if total_size > 0:
        train_pct = round((train_count / total_size) * 100, 2)
        val_pct = round((val_count / total_size) * 100, 2)
    else:
        train_pct, val_pct = 0.0, 0.0

    return total_size, {"train_percentage": train_pct, "test_val_percentage": val_pct}


def run_training(data_yaml, epochs=3, imgsz=320, fraction=1.0, base_model="yolov10n.pt", mongo_uri="mongodb://localhost:27017/"):
    """
    Executes model fine-tuning, calculates validation performance metrics
    (Precision, Recall, F1, mAP50), and logs all metadata into MongoDB.
    """
    # 1. Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client["yolo_mongo_db"]
    collection = db["training_results"]

    # 2. Initialize base YOLO model
    model = YOLO(base_model)

    # 3. Track start time and execute training
    start_time = datetime.now()
    print(f"[Training] Started training on '{data_yaml}' at {start_time}")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        fraction=fraction,
        verbose=True,
    )

    end_time = datetime.now()
    duration_seconds = round((end_time - start_time).total_seconds(), 2)

    # 4. Extract dataset size and split ratios post-training
    dataset_size, split_info = get_dataset_info(data_yaml)

    # 5. Evaluate validation metrics
    print("[Validation] Evaluating model metrics...")
    metrics = model.val()

    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)
    map50 = float(metrics.box.map50)

    # Calculate harmonic mean F1-Score
    if (precision + recall) > 0:
        f1_score = round(2 * (precision * recall) / (precision + recall), 4)
    else:
        f1_score = 0.0

    # 6. Format result log document matching assignment requirements
    log_document = {
        "start_time": start_time,
        "duration_seconds": duration_seconds,
        "dataset_size": dataset_size,
        "split_ratio": split_info,
        "training_results": {
            "accuracy_map50": round(map50, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": f1_score,
        },
    }

    # 7. Save result document into 'training_results' collection
    inserted = collection.insert_one(log_document)

    print("\n" + "=" * 50)
    print(f"[MongoDB] Saved training result with _id: {inserted.inserted_id}")
    print(f"Duration: {duration_seconds}s | Dataset Size: {dataset_size}")
    print(f"Split: {split_info['train_percentage']}% Train / {split_info['test_val_percentage']}% Val")
    print(f"Results -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, mAP50: {map50:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step D: YOLOv10 Custom Model Training & Evaluation")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size for training")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g., 0.1 for 10%)")
    parser.add_argument("--model", type=str, default="yolov10n.pt", help="Base YOLO model weights")
    parser.add_argument("--mongo-uri", type=str, default="mongodb://localhost:27017/", help="MongoDB URI")

    args = parser.parse_args()
    run_training(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        fraction=args.fraction,
        base_model=args.model,
        mongo_uri=args.mongo_uri,
    )