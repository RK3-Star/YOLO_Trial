import argparse
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_file
import yaml


def get_dataset_info(data_yaml_path):
  """Parses data.yaml (Roboflow local or Ultralytics built-in) to accurately count images and compute train/val split percentages, correctly resolving '../' paths."""
  valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

  # 1. Resolve YAML location
  try:
    yaml_path = Path(check_file(data_yaml_path))
  except Exception:
    yaml_path = Path(data_yaml_path).resolve()

  if not yaml_path.exists():
    return 0, {"train_percentage": 0.0, "test_val_percentage": 0.0}

  # 2. Read configuration file
  with open(yaml_path, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f) or {}

  base_dir = yaml_path.parent
  root_setting = data_config.get("path", "")
  global_datasets_dir = Path(
      SETTINGS.get("datasets_dir", Path.home() / "datasets")
  )
  dataset_name = Path(data_yaml_path).stem

  # List candidate directories where image subfolders might reside
  candidate_roots = [
      base_dir,
      global_datasets_dir / dataset_name,
      Path.cwd() / "datasets" / dataset_name,
      (
          Path(root_setting)
          if root_setting and Path(root_setting).is_absolute()
          else None
      ),
      (base_dir / root_setting).resolve() if root_setting else None,
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

        # Fix Roboflow '../' path issue by stripping leading parent references
        clean_parts = [part for part in p.parts if part != ".."]
        clean_rel = Path(*clean_parts) if clean_parts else Path()

        search_dirs = []
        if p.is_absolute():
          search_dirs.append(p)

        for root in candidate_roots:
          search_dirs.extend([
              root / clean_rel,
              root / p,
              root / key,
              root / key / "images",
          ])

        for s_dir in search_dirs:
          if s_dir and s_dir.exists() and s_dir.is_dir():
            imgs = [
                f
                for f in s_dir.rglob("*")
                if f.suffix.lower() in valid_extensions
            ]
            if imgs:
              return len(imgs)
    return 0

  train_count = count_images_for_keys(["train"])
  val_count = count_images_for_keys(["val", "valid", "test"])
  total_size = train_count + val_count

  train_pct = (
      round((train_count / total_size) * 100, 2) if total_size > 0 else 0.0
  )
  val_pct = round((val_count / total_size) * 100, 2) if total_size > 0 else 0.0

  return total_size, {
      "train_percentage": train_pct,
      "test_val_percentage": val_pct,
  }


def run_training(
    data_yaml,
    epochs=3,
    imgsz=320,
    fraction=1.0,
    base_model="yolo26n.pt",
    mongo_uri="mongodb://localhost:27017/",
):
  client = MongoClient(mongo_uri)
  db = client["yolo_mongo_db"]
  collection = db["training_results"]

  model = YOLO(base_model)

  start_time = datetime.now()
  print(
      f"[Training] Started training on '{data_yaml}' (fraction={fraction},"
      f" epochs={epochs}, imgsz={imgsz}) at {start_time}"
  )

  # Train model with fraction subsampling for fast completion
  model.train(
      data=data_yaml,
      epochs=epochs,
      imgsz=imgsz,
      fraction=fraction,
      verbose=True,
  )

  end_time = datetime.now()
  duration_seconds = round((end_time - start_time).total_seconds(), 2)

  # Extract dataset metadata post-training
  dataset_size, split_info = get_dataset_info(data_yaml)

  print("[Validation] Evaluating model metrics on validation split...")
  metrics = model.val()

  precision = float(metrics.box.mp)
  recall = float(metrics.box.mr)
  map50 = float(metrics.box.map50)

  if (precision + recall) > 0:
    f1_score = round(2 * (precision * recall) / (precision + recall), 4)
  else:
    f1_score = 0.0

  log_document = {
      "start_time": start_time,
      "duration_seconds": duration_seconds,
      "dataset_size": dataset_size,
      "train_test_split": split_info,
      "training_results": {
          "accuracy_map50": round(map50, 4),
          "precision": round(precision, 4),
          "recall": round(recall, 4),
          "f1_score": f1_score,
      },
  }

  inserted = collection.insert_one(log_document)

  print("\n" + "=" * 60)
  print("[MongoDB] Training session logged successfully!")
  print(f"Collection  : training_results")
  print(f"Document ID : {inserted.inserted_id}")
  print(f"Start Time  : {start_time}")
  print(f"Duration    : {duration_seconds} seconds")
  print(f"Dataset Size: {dataset_size} total images")
  print(
      f"Split       : {split_info['train_percentage']}% Train /"
      f" {split_info['test_val_percentage']}% Val"
  )
  print(
      f"Results     : Precision={round(precision, 4)},"
      f" Recall={round(recall, 4)}, F1={f1_score}, Accuracy"
      f" (mAP50)={round(map50, 4)}"
  )
  print("=" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Train YOLO model and log metrics to MongoDB"
  )
  parser.add_argument(
      "--data",
      type=str,
      required=True,
      help="Path to data.yaml file or dataset name",
  )
  parser.add_argument(
      "--epochs", type=int, default=3, help="Number of training epochs"
  )
  parser.add_argument(
      "--imgsz", type=int, default=320, help="Image size for training"
  )
  parser.add_argument(
      "--fraction",
      type=float,
      default=1.0,
      help="Fraction of dataset to train on (e.g. 0.05 for 5%)",
  )
  parser.add_argument(
      "--model",
      type=str,
      default="yolo26n.pt",
      help="Base YOLO model weights",
  )

  args = parser.parse_args()
  run_training(
      data_yaml=args.data,
      epochs=args.epochs,
      imgsz=args.imgsz,
      fraction=args.fraction,
      base_model=args.model,
  )