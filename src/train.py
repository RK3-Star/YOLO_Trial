import argparse
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from ultralytics import YOLO
from ultralytics.utils import SETTINGS
from ultralytics.utils.checks import check_file
import yaml


def get_dataset_info(data_yaml_path):
  """Parses data.yaml (local or built-in) to count total images and compute train/val split percentages."""
  try:
    yaml_path = Path(check_file(data_yaml_path))
  except Exception:
    yaml_path = Path(data_yaml_path).resolve()

  if not yaml_path.exists():
    return 0, {"train_percentage": 0.0, "test_val_percentage": 0.0}

  with open(yaml_path, "r", encoding="utf-8") as f:
    data_config = yaml.safe_load(f)

  base_dir = yaml_path.parent
  root_setting = data_config.get("path", "")
  global_datasets_dir = Path(
      SETTINGS.get("datasets_dir", Path.home() / "datasets")
  )

  dataset_name = Path(data_yaml_path).stem

  # Candidate locations for dataset root (including Ultralytics global datasets directory)
  candidate_roots = [
      global_datasets_dir / dataset_name,
      Path.cwd() / "datasets" / dataset_name,
      (
          Path(root_setting)
          if root_setting and Path(root_setting).is_absolute()
          else None
      ),
      (base_dir / root_setting).resolve() if root_setting else None,
      (Path.cwd() / root_setting).resolve() if root_setting else None,
      base_dir,
      Path.cwd(),
  ]

  dataset_root = next(
      (p for p in candidate_roots if p and p.exists()), base_dir
  )

  def resolve_dir(key):
    rel_path = data_config.get(key, "")
    if not rel_path:
      return None
    p = Path(rel_path)
    if p.is_absolute():
      return p
    if (dataset_root / p).exists():
      return dataset_root / p
    if (base_dir / p).exists():
      return base_dir / p
    return None

  train_dir = resolve_dir("train")
  val_dir = resolve_dir("val")

  valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

  def count_images(directory):
    if not directory or not directory.exists():
      return 0
    return len([
        f for f in directory.rglob("*") if f.suffix.lower() in valid_extensions
    ])

  train_count = count_images(train_dir)
  val_count = count_images(val_dir)
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
    base_model="yolo26n.pt",
    mongo_uri="mongodb://localhost:27017/",
):
  # 1. Connect to MongoDB
  client = MongoClient(mongo_uri)
  db = client["yolo_mongo_db"]
  collection = db["training_results"]

  # 2. Initialize YOLO model
  model = YOLO(base_model)

  # 3. Measure start time and execute training
  start_time = datetime.now()
  print(f"[Training] Started training on dataset '{data_yaml}' at {start_time}")

  model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, verbose=True)

  end_time = datetime.now()
  duration_seconds = round((end_time - start_time).total_seconds(), 2)

  # 4. Extract dataset info post-training (ensures auto-downloaded datasets exist)
  dataset_size, split_info = get_dataset_info(data_yaml)

  # 5. Extract validation performance metrics
  print("[Validation] Evaluating model metrics on validation split...")
  metrics = model.val()

  precision = float(metrics.box.mp)
  recall = float(metrics.box.mr)
  map50 = float(metrics.box.map50)

  # Calculate F1 score: 2 * (Precision * Recall) / (Precision + Recall)
  if (precision + recall) > 0:
    f1_score = round(2 * (precision * recall) / (precision + recall), 4)
  else:
    f1_score = 0.0

  # 6. Format document matching Step D requirements
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

  # 7. Save document to MongoDB collection 'training_results'
  inserted = collection.insert_one(log_document)

  print("\n" + "=" * 60)
  print("[MongoDB] Training session logged successfully!")
  print(f"Collection  : training_results")
  print(f"Document ID : {inserted.inserted_id}")
  print(f"Start Time  : {start_time}")
  print(f"Duration    : {duration_seconds} seconds")
  print(f"Dataset Size: {dataset_size} images")
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
      help="Path to data.yaml file or built-in dataset name (e.g. coco8.yaml)",
  )
  parser.add_argument(
      "--epochs", type=int, default=3, help="Number of training epochs"
  )
  parser.add_argument(
      "--imgsz", type=int, default=320, help="Image size for training"
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
      base_model=args.model,
  )