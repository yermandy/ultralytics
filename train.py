import os

import torch

from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.models.yolo.pose import PoseTrainer, PoseValidator

# os.environ["WANDB_MODE"] = "disabled"

args = dict(
    model="yolov8n.yaml",
    data="coco.yaml",
    project="debug",
    epochs=100,
    imgsz=640,
    name="AdamW-0.0003",
    wandb_project="adamw-schedule-free-yolo",
    workers=8,
    batch=128,
    device=1,
    optimizer="AdamWS",
    lr0=0.0003,
    lrf=1,
    warmup_epochs=0,
)

trainer = DetectionTrainer(overrides=args)
trainer.train()

# ----------------------------

args = dict(
    model="yolov8n.yaml",
    data="coco.yaml",
    project="debug",
    epochs=100,
    imgsz=640,
    name="AdamWScheduleFree-0.0003",
    wandb_project="adamw-schedule-free-yolo",
    workers=8,
    batch=128,
    device=1,
    optimizer="AdamWScheduleFree",
    lr0=0.0003,
    lrf=1,
    warmup_epochs=0,
)

trainer = DetectionTrainer(overrides=args)
trainer.train()

# ----------------------------

args = dict(
    model="yolov8n.yaml",
    data="coco.yaml",
    project="debug",
    epochs=100,
    imgsz=640,
    name="AdamWScheduleFree-0.003",
    wandb_project="adamw-schedule-free-yolo",
    workers=8,
    batch=128,
    device=1,
    optimizer="AdamWScheduleFree",
    lr0=0.003,
    lrf=1,
    warmup_epochs=0,
)

trainer = DetectionTrainer(overrides=args)
trainer.train()

# ----------------------------

args = dict(
    model="yolov8n.yaml",
    data="coco.yaml",
    project="debug",
    epochs=100,
    imgsz=640,
    name="AdamWScheduleFree-0.03",
    wandb_project="adamw-schedule-free-yolo",
    workers=8,
    batch=128,
    device=1,
    optimizer="AdamWScheduleFree",
    lr0=0.03,
    lrf=1,
    warmup_epochs=0,
)

trainer = DetectionTrainer(overrides=args)
trainer.train()
