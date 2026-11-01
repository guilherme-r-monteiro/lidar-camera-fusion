from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from fusion.coco_labels import COCO_CLASSES

INPUT_SIZE = 640
PAD_VALUE = 114
NMS_IOU_THRESHOLD = 0.45
DEFAULT_CONFIDENCE = 0.35


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Letterbox:
    tensor: np.ndarray
    scale: float
    pad_x: float
    pad_y: float


def letterbox(image: Image.Image) -> Letterbox:
    width, height = image.size
    scale = min(INPUT_SIZE / width, INPUT_SIZE / height)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = image.convert("RGB").resize((new_width, new_height))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    pad_x, pad_y = (INPUT_SIZE - new_width) / 2, (INPUT_SIZE - new_height) / 2
    canvas[int(pad_y):int(pad_y) + new_height, int(pad_x):int(pad_x) + new_width] = np.asarray(resized)
    tensor = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return Letterbox(tensor, scale, pad_x, pad_y)


def iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area + areas - inter + 1e-9)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, threshold: float = NMS_IOU_THRESHOLD) -> list[int]:
    order = scores.argsort()[::-1]
    kept: list[int] = []
    while order.size:
        head = order[0]
        kept.append(int(head))
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[iou(boxes[head], boxes[rest]) < threshold]
    return kept


def decode(output: np.ndarray, frame: Letterbox, minimum_confidence: float) -> list[Detection]:
    predictions = output[0].T
    class_scores = predictions[:, 4:]
    best_class = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)
    keep = best_score >= minimum_confidence
    predictions, best_class, best_score = predictions[keep], best_class[keep], best_score[keep]
    centers_x, centers_y, widths, heights = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    boxes = np.stack(
        [centers_x - widths / 2, centers_y - heights / 2, centers_x + widths / 2, centers_y + heights / 2], axis=1
    )
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - frame.pad_x) / frame.scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - frame.pad_y) / frame.scale
    detections: list[Detection] = []
    for class_id in np.unique(best_class):
        indexes = np.where(best_class == class_id)[0]
        for kept in non_max_suppression(boxes[indexes], best_score[indexes]):
            index = indexes[kept]
            detections.append(
                Detection(COCO_CLASSES[int(class_id)], float(best_score[index]), tuple(float(v) for v in boxes[index]))
            )
    return sorted(detections, key=lambda d: -d.confidence)


class OnnxDetector:
    def __init__(self, model_path: Path) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image_path: Path, minimum_confidence: float = DEFAULT_CONFIDENCE) -> list[Detection]:
        frame = letterbox(Image.open(image_path))
        output = self.session.run(None, {self.input_name: frame.tensor})[0]
        return decode(output, frame, minimum_confidence)
