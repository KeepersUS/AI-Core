import os
import sys
import json
import time
import traceback
import logging
from typing import Optional, Dict, Any, List
import subprocess
import threading

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# RF-DETR & project imports
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_AI_DEV_DIR = os.path.join(_REPO_ROOT, "ai_dev")
sys.path.insert(0, _AI_DEV_DIR)

try:
    import torch
    from PIL import Image
    from torchvision import transforms
    from str_config import STR_CORE_CLASSES
    from rfdetr import RFDETRLarge
    LOCAL_IMPORT_OK = True
except Exception as e:
    LOCAL_IMPORT_OK = False
    LOCAL_IMPORT_ERROR_MSG = str(e)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RF-DETR API",
    version="2.0.0",
    description="Object detection API powered by RF-DETR with per-class confidence thresholds",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Runtime config — paths overridable via environment variables
# ---------------------------------------------------------------------------
WEIGHTS_PATH = os.environ.get(
    "WEIGHTS_PATH",
    os.path.join(_AI_DEV_DIR, "checkpoint_best_ema.pth"),
)
THRESHOLDS_PATH = os.environ.get(
    "THRESHOLDS_PATH",
    os.path.join(_AI_DEV_DIR, "per_class_thresholds_CL4-Martin.json"),
)

# IoU threshold for matching detections to reference objects (preserved from original eval contract)
_IOU_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_detector_instance: Optional[Any] = None
_request_counter: int = 0


# ---------------------------------------------------------------------------
# RFDETRDetector
# ---------------------------------------------------------------------------
class RFDETRDetector:
    """RF-DETR model with per-class confidence thresholds."""

    def __init__(self, weights_path: str = WEIGHTS_PATH,
                 thresholds_path: str = THRESHOLDS_PATH,
                 use_gpu: bool = True):
        self.class_list: List[str] = STR_CORE_CLASSES
        self.preprocess = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])

        if not os.path.exists(thresholds_path):
            raise FileNotFoundError(f"Thresholds file not found: {thresholds_path}")
        with open(thresholds_path) as f:
            thresh_data = json.load(f)
        self.per_class_thresholds: Dict[str, float] = thresh_data["thresholds"]
        self.global_threshold: float = thresh_data.get("global_baseline", 0.89)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        self.torch_model, self.num_classes, self.device = self._load_model(weights_path, use_gpu)

    def _load_model(self, weights_path: str, use_gpu: bool):
        import io as _io
        logger.info(f"RF-DETR: loading checkpoint from {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        if "model_ema" in checkpoint:
            state_dict = checkpoint["model_ema"]
            logger.info("RF-DETR: using EMA weights")
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
            logger.info("RF-DETR: using model weights")
        else:
            state_dict = checkpoint

        num_classes = state_dict["class_embed.weight"].shape[0]
        logger.info(f"RF-DETR: detected {num_classes} classes from checkpoint")

        # pretrain_weights=None skips the default weight download entirely;
        # our checkpoint is loaded below via load_state_dict.
        old_stdout = sys.stdout
        sys.stdout = _io.StringIO()
        try:
            model = RFDETRLarge(pretrain_weights=None)
        finally:
            sys.stdout = old_stdout

        # Navigate to inner torch model
        torch_model = model.model.model if hasattr(model.model, "model") else model.model

        # Resize classification heads to match checkpoint shape
        for name, param in state_dict.items():
            if "class_embed" in name or "enc_out_class_embed" in name:
                parts = name.split(".")
                obj = torch_model
                for part in parts[:-1]:
                    obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
                target = getattr(obj, parts[-1])
                if target.shape != param.shape:
                    setattr(obj, parts[-1], torch.nn.Parameter(param.clone()))

        torch_model.load_state_dict(state_dict, strict=False)

        device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        torch_model.to(device)
        torch_model.eval()
        logger.info(f"RF-DETR: model loaded on {device}")
        return torch_model, num_classes, device

    def detect(self, image_path: str,
               confidence_threshold_override: Optional[float] = None) -> List[Dict[str, Any]]:
        """Run inference; apply per-class thresholds (or override) and return detections."""
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.torch_model(img_tensor)

        pred_logits = outputs["pred_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]
        probs = pred_logits.softmax(-1)
        scores, labels = probs.max(-1)

        detections: List[Dict[str, Any]] = []
        for score, label, box in zip(scores, labels, pred_boxes):
            score_val = float(score.item())
            label_val = int(label.item())

            if label_val >= self.num_classes or label_val >= len(self.class_list):
                continue

            class_name = self.class_list[label_val]
            threshold = (
                confidence_threshold_override
                if confidence_threshold_override is not None
                else self.per_class_thresholds.get(class_name, self.global_threshold)
            )
            if score_val < threshold:
                continue

            cx, cy, w, h = box.tolist()
            x1 = max(0, int((cx - w / 2) * orig_w))
            y1 = max(0, int((cy - h / 2) * orig_h))
            x2 = min(orig_w, int((cx + w / 2) * orig_w))
            y2 = min(orig_h, int((cy + h / 2) * orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({"class": class_name, "bbox": [x1, y1, x2, y2], "confidence": score_val})

        return detections


# ---------------------------------------------------------------------------
# STRConfusionMatrix — replaces model_tests.ConfusionMatrix for STR_CORE_CLASSES
# ---------------------------------------------------------------------------
class _BBox:
    __slots__ = ("class_name", "bbox")

    def __init__(self, class_name: str, bbox: List[int]):
        self.class_name = class_name
        self.bbox = bbox


class STRConfusionMatrix:
    """Per-class confusion matrix scoped to STR_CORE_CLASSES."""

    def __init__(self, reference_json: List[Dict[str, Any]]):
        self.class_list: List[str] = STR_CORE_CLASSES
        self.index_dict: Dict[str, int] = {cls: i for i, cls in enumerate(self.class_list)}
        n = len(self.class_list)
        self.confusion_matrix: List[List[int]] = [[0] * n for _ in range(n)]
        self.reference_object_array: List[_BBox] = [
            _BBox(item["class"], item["bbox"]) for item in reference_json
        ]
        self.unmatched_objects: List[_BBox] = []
        self.missing_objects: List[_BBox] = []

    def _iou(self, b1: List[int], b2: List[int]) -> float:
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0.0

    def handle_object_data(self, class_name: str, bbox: List[int]) -> Optional[_BBox]:
        detected = _BBox(class_name, bbox)
        for ref_obj in self.reference_object_array:
            if self._iou(detected.bbox, ref_obj.bbox) > _IOU_THRESHOLD:
                ri = self.index_dict.get(ref_obj.class_name, -1)
                di = self.index_dict.get(detected.class_name, -1)
                if ri >= 0 and di >= 0:
                    self.confusion_matrix[ri][di] += 1
                return ref_obj
        self.unmatched_objects.append(detected)
        return None

    def get_confusion_matrix(self) -> List[List[int]]:
        return self.confusion_matrix

    def get_matrix_metrics(self):
        n = len(self.class_list)
        class_metrics = []
        total_prec = total_f1 = total_acc = 0.0
        prec_count = f1_count = 0

        for class_name in self.class_list:
            idx = self.index_dict[class_name]
            tp = self.confusion_matrix[idx][idx]
            fp = sum(self.confusion_matrix[i][idx] for i in range(n) if i != idx)
            fn = sum(self.confusion_matrix[idx][j] for j in range(n) if j != idx)
            tn = sum(
                self.confusion_matrix[i][j]
                for i in range(n) for j in range(n)
                if i != idx and j != idx
            )
            for obj in self.unmatched_objects:
                if obj.class_name == class_name:
                    fp += 1
                else:
                    tn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
            acc  = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

            class_metrics.append({
                "class_name": class_name,
                "precision": prec,
                "sensitivity": sens,
                "f1_score": f1,
                "accuracy": acc,
            })

            if prec > 0:
                total_prec += prec
                prec_count += 1
            if f1 > 0:
                total_f1 += f1
                f1_count += 1
            total_acc += acc

        count = len(self.class_list)
        mean_ap  = total_prec / prec_count if prec_count > 0 else 0.0
        mean_f1  = total_f1  / f1_count   if f1_count  > 0 else 0.0
        mean_acc = total_acc / count       if count     > 0 else 0.0
        return class_metrics, mean_ap, mean_f1, mean_acc


def _format_rfdetr_results(
    class_metrics, mean_ap: float, mean_f1: float, mean_accuracy: float,
    matrix: STRConfusionMatrix,
) -> Dict[str, Any]:
    return {
        "mean_average_precision": mean_ap,
        "mean_f1_score": mean_f1,
        "class_metrics": {
            m["class_name"]: {
                "precision": m["precision"],
                "sensitivity": m["sensitivity"],
                "f1_score": m["f1_score"],
            }
            for m in class_metrics
        },
        "confusion_matrix": matrix.get_confusion_matrix(),
        "unmatched_objects": len(matrix.unmatched_objects),
        "missing_objects": len(matrix.missing_objects),
    }


def _save_detections(image_path: str, detections: List[Dict[str, Any]]) -> None:
    os.makedirs("outputs", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join("outputs", f"{base}_detections.json"), "w") as f:
        json.dump(detections, f, indent=2)


# ---------------------------------------------------------------------------
# Core inference helper (used by /run)
# ---------------------------------------------------------------------------
def _run_rfdetr(
    image_path: str,
    reference_objects: List[Dict[str, Any]],
    use_gpu: bool,
    create_overlay: bool,
) -> Dict[str, Any]:
    global _detector_instance

    if not LOCAL_IMPORT_OK:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Failed to import local project modules",
                "message": f"Could not import RF-DETR dependencies: {LOCAL_IMPORT_ERROR_MSG}",
                "hint": "Ensure rfdetr is installed and ai_dev/ is present.",
                "traceback": None,
            },
        )

    if _detector_instance is None:
        try:
            _detector_instance = RFDETRDetector(use_gpu=use_gpu)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=503,
                detail={"error": "Failed to load RF-DETR model", "message": str(e), "traceback": tb},
            )

    try:
        detections = _detector_instance.detect(image_path)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model prediction failed",
                "message": str(e),
                "traceback": tb,
                "image_path": image_path,
            },
        )

    detections_for_cm   = [{"class": d["class"], "bbox": d["bbox"], "confidence": d["confidence"]} for d in detections]
    detections_for_save = [{"object": d["class"], "bbox": d["bbox"], "confidence": d["confidence"]} for d in detections]

    try:
        matrix = STRConfusionMatrix(reference_objects)
        for det in detections_for_cm:
            matrix.handle_object_data(det["class"], det["bbox"])
        class_metrics, mean_ap, mean_f1, mean_accuracy = matrix.get_matrix_metrics()
        metrics = _format_rfdetr_results(class_metrics, mean_ap, mean_f1, mean_accuracy, matrix)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to compute confusion matrix or metrics",
                "message": str(e),
                "traceback": tb,
                "num_detections": len(detections_for_cm),
                "num_references": len(reference_objects),
            },
        )

    # Remove verbose per-class metrics; replace raw matrix with sparse representation
    metrics.pop("class_metrics", None)
    raw_confusion = metrics.pop("confusion_matrix", None)
    if isinstance(raw_confusion, list):
        labels = STR_CORE_CLASSES
        non_zero_pairs: List[Dict[str, Any]] = []
        labels_used: set = set()
        for i, row in enumerate(raw_confusion):
            for j, val in enumerate(row):
                count = int(val) if val else 0
                if count > 0:
                    non_zero_pairs.append({"true": labels[i], "pred": labels[j], "count": count})
                    labels_used.add(labels[i])
                    labels_used.add(labels[j])
        non_zero_pairs.sort(key=lambda x: (x["true"] != x["pred"], -x["count"]))
        metrics["confusion_pairs"] = non_zero_pairs
        metrics["confusion_labels_used"] = sorted(labels_used)

    save_warnings: List[Dict[str, Any]] = []
    try:
        _save_detections(image_path, detections_for_save)
    except Exception as e:
        tb = traceback.format_exc()
        save_warnings.append({"warning": "Failed to save results", "message": str(e), "traceback": tb})

    # create_overlay accepted for API contract compatibility; visualization not implemented for RF-DETR
    _ = create_overlay

    result: Dict[str, Any] = {"comparison_results": metrics}
    if save_warnings:
        result["warnings"] = save_warnings
    return result


# ---------------------------------------------------------------------------
# Shared upload/parse helpers (unchanged from original)
# ---------------------------------------------------------------------------
def _read_upload_to_disk(upload: UploadFile, directory: str,
                          target_filename: Optional[str] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    filename = target_filename or upload.filename or "upload.bin"
    filename = os.path.basename(filename)
    path = os.path.join(directory, filename)
    with open(path, "wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return path


def _parse_reference_json_bytes(data: bytes) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid reference JSON", "message": str(e), "traceback": tb},
        )

    if isinstance(payload, dict) and "objects" in payload:
        return payload["objects"]
    if isinstance(payload, list):
        return payload
    raise HTTPException(
        status_code=400,
        detail={
            "error": "Unsupported reference JSON format",
            "message": "Expected list or { 'objects': [...] }",
            "received_type": type(payload).__name__,
            "sample": str(payload)[:200] if payload else None,
        },
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": _detector_instance is not None,
        "device": _detector_instance.device if _detector_instance else None,
        "api_version": app.version,
    })


@app.post("/warmup")
async def warmup_model(use_gpu: bool = Form(True)):
    global _detector_instance

    if not LOCAL_IMPORT_OK:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Failed to import local project modules",
                "message": f"Could not import RF-DETR dependencies: {LOCAL_IMPORT_ERROR_MSG}",
                "hint": "Ensure rfdetr is installed and ai_dev/ is present.",
                "traceback": None,
            },
        )

    if _detector_instance is not None:
        return JSONResponse(content={
            "status": "already_loaded",
            "timestamp": time.time(),
            "message": "Model was already loaded",
            "device": _detector_instance.device,
        })

    started = time.time()
    try:
        _detector_instance = RFDETRDetector(use_gpu=use_gpu)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=503,
            detail={"error": "Failed to load RF-DETR model", "message": str(e), "traceback": tb},
        )

    elapsed = time.time() - started
    return JSONResponse(content={
        "status": "ok",
        "timestamp": time.time(),
        "message": "Model loaded successfully",
        "load_time_seconds": elapsed,
        "device": _detector_instance.device,
        "config": {
            "box_threshold": None,
            "text_threshold": None,
            "confidence_threshold": _detector_instance.global_threshold,
            "per_class_thresholds_loaded": True,
            "num_classes": _detector_instance.num_classes,
        },
    })


@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(..., description="Image file to analyze"),
    classes: Optional[str] = Form(None, description="Unused; RF-DETR uses a fixed class list"),
    use_gpu: bool = Form(True),
    box_threshold: Optional[float] = Form(None, description="Unused; RF-DETR uses per-class thresholds"),
    text_threshold: Optional[float] = Form(None, description="Unused; RF-DETR does not use text queries"),
    confidence_threshold: Optional[float] = Form(None, description="Optional global threshold override"),
):
    global _detector_instance, _request_counter

    _request_counter += 1
    if _request_counter % 6 == 0:
        _log_nvidia_smi_stats()

    started = time.time()
    image_path = None

    try:
        if not LOCAL_IMPORT_OK:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Failed to import local project modules",
                    "message": f"Could not import RF-DETR dependencies: {LOCAL_IMPORT_ERROR_MSG}",
                    "hint": "Ensure rfdetr is installed and ai_dev/ is present.",
                    "traceback": None,
                },
            )

        if _detector_instance is None:
            try:
                _detector_instance = RFDETRDetector(use_gpu=use_gpu)
            except Exception as e:
                tb = traceback.format_exc()
                raise HTTPException(
                    status_code=503,
                    detail={"error": "Failed to load RF-DETR model", "message": str(e), "traceback": tb},
                )

        try:
            image_path = _read_upload_to_disk(image, "uploads")
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to save uploaded image", "message": str(e), "traceback": tb,
                        "filename": getattr(image, "filename", None)},
            )

        try:
            img = Image.open(image_path).convert("RGB")
            w, h = img.size
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=422,
                detail={"error": "Failed to load image", "message": str(e),
                        "hint": "Ensure the file is a valid image format (JPEG, PNG, etc.) and is not corrupted.",
                        "traceback": tb, "image_path": image_path},
            )

        try:
            detections = _detector_instance.detect(
                image_path,
                confidence_threshold_override=confidence_threshold,
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail={"error": "Model prediction failed", "message": str(e), "traceback": tb,
                        "device": _detector_instance.device},
            )

        elapsed = time.time() - started
        return JSONResponse(content={
            "status": "ok",
            "timestamp": time.time(),
            "image": os.path.basename(image_path),
            "objects": detections,
            "execution_time_seconds": elapsed,
            "image_dimensions": {"width": w, "height": h},
            "num_detections": len(detections),
            "config": {
                "box_threshold": None,
                "text_threshold": None,
                "confidence_threshold": confidence_threshold,
                "per_class_thresholds_applied": confidence_threshold is None,
                "classes_queried": len(STR_CORE_CLASSES),
                "device": _detector_instance.device,
            },
        })

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        elapsed = time.time() - started
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected error during detection", "message": str(e), "traceback": tb,
                    "execution_time_seconds": elapsed, "image_path": image_path},
        )


@app.post("/run")
async def run_model(
    image: UploadFile = File(..., description="Image file to analyze"),
    reference_json: UploadFile = File(..., description="Reference JSON with ground truth annotations"),
    use_gpu: bool = Form(True),
    create_overlay: bool = Form(True),
):
    global _request_counter

    _request_counter += 1
    if _request_counter % 6 == 0:
        _log_nvidia_smi_stats()

    started = time.time()
    image_path = None

    try:
        class _Sampler:
            def __init__(self, interval_s: float = 0.5):
                self.interval_s = interval_s
                self._stop = threading.Event()
                self.samples: List[Dict[str, Any]] = []
                self.thread: Optional[threading.Thread] = None

            def start(self):
                def _loop():
                    while not self._stop.is_set():
                        try:
                            self.samples.append(_collect_system_metrics())
                        except Exception:
                            pass
                        self._stop.wait(self.interval_s)
                self.thread = threading.Thread(target=_loop, daemon=True)
                self.thread.start()

            def stop_and_aggregate(self) -> Dict[str, Any]:
                self._stop.set()
                if self.thread:
                    self.thread.join(timeout=2.0)
                if not self.samples:
                    return {}

                def _avg(vals):
                    nums = [v for v in vals if isinstance(v, (int, float))]
                    return (sum(nums) / len(nums)) if nums else None

                gpu_utils: List[Optional[float]] = []
                gpu_mems: List[Optional[float]] = []
                for s in self.samples:
                    g0 = None
                    try:
                        g0 = s.get("gpus", [])[0] if isinstance(s.get("gpus"), list) and s.get("gpus") else None
                    except Exception:
                        pass
                    if isinstance(g0, dict):
                        gpu_utils.append(g0.get("utilization_gpu_percent"))
                        gpu_mems.append(g0.get("memory_percent"))
                    else:
                        gpu_utils.append(None)
                        gpu_mems.append(None)

                return {
                    "cpu_percent_avg": _avg([s.get("cpu_percent") for s in self.samples]),
                    "memory_percent_avg": _avg([
                        s.get("memory", {}).get("percent")
                        if isinstance(s.get("memory"), dict) else None
                        for s in self.samples
                    ]),
                    "gpu_utilization_percent_avg": _avg(gpu_utils),
                    "gpu_memory_percent_avg": _avg(gpu_mems),
                    "num_samples": len(self.samples),
                    "sample_interval_seconds": self.interval_s,
                }

        sampler = _Sampler(interval_s=0.5)
        sampler.start()

        try:
            image_path = _read_upload_to_disk(image, "uploads")
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to save uploaded image", "message": str(e), "traceback": tb,
                        "filename": getattr(image, "filename", None)},
            )

        try:
            ref_bytes = await reference_json.read()
            reference_objects = _parse_reference_json_bytes(ref_bytes)
        except HTTPException:
            raise
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=400,
                detail={"error": "Failed to read reference JSON", "message": str(e), "traceback": tb,
                        "filename": getattr(reference_json, "filename", None)},
            )

        result = _run_rfdetr(image_path, reference_objects, use_gpu=use_gpu, create_overlay=create_overlay)

        elapsed = time.time() - started
        usage_avg = sampler.stop_and_aggregate()
        device_info = _detector_instance.device if _detector_instance else ("cuda" if use_gpu else "cpu")

        response = {
            "status": "ok",
            "timestamp": time.time(),
            "execution_time_seconds": elapsed,
            "comparison_results": result.get("comparison_results"),
            "image": os.path.basename(image_path) if image_path else None,
            "image_path": image_path,
            "num_reference_objects": len(reference_objects),
            "config": {
                "use_gpu": use_gpu,
                "create_overlay": create_overlay,
                "device": device_info,
            },
            "resource_usage_avg": usage_avg,
        }

        if "warnings" in result:
            response["warnings"] = result["warnings"]

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        elapsed = time.time() - started
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected error during execution", "message": str(e), "traceback": tb,
                    "execution_time_seconds": elapsed, "image_path": image_path},
        )


# ---------------------------------------------------------------------------
# System metrics helpers (unchanged from original)
# ---------------------------------------------------------------------------
def _log_nvidia_smi_stats():
    global _request_counter
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=2.0,
        )
        lines = out.decode("utf-8", errors="ignore").strip().splitlines()
        if lines:
            logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}):")
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    uuid, name, util_gpu, mem_used, mem_total = parts[:5]
                    try:
                        mem_pct = float(mem_used) / float(mem_total) * 100.0 if float(mem_total) > 0 else 0.0
                        logger.info(f"  GPU {idx}: {name} | UUID: {uuid} | Utilization: {util_gpu}% | "
                                    f"Memory: {mem_used}/{mem_total} MB ({mem_pct:.1f}%)")
                    except Exception:
                        logger.info(f"  GPU {idx}: {name} | UUID: {uuid} | Data: {line}")
        else:
            logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}): No GPU data returned")
    except Exception:
        logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}): nvidia-smi not available or no GPUs detected")


def _collect_system_metrics() -> Dict[str, Any]:
    try:
        import psutil
    except Exception:
        psutil = None

    cpu_percent = memory = uptime_seconds = None
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            memory = {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
                      "used_bytes": int(vm.used), "percent": float(vm.percent)}
        except Exception:
            pass
        try:
            uptime_seconds = max(0.0, time.time() - float(psutil.boot_time()))
        except Exception:
            pass

    gpus: List[Dict[str, Any]] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=2.0,
        )
        for line in out.decode("utf-8", errors="ignore").strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                uuid, name, util_gpu, mem_used, mem_total = parts[:5]
                try: util_gpu = float(util_gpu)
                except Exception: util_gpu = None
                try: mem_used = float(mem_used)
                except Exception: mem_used = None
                try: mem_total = float(mem_total)
                except Exception: mem_total = None
                mem_pct = (mem_used / mem_total * 100.0) if (mem_used and mem_total) else None
                gpus.append({"uuid": uuid, "name": name, "utilization_gpu_percent": util_gpu,
                             "memory_used_mb": mem_used, "memory_total_mb": mem_total,
                             "memory_percent": mem_pct})
    except Exception:
        pass

    return {"cpu_percent": cpu_percent, "memory": memory,
            "uptime_seconds": uptime_seconds, "gpus": gpus}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="RF-DETR API Server")
    parser.add_argument("--local", "-l", action="store_true",
                        help="Run in local development mode (enables hot reload, localhost only)")
    args = parser.parse_args()

    if args.local:
        host = "127.0.0.1"
        port = int(os.environ.get("PORT", "8080"))
        logger.info("Starting in LOCAL development mode (reload enabled)")
        uvicorn.run("dinoAPI:app", host=host, port=port, reload=True)
    else:
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8080"))
        uvicorn.run("dinoAPI:app", host=host, port=port, reload=False)
