import os
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

try:
    # Local project imports only; external runtime deps are imported lazily
    from grounding_dino import GroundingDINODetector, ConfusionMatrix, format_results
    from COCO_CLASSES import INDOOR_BUSINESS_CLASSES
    LOCAL_IMPORT_OK = True
except Exception as e:
    LOCAL_IMPORT_OK = False
    LOCAL_IMPORT_ERROR_MSG = str(e)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Grounding DINO API",
              version="1.0.0",
              description="API to run Grounding DINO evaluation with provided image and reference JSON")

# Open CORS for testing; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance (initialized by warmup endpoint)
_detector_instance: Optional[Any] = None

# Counter for logging nvidia-smi stats
_request_counter: int = 0


def _read_upload_to_disk(upload: UploadFile, directory: str, target_filename: Optional[str] = None) -> str:
    os.makedirs(directory, exist_ok=True)
    filename = target_filename or upload.filename or "upload.bin"
    # Normalize filename (avoid path traversal)
    filename = os.path.basename(filename)
    path = os.path.join(directory, filename)
    with open(path, "wb") as out:
        # Stream to avoid loading into memory
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
            detail={
                "error": "Invalid reference JSON",
                "message": str(e),
                "traceback": tb
            }
        )

    # Normalize supported formats to list of objects
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
            "sample": str(payload)[:200] if payload else None
        }
    )


def _run_grounding_dino(image_path: str, reference_objects: List[Dict[str, Any]], use_gpu: bool, create_overlay: bool) -> Dict[str, Any]:
    global _detector_instance
    
    if not LOCAL_IMPORT_OK:
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to import local modules",
                "message": LOCAL_IMPORT_ERROR_MSG,
                "traceback": None
            }
        )

    # Use global detector instance if available, otherwise create one
    if _detector_instance is not None:
        detector = _detector_instance
    else:
        try:
            device = "auto" if use_gpu else "cpu"
            detector = GroundingDINODetector(device=device)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to create GroundingDINODetector",
                    "message": str(e),
                    "traceback": tb,
                    "device": "auto" if use_gpu else "cpu"
                }
            )

        if detector.model is None:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Grounding DINO model not loaded",
                    "message": "Check weights/config paths",
                    "detector_device": str(detector.device) if hasattr(detector, 'device') else None
                }
            )

    # Import external GroundingDINO utilities lazily to provide clearer errors
    try:
        from groundingdino.util.inference import load_image, predict
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Missing GroundingDINO runtime dependency",
                "message": (
                    "Install it first, e.g.: "
                    "pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git' "
                    "and ensure torch/torchvision are installed with CUDA if using GPU."
                ),
                "original_error": str(e),
                "traceback": tb
            }
        )

    # Run prediction (replicates detect_objects_in_image and detector.detect_objects logic)
    try:
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to load image",
                "message": str(e),
                "traceback": tb,
                "image_path": image_path
            }
        )

    text_query = ". ".join(INDOOR_BUSINESS_CLASSES) + "."

    # Use the detector's actual device if using pre-warmed instance
    inference_device = str(detector.device) if hasattr(detector, 'device') else ("cuda" if use_gpu else "cpu")
    try:
        boxes, confidences, labels = predict(
            model=detector.model,
            image=image,
            caption=text_query,
            box_threshold=detector.box_threshold,
            text_threshold=detector.text_threshold,
            device=inference_device,
        )
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Prediction failed",
                "message": str(e),
                "traceback": tb,
                "text_query": text_query,
                "box_threshold": detector.box_threshold,
                "text_threshold": detector.text_threshold,
                "device": ("cpu" if not use_gpu or str(detector.device) == "cpu" else "cuda")
            }
        )

    detections_for_cm: List[Dict[str, Any]] = []  # [{'class', 'bbox', 'confidence'}]
    detections_for_save: List[Dict[str, Any]] = []  # [{'object', 'bbox', 'confidence'}]

    for box, confidence, label in zip(boxes, confidences, labels):
        if confidence < detector.confidence_threshold:
            continue

        cx_norm, cy_norm, w_norm, h_norm = box
        cx = cx_norm * w
        cy = cy_norm * h
        box_w = w_norm * w
        box_h = h_norm * h

        x1 = int(cx - box_w / 2)
        y1 = int(cy - box_h / 2)
        x2 = int(cx + box_w / 2)
        y2 = int(cy + box_h / 2)

        # Clamp to bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = label.strip().lower()
        if class_name not in INDOOR_BUSINESS_CLASSES:
            continue

        detections_for_cm.append({
            "class": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confidence),
        })
        detections_for_save.append({
            "object": class_name,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(confidence),
        })

    # Build confusion matrix from provided reference JSON
    try:
        matrix = ConfusionMatrix(reference_objects)
        for det in detections_for_cm:
            matrix.handle_object_data(det["class"], det["bbox"])

        class_metrics, mean_ap, mean_f1, mean_accuracy = matrix.get_matrix_metrics()
        metrics = format_results(class_metrics, mean_ap, mean_f1, mean_accuracy, matrix)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to compute confusion matrix or metrics",
                "message": str(e),
                "traceback": tb,
                "num_detections": len(detections_for_cm),
                "num_references": len(reference_objects)
            }
        )
    # Remove verbose per-class metrics from API response
    metrics.pop("class_metrics", None)
    # Replace raw confusion matrix with a readable sparse representation
    raw_confusion = metrics.pop("confusion_matrix", None)
    if isinstance(raw_confusion, list):
        labels = INDOOR_BUSINESS_CLASSES
        non_zero_pairs: List[Dict[str, Any]] = []
        labels_used = set()
        for i, row in enumerate(raw_confusion):
            for j, val in enumerate(row):
                try:
                    count = int(val)
                except Exception:
                    # Fallback if values are floats
                    count = int(val) if val else 0
                if count > 0:
                    non_zero_pairs.append({
                        "true": labels[i],
                        "pred": labels[j],
                        "count": count,
                    })
                    labels_used.add(labels[i])
                    labels_used.add(labels[j])
        # Sort: diagonals first (TP), then by count desc
        non_zero_pairs.sort(key=lambda x: (x["true"] != x["pred"], -x["count"]))
        metrics["confusion_pairs"] = non_zero_pairs
        metrics["confusion_labels_used"] = sorted(labels_used)

    # Persist outputs (side-effects) but don't include in response
    save_warnings = []
    try:
        detector.save_results(image_path, detections_for_save)
    except Exception as e:
        tb = traceback.format_exc()
        save_warnings.append({
            "warning": "Failed to save results",
            "message": str(e),
            "traceback": tb
        })
    
    try:
        if create_overlay:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            comparison_path = os.path.join("outputs", f"dino_{base_name}_comparison.jpg")
            detector.create_comparison_visualization(image_path, detections_for_save, matrix, comparison_path)
    except Exception as e:
        tb = traceback.format_exc()
        save_warnings.append({
            "warning": "Failed to create comparison visualization",
            "message": str(e),
            "traceback": tb
        })

    result = {
        "comparison_results": metrics,
    }
    if save_warnings:
        result["warnings"] = save_warnings
    return result


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": _detector_instance is not None,
        "device": str(_detector_instance.device) if _detector_instance and hasattr(_detector_instance, 'device') else None,
        "api_version": app.version
    })


@app.post("/warmup")
async def warmup_model(use_gpu: bool = Form(True)):
    """
    Warmup endpoint to pre-load the Grounding DINO model.
    Call this once at startup to avoid loading delays on the first detection request.
    """
    global _detector_instance
    
    if not LOCAL_IMPORT_OK:
        raise HTTPException(
            status_code=503,  # Service unavailable - dependencies not ready
            detail={
                "error": "Failed to import local modules",
                "message": LOCAL_IMPORT_ERROR_MSG,
                "traceback": None
            }
        )
    
    # Check if model is already loaded
    if _detector_instance is not None:
        return JSONResponse(content={
            "status": "already_loaded",
            "timestamp": time.time(),
            "message": "Model was already loaded",
            "device": str(_detector_instance.device) if hasattr(_detector_instance, 'device') else "unknown"
        })
    
    # Load the model
    started = time.time()
    try:
        device = "auto" if use_gpu else "cpu"
        _detector_instance = GroundingDINODetector(device=device)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Failed to create GroundingDINODetector",
                "message": str(e),
                "traceback": tb,
                "device": "auto" if use_gpu else "cpu"
            }
        )
    
    if _detector_instance.model is None:
        _detector_instance = None  # Reset on failure
        raise HTTPException(
            status_code=503,  # Service unavailable - model failed to load
            detail={
                "error": "Grounding DINO model not loaded",
                "message": "Check weights/config paths",
                "detector_device": str(_detector_instance.device) if _detector_instance and hasattr(_detector_instance, 'device') else None
            }
        )
    
    elapsed = time.time() - started
    
    return JSONResponse(content={
        "status": "ok",
        "timestamp": time.time(),
        "message": "Model loaded successfully",
        "load_time_seconds": elapsed,
        "device": str(_detector_instance.device) if hasattr(_detector_instance, 'device') else "unknown",
        "config": {
            "box_threshold": _detector_instance.box_threshold if hasattr(_detector_instance, 'box_threshold') else None,
            "text_threshold": _detector_instance.text_threshold if hasattr(_detector_instance, 'text_threshold') else None,
            "confidence_threshold": _detector_instance.confidence_threshold if hasattr(_detector_instance, 'confidence_threshold') else None
        }
    })


@app.post("/detect")
async def detect_objects(
    image: UploadFile = File(..., description="Image file to analyze"),
    classes: Optional[str] = Form(None, description="Comma-separated list of classes to detect (optional)"),
    use_gpu: bool = Form(True),
    box_threshold: Optional[float] = Form(None, description="Box confidence threshold (optional)"),
    text_threshold: Optional[float] = Form(None, description="Text confidence threshold (optional)"),
    confidence_threshold: Optional[float] = Form(None, description="Minimum confidence for final detections (optional)"),
):
    """
    Detect objects in a single image and return bounding boxes.
    This endpoint does not require reference JSON and is designed for annotation workflows.
    """
    global _detector_instance, _request_counter
    
    # Increment request counter and log nvidia-smi stats every 6 requests
    _request_counter += 1
    if _request_counter % 6 == 0:
        _log_nvidia_smi_stats()
    
    started = time.time()
    image_path = None
    
    try:
        if not LOCAL_IMPORT_OK:
            raise HTTPException(
                status_code=503,  # Service unavailable - dependencies not ready
                detail={
                    "error": "Failed to import local modules",
                    "message": LOCAL_IMPORT_ERROR_MSG,
                    "traceback": None
                }
            )
        
        # Use global detector instance if available, otherwise create one
        if _detector_instance is not None:
            detector = _detector_instance
        else:
            try:
                device = "auto" if use_gpu else "cpu"
                detector = GroundingDINODetector(device=device)
            except Exception as e:
                tb = traceback.format_exc()
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "error": "Failed to create GroundingDINODetector",
                        "message": str(e),
                        "traceback": tb,
                        "device": "auto" if use_gpu else "cpu"
                    }
                )

            if detector.model is None:
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "error": "Grounding DINO model not loaded",
                        "message": "Check weights/config paths",
                        "detector_device": str(detector.device) if hasattr(detector, 'device') else None
                    }
                )
        
        # Import external GroundingDINO utilities
        try:
            from groundingdino.util.inference import load_image, predict
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Missing GroundingDINO runtime dependency",
                    "message": str(e),
                    "traceback": tb
                }
            )
        
        # Save uploaded image
        uploads_dir = os.path.join("uploads")
        try:
            image_path = _read_upload_to_disk(image, uploads_dir)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to save uploaded image",
                    "message": str(e),
                    "traceback": tb,
                    "filename": getattr(image, 'filename', None)
                }
            )
        
        # Load image
        try:
            image_source, image_tensor = load_image(image_path)
            h, w, _ = image_source.shape
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to load image",
                    "message": str(e),
                    "traceback": tb,
                    "image_path": image_path
                }
            )
        
        # Determine which classes to detect
        if classes:
            # Use provided classes (comma-separated)
            class_list = [c.strip().lower() for c in classes.split(",") if c.strip()]
        else:
            # Use default INDOOR_BUSINESS_CLASSES
            class_list = INDOOR_BUSINESS_CLASSES
        
        text_query = ". ".join(class_list) + "."
        
        # Override thresholds if provided
        box_thresh = box_threshold if box_threshold is not None else detector.box_threshold
        txt_thresh = text_threshold if text_threshold is not None else detector.text_threshold
        conf_thresh = confidence_threshold if confidence_threshold is not None else detector.confidence_threshold
        
        # Run detection
        # Use the detector's actual device if using pre-warmed instance
        inference_device = str(detector.device) if hasattr(detector, 'device') else ("cuda" if use_gpu else "cpu")
        try:
            boxes, confidences, labels = predict(
                model=detector.model,
                image=image_tensor,
                caption=text_query,
                box_threshold=box_thresh,
                text_threshold=txt_thresh,
                device=inference_device,
            )
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Prediction failed",
                    "message": str(e),
                    "traceback": tb,
                    "text_query": text_query,
                    "box_threshold": box_thresh,
                    "text_threshold": txt_thresh,
                    "device": ("cpu" if not use_gpu or str(detector.device) == "cpu" else "cuda")
                }
            )
        
        # Convert detections to annotation format
        detections = []
        for box, confidence, label in zip(boxes, confidences, labels):
            if confidence < conf_thresh:
                continue

            cx_norm, cy_norm, w_norm, h_norm = box
            cx = cx_norm * w
            cy = cy_norm * h
            box_w = w_norm * w
            box_h = h_norm * h

            x1 = int(cx - box_w / 2)
            y1 = int(cy - box_h / 2)
            x2 = int(cx + box_w / 2)
            y2 = int(cy + box_h / 2)

            # Clamp to bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            class_name = label.strip().lower()
            
            # Only include if in requested class list
            if class_name in class_list:
                detections.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(confidence),
                })
        
        elapsed = time.time() - started
        
        response = {
            "status": "ok",
            "timestamp": time.time(),
            "image": os.path.basename(image_path),
            "objects": detections,
            "execution_time_seconds": elapsed,
            "image_dimensions": {"width": w, "height": h},
            "num_detections": len(detections),
            "config": {
                "box_threshold": box_thresh,
                "text_threshold": txt_thresh,
                "confidence_threshold": conf_thresh,
                "classes_queried": len(class_list),
                "device": str(detector.device) if hasattr(detector, 'device') else ("cuda" if use_gpu else "cpu")
            }
        }
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        elapsed = time.time() - started
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Unexpected error during detection",
                "message": str(e),
                "traceback": tb,
                "execution_time_seconds": elapsed,
                "image_path": image_path
            }
        )


@app.post("/run")
async def run_model(
    image: UploadFile = File(..., description="Image file to analyze"),
    reference_json: UploadFile = File(..., description="Reference JSON with ground truth annotations"),
    use_gpu: bool = Form(True),
    create_overlay: bool = Form(True),
):
    global _request_counter
    
    # Increment request counter and log nvidia-smi stats every 6 requests
    _request_counter += 1
    if _request_counter % 6 == 0:
        _log_nvidia_smi_stats()
    
    started = time.time()
    image_path = None
    
    try:
        # Lightweight sampler to capture average system utilization during execution
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
                # Aggregate numeric fields
                def _avg(vals: List[Optional[float]]) -> Optional[float]:
                    nums = [v for v in vals if isinstance(v, (int, float))]
                    return (sum(nums) / len(nums)) if nums else None
                cpu_avg = _avg([s.get("cpu_percent") for s in self.samples])
                mem_pct_avg = _avg([s.get("memory", {}).get("percent") if isinstance(s.get("memory"), dict) else None for s in self.samples])
                gpu_util_avg = None
                gpu_mem_pct_avg = None
                # If multiple GPUs, average first GPU across samples
                gpu_utils: List[Optional[float]] = []
                gpu_mems: List[Optional[float]] = []
                for s in self.samples:
                    g0 = None
                    try:
                        g0 = s.get("gpus", [])[0] if isinstance(s.get("gpus"), list) and s.get("gpus") else None
                    except Exception:
                        g0 = None
                    if isinstance(g0, dict):
                        gpu_utils.append(g0.get("utilization_gpu_percent"))
                        gpu_mems.append(g0.get("memory_percent"))
                    else:
                        gpu_utils.append(None)
                        gpu_mems.append(None)
                gpu_util_avg = _avg(gpu_utils)
                gpu_mem_pct_avg = _avg(gpu_mems)
                return {
                    "cpu_percent_avg": cpu_avg,
                    "memory_percent_avg": mem_pct_avg,
                    "gpu_utilization_percent_avg": gpu_util_avg,
                    "gpu_memory_percent_avg": gpu_mem_pct_avg,
                    "num_samples": len(self.samples),
                    "sample_interval_seconds": self.interval_s,
                }

        sampler = _Sampler(interval_s=0.5)
        sampler.start()
        # Persist uploads
        uploads_dir = os.path.join("uploads")
        try:
            image_path = _read_upload_to_disk(image, uploads_dir)
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to save uploaded image",
                    "message": str(e),
                    "traceback": tb,
                    "filename": getattr(image, 'filename', None)
                }
            )

        try:
            ref_bytes = await reference_json.read()
            reference_objects = _parse_reference_json_bytes(ref_bytes)
        except HTTPException:
            raise  # Re-raise HTTPException with details from _parse_reference_json_bytes
        except Exception as e:
            tb = traceback.format_exc()
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "Failed to read reference JSON",
                    "message": str(e),
                    "traceback": tb,
                    "filename": getattr(reference_json, 'filename', None)
                }
            )

        # Execute detection and evaluation
        result = _run_grounding_dino(image_path, reference_objects, use_gpu=use_gpu, create_overlay=create_overlay)
        
        elapsed = time.time() - started
        usage_avg = sampler.stop_and_aggregate()
        
        # Get device info from detector
        device_info = str(_detector_instance.device) if _detector_instance and hasattr(_detector_instance, 'device') else ("cuda" if use_gpu else "cpu")
        
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
                "device": device_info
            },
            "resource_usage_avg": usage_avg
        }
        
        if "warnings" in result:
            response["warnings"] = result["warnings"]
        
        return JSONResponse(content=response)
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is (they already have detailed error info)
        raise
    except Exception as e:
        # Catch any unexpected errors
        tb = traceback.format_exc()
        elapsed = time.time() - started
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Unexpected error during execution",
                "message": str(e),
                "traceback": tb,
                "execution_time_seconds": elapsed,
                "image_path": image_path
            }
        )


def _log_nvidia_smi_stats():
    """
    Logs nvidia-smi GPU statistics.
    Called every 6 requests from main endpoints.
    """
    global _request_counter
    
    try:
        # Query nvidia-smi
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.STDOUT, timeout=2.0)
        lines = out.decode("utf-8", errors="ignore").strip().splitlines()
        
        if lines:
            logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}):")
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    uuid, name, util_gpu, mem_used, mem_total = parts[:5]
                    try:
                        util_gpu_val = float(util_gpu)
                        mem_used_val = float(mem_used)
                        mem_total_val = float(mem_total)
                        mem_percent = (mem_used_val / mem_total_val) * 100.0 if mem_total_val > 0 else 0.0
                        
                        logger.info(
                            f"  GPU {idx}: {name} | UUID: {uuid} | "
                            f"Utilization: {util_gpu_val}% | "
                            f"Memory: {mem_used_val}/{mem_total_val} MB ({mem_percent:.1f}%)"
                        )
                    except Exception:
                        logger.info(f"  GPU {idx}: {name} | UUID: {uuid} | Data: {line}")
        else:
            logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}): No GPU data returned")
    except Exception:
        # GPU info not available
        logger.info(f"NVIDIA-SMI Stats (Request #{_request_counter}): nvidia-smi not available or no GPUs detected")


def _collect_system_metrics() -> Dict[str, Any]:
    try:
        import psutil  # installed in container
    except Exception:
        psutil = None

    cpu_percent = None
    memory = None
    uptime_seconds = None
    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
        except Exception:
            cpu_percent = None
        try:
            vm = psutil.virtual_memory()
            memory = {
                "total_bytes": int(vm.total),
                "available_bytes": int(vm.available),
                "used_bytes": int(vm.used),
                "percent": float(vm.percent),
            }
        except Exception:
            memory = None
        try:
            now = time.time()
            uptime_seconds = max(0.0, now - float(psutil.boot_time()))
        except Exception:
            uptime_seconds = None

    gpus: List[Dict[str, Any]] = []
    try:
        # Query nvidia-smi; available when running with NVIDIA runtime
        # Returns per-GPU rows like: "45, 1024, 16384"
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=uuid,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.STDOUT, timeout=2.0)
        lines = out.decode("utf-8", errors="ignore").strip().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                uuid, name, util_gpu, mem_used, mem_total = parts[:5]
                try:
                    util_gpu = float(util_gpu)
                except Exception:
                    util_gpu = None
                try:
                    mem_used = float(mem_used)
                except Exception:
                    mem_used = None
                try:
                    mem_total = float(mem_total)
                except Exception:
                    mem_total = None
                mem_percent = None
                if mem_used is not None and mem_total:
                    try:
                        mem_percent = (mem_used / mem_total) * 100.0
                    except Exception:
                        mem_percent = None
                gpus.append({
                    "uuid": uuid,
                    "name": name,
                    "utilization_gpu_percent": util_gpu,
                    "memory_used_mb": mem_used,
                    "memory_total_mb": mem_total,
                    "memory_percent": mem_percent
                })
    except Exception:
        # GPU info not available; leave empty
        gpus = []

    return {
        "cpu_percent": cpu_percent,
        "memory": memory,
        "uptime_seconds": uptime_seconds,
        "gpus": gpus
    }


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Grounding DINO API Server")
    parser.add_argument(
        "--local", "-l",
        action="store_true",
        help="Run in local development mode (enables hot reload, localhost only)"
    )
    args = parser.parse_args()

    if args.local:
        # Local development: localhost only, hot reload enabled
        host = "127.0.0.1"
        port = int(os.environ.get("PORT", "8080"))
        logger.info("Starting in LOCAL development mode (reload enabled)")
        uvicorn.run("dinoAPI:app", host=host, port=port, reload=True)
    else:
        # Production mode
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8080"))
        uvicorn.run("dinoAPI:app", host=host, port=port, reload=False)


