"""
Service Test Script for Grounding DINO API

Test Flow:
==========
1. Warmup Phase (Cold Start):
   - Calls POST /warmup endpoint to pre-load the model
   - Times the model loading process
   - Model is cached globally in the API server

2. Sequential Testing Phase:
   - Calls POST /run endpoint for each test image
   - Uses the pre-loaded model (no loading overhead)
   - Times each detection request

GPU Execution Path:
===================
1. service_test.py (this file):
   - Sends use_gpu=True to /warmup and /run endpoints

2. dinoAPI.py (API endpoint):
   - /warmup: Loads model into global _detector_instance
   - /run: Reuses the pre-loaded _detector_instance
   - No redundant model loading on subsequent requests

3. grounding_dino.py (Core detector):
   - Model loaded on GPU/CPU based on use_gpu parameter
   - Cached globally for reuse across requests

Result: Fast inference after warmup, model loaded once per server lifecycle.
"""

import os
import requests
import time
from pathlib import Path

# Configuration
IP = "https://ai-core-787266927042.us-central1.run.app"  # Cloud endpoint
# IP = "http://localhost:8080"  # For local testing
USE_GPU = True  # Set to False to use CPU only
CREATE_OVERLAY = False

def warmup_model(use_gpu=True):
    """Call the warmup endpoint to pre-load the model."""
    url = f"{IP}/warmup"
    
    data = {
        "use_gpu": str(use_gpu).lower(),
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, data=data, timeout=300)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            return response.json(), elapsed_time
        else:
            return {"error": response.text, "status_code": response.status_code}, elapsed_time
    except requests.exceptions.ConnectionError as e:
        elapsed_time = time.time() - start_time
        return {"error": f"Connection failed: {str(e)}", "status_code": None, "connection_error": True}, elapsed_time
    except requests.exceptions.Timeout as e:
        elapsed_time = time.time() - start_time
        return {"error": f"Request timed out: {str(e)}", "status_code": None}, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {"error": f"Unexpected error: {str(e)}", "status_code": None}, elapsed_time


def test_photo(photo_path, json_path, use_gpu=True, create_overlay=True):
    """Test a single photo with its corresponding reference JSON."""
    url = f"{IP}/run"
    
    with open(photo_path, "rb") as img_file, open(json_path, "rb") as json_file:
        files = {
            "image": (os.path.basename(photo_path), img_file, "image/jpeg"),
            "reference_json": (os.path.basename(json_path), json_file, "application/json"),
        }
        data = {
            "use_gpu": str(use_gpu).lower(),
            "create_overlay": str(create_overlay).lower(),
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, files=files, data=data, timeout=300)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                return response.json(), elapsed_time
            else:
                return {"error": response.text, "status_code": response.status_code}, elapsed_time
        except requests.exceptions.ConnectionError as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Connection failed: {str(e)}", "status_code": None, "connection_error": True}, elapsed_time
        except requests.exceptions.Timeout as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Request timed out: {str(e)}", "status_code": None}, elapsed_time
        except Exception as e:
            elapsed_time = time.time() - start_time
            return {"error": f"Unexpected error: {str(e)}", "status_code": None}, elapsed_time


def main():
    images_dir = Path("test_photos/images")
    labels_dir = Path("test_photos/labels")
    
    # Get all image files
    image_files = sorted(images_dir.glob("*"))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print("No images found in test_photos/images")
        return
    
    print(f"🌐 Testing endpoint: {IP}")
    print(f"📁 Found {len(image_files)} images to test")
    print(f"🖥️  GPU Mode: {'ENABLED ✅' if USE_GPU else 'DISABLED (CPU only)'}")
    print(f"🎨 Overlay Creation: {'ENABLED' if CREATE_OVERLAY else 'DISABLED'}")
    print("=" * 80)
    
    # Cold start: warmup the model
    print(f"\n🔥 COLD START: Warming up model...")
    print("-" * 80)
    warmup_result, warmup_elapsed = warmup_model(use_gpu=USE_GPU)
    
    if "error" in warmup_result:
        print(f"❌ Warmup failed: {warmup_result['error']}")
        if warmup_result.get('status_code'):
            print(f"Status code: {warmup_result.get('status_code')}")
        if warmup_result.get('connection_error'):
            print(f"\n💡 Tip: Make sure the API server is running.")
            print(f"   - For local: Start the server with 'python dinoAPI.py'")
            print(f"   - For cloud: Check if the endpoint URL is correct")
        return
    else:
        print(f"✅ Model loaded successfully")
        print(f"⏱️  Cold start time: {warmup_elapsed:.2f}s")
        print(f"⏱️  Server load time: {warmup_result.get('load_time_seconds', 0):.2f}s")
        print(f"🖥️  Device: {warmup_result.get('device', 'unknown')}")
        print(f"ℹ️  Status: {warmup_result.get('message', warmup_result.get('status', 'ok'))}")
    
    print("=" * 80)
    
    # Loop through all images
    print(f"\n🔄 Testing {len(image_files)} images with warmed-up model...")
    print("=" * 80)
    
    results_summary = []
    
    for idx, image_path in enumerate(image_files, start=1):
        json_path = labels_dir / f"{image_path.stem}.json"
        
        if not json_path.exists():
            print(f"\n[{idx}/{len(image_files)}] ⚠️  Skipping {image_path.name} - no matching JSON")
            continue
        
        print(f"\n[{idx}/{len(image_files)}] Testing: {image_path.name}")
        result, elapsed = test_photo(str(image_path), str(json_path), use_gpu=USE_GPU, create_overlay=CREATE_OVERLAY)
        
        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
            results_summary.append({
                "image": image_path.name,
                "status": "error",
                "elapsed": elapsed,
            })
        else:
            comp_results = result["comparison_results"]
            mean_ap = comp_results["mean_average_precision"]
            mean_f1 = comp_results["mean_f1_score"]
            unmatched = comp_results["unmatched_objects"]
            missing = comp_results["missing_objects"]
            exec_time = result["execution_time_seconds"]
            
            print(f"   ✅ AP: {mean_ap:.3f} | F1: {mean_f1:.3f} | Unmatched: {unmatched} | Missing: {missing} | Time: {exec_time:.2f}s")
            
            results_summary.append({
                "image": image_path.name,
                "status": "success",
                "request_time": elapsed,
                "execution_time": exec_time,
                "mean_ap": mean_ap,
                "mean_f1": mean_f1,
                "unmatched": unmatched,
                "missing": missing,
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results_summary if r["status"] == "success"]
    failed = [r for r in results_summary if r["status"] == "error"]
    
    print(f"Cold Start Time: {warmup_elapsed:.2f}s")
    print(f"Total tests: {len(results_summary)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_ap = sum(r["mean_ap"] for r in successful) / len(successful)
        avg_f1 = sum(r["mean_f1"] for r in successful) / len(successful)
        total_unmatched = sum(r["unmatched"] for r in successful)
        total_missing = sum(r["missing"] for r in successful)
        avg_exec_time = sum(r["execution_time"] for r in successful) / len(successful)
        avg_request_time = sum(r["request_time"] for r in successful) / len(successful)
        
        print(f"\n📊 Average Metrics (successful tests):")
        print(f"   Mean AP: {avg_ap:.3f}")
        print(f"   Mean F1: {avg_f1:.3f}")
        print(f"   Total Unmatched: {total_unmatched}")
        print(f"   Total Missing: {total_missing}")
        print(f"   Avg Execution Time: {avg_exec_time:.2f}s (server-side only)")
        print(f"   Avg Request Time: {avg_request_time:.2f}s (includes network)")


if __name__ == "__main__":
    main()