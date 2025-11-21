#!/usr/bin/env python3
"""
Image Annotation Tool with GroundingDINO Integration
-----------------------------------------------------
Features:
- Automatic first-pass detection using GroundingDINO
- Interactive bounding box drawing and editing
- Add, remove, and modify annotations
- JSON export in specified format
- Optional object_list.py for predefined classes
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Try to import GroundingDINO detector from parent directory
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from grounding_dino import GroundingDINODetector
    from COCO_CLASSES import INDOOR_BUSINESS_CLASSES
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    INDOOR_BUSINESS_CLASSES = []

# Try to import object_list.py
try:
    from object_list import OBJECT_CLASSES
    print("[OK] Loaded object classes from object_list.py")
except ImportError:
    # Use INDOOR_BUSINESS_CLASSES as fallback
    OBJECT_CLASSES = INDOOR_BUSINESS_CLASSES if INDOOR_BUSINESS_CLASSES else [
        "chair", "table", "desk", "couch", "bed", "potted plant",
        "laptop", "monitor", "keyboard", "mouse", "tv", "bottle",
        "cup", "book", "sink", "refrigerator", "microwave", "clock"
    ]
    print("[INFO] Using default object classes")


class BoundingBox:
    """Represents a bounding box annotation"""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_name: str, confidence: float = 1.0):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.class_name = class_name
        self.confidence = confidence
        self.selected = False
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside bbox"""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2
    
    def get_handle_at_point(self, x: int, y: int, threshold: int = 10) -> Optional[str]:
        """Check if point is near a resize handle"""
        handles = {
            'nw': (self.x1, self.y1),
            'ne': (self.x2, self.y1),
            'sw': (self.x1, self.y2),
            'se': (self.x2, self.y2),
            'n': ((self.x1 + self.x2) // 2, self.y1),
            's': ((self.x1 + self.x2) // 2, self.y2),
            'w': (self.x1, (self.y1 + self.y2) // 2),
            'e': (self.x2, (self.y1 + self.y2) // 2),
        }
        
        for handle, (hx, hy) in handles.items():
            if abs(x - hx) <= threshold and abs(y - hy) <= threshold:
                return handle
        
        return None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-compatible dict"""
        return {
            "class": self.class_name,
            "bbox": [self.x1, self.y1, self.x2, self.y2]
        }


class AnnotationTool:
    """Main annotation tool GUI"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Annotation Tool")
        self.root.geometry("1400x900")
        
        # State variables
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.photo_image = None
        self.bboxes: List[BoundingBox] = []
        self.current_bbox = None
        self.drawing = False
        self.dragging = False
        self.resizing = False
        self.resize_handle = None
        self.start_x = 0
        self.start_y = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.scale_factor = 1.0
        self.canvas_width = 1000
        self.canvas_height = 800
        
        # Zoom and pan variables
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        # API Configuration
        self.api_url = "https://ai-core-787266927042.us-central1.run.app"  # Default API URL
        self.use_api = True  # Toggle between local model and API
        
        # Colors
        self.bbox_color = "#00FF00"  # Green for unselected
        self.selected_color = "#FF0000"  # Red for selected
        self.drawing_color = "#FFFF00"  # Yellow for drawing
        
        # Setup UI
        self.setup_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<Delete>', self.delete_selected_bbox)
        self.root.bind('<BackSpace>', self.delete_selected_bbox)
        self.root.bind('<Escape>', self.deselect_all)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Top menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Save Annotations", command=self.save_annotations)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Run GroundingDINO (Local)", command=self.run_grounding_dino)
        edit_menu.add_command(label="Run API Detection", command=self.run_api_detection)
        edit_menu.add_command(label="Batch Process Directory (API)", command=self.batch_process_directory)
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear All Boxes", command=self.clear_all_boxes)
        edit_menu.add_command(label="Delete Selected (Del)", command=self.delete_selected_bbox)
        
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure API Endpoint", command=self.configure_api)
        settings_menu.add_command(label="Test API Connection", command=self.test_api_health)
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        view_menu.add_separator()
        view_menu.add_command(label="Resize Image to 640x480", command=self.resize_image_640x480)
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Canvas
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars
        canvas_frame = tk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                               bg='gray', cursor='cross')
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky='nsew')
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas bindings
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
        self.canvas.bind('<Motion>', self.on_mouse_move)
        
        # Right-click for panning
        self.canvas.bind('<ButtonPress-3>', self.on_pan_start)
        self.canvas.bind('<B3-Motion>', self.on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>', self.on_pan_end)
        
        # Mouse wheel for zoom
        self.canvas.bind('<MouseWheel>', self.on_mousewheel_zoom)
        
        # Bottom toolbar
        toolbar_frame = tk.Frame(left_frame)
        toolbar_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(toolbar_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Auto-Detect (Local)", command=self.run_grounding_dino,
                 bg='lightblue').pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Auto-Detect (API)", command=self.run_api_detection,
                 bg='lightyellow').pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Batch Process Dir", command=self.batch_process_directory,
                 bg='#FFD700').pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Save JSON", command=self.save_annotations,
                 bg='lightgreen').pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar_frame, text="Clear All", command=self.clear_all_boxes).pack(side=tk.LEFT, padx=2)
        
        self.status_label = tk.Label(toolbar_frame, text="No image loaded", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Zoom control
        zoom_frame = tk.Frame(left_frame)
        zoom_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=5)
        
        self.zoom_var = tk.DoubleVar(value=100.0)
        self.zoom_slider = tk.Scale(zoom_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                                    variable=self.zoom_var, command=self.on_zoom_slider_change,
                                    showvalue=False, length=200)
        self.zoom_slider.pack(side=tk.LEFT, padx=5)
        
        self.zoom_label = tk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        tk.Button(zoom_frame, text="Reset", command=self.reset_zoom, width=6).pack(side=tk.LEFT, padx=2)
        
        tk.Label(zoom_frame, text="(Right-click + drag to pan)", font=('Arial', 8), 
                fg='gray').pack(side=tk.LEFT, padx=10)
        
        # Right panel - Controls
        right_frame = tk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # Instructions
        instructions = tk.LabelFrame(right_frame, text="Instructions", padx=10, pady=10)
        instructions.pack(fill=tk.X, pady=5)
        
        instructions_text = """
• Open an image to start
• Auto-Detect: Local model or API
• Batch Process: Process entire directory
• Draw: Click and drag to create box
• Select: Click on existing box
• Move: Drag selected box
• Resize: Drag corner/edge handles
• Delete: Press Delete or Backspace
• Label: Search and select class
• Settings: Configure API endpoint
        """
        tk.Label(instructions, text=instructions_text, justify=tk.LEFT, font=('Arial', 9)).pack()
        
        # Bounding boxes list
        bbox_frame = tk.LabelFrame(right_frame, text="Bounding Boxes", padx=10, pady=10)
        bbox_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Listbox with scrollbar
        list_frame = tk.Frame(bbox_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.bbox_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        self.bbox_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.bbox_listbox.yview)
        
        self.bbox_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        
        # Class selection
        class_frame = tk.LabelFrame(right_frame, text="Object Class", padx=10, pady=10)
        class_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(class_frame, text="Assign class to selected box:").pack(anchor=tk.W)
        
        # Search field for classes
        search_frame = tk.Frame(class_frame)
        search_frame.pack(fill=tk.X, pady=(5, 2))
        
        tk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        self.class_search_var = tk.StringVar()
        self.class_search_var.trace('w', self.filter_classes)
        self.class_search_entry = tk.Entry(search_frame, textvariable=self.class_search_var)
        self.class_search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Class dropdown
        self.class_var = tk.StringVar()
        self.class_combobox = ttk.Combobox(class_frame, textvariable=self.class_var,
                                           values=sorted(OBJECT_CLASSES), state='readonly')
        self.class_combobox.pack(fill=tk.X, pady=5)
        if OBJECT_CLASSES:
            self.class_combobox.current(0)
        
        self.class_combobox.bind('<<ComboboxSelected>>', self.on_class_changed)
        self.all_classes = sorted(OBJECT_CLASSES)  # Store all classes for filtering
        
        # Quick actions
        actions_frame = tk.Frame(right_frame, padx=10, pady=5)
        actions_frame.pack(fill=tk.X)
        
        tk.Button(actions_frame, text="Delete Selected", command=self.delete_selected_bbox,
                 bg='#ffcccc').pack(fill=tk.X, pady=2)
        
        # Stats
        self.stats_label = tk.Label(right_frame, text="Boxes: 0", relief=tk.SUNKEN, pady=5)
        self.stats_label.pack(fill=tk.X, pady=5)
    
    def update_status(self, message: str):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def open_image(self):
        """Open an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path: str):
        """Load an image and display it"""
        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            
            # Clear existing annotations
            self.bboxes = []
            self.current_bbox = None
            
            # Reset zoom
            self.zoom_level = 1.0
            self.zoom_var.set(100.0)
            self.zoom_label.config(text="100%")
            
            # Calculate scale factor to fit canvas
            img_width, img_height = self.original_image.size
            scale_x = self.canvas_width / img_width
            scale_y = self.canvas_height / img_height
            self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            # Resize for display
            display_width = int(img_width * self.scale_factor)
            display_height = int(img_height * self.scale_factor)
            self.display_image = self.original_image.resize((display_width, display_height), Image.LANCZOS)
            
            # Update canvas with zoom
            zoomed_width = int(display_width * self.zoom_level)
            zoomed_height = int(display_height * self.zoom_level)
            self.canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))
            self.redraw_canvas()
            
            # Update UI
            filename = os.path.basename(file_path)
            self.update_status(f"Loaded: {filename} ({img_width}x{img_height})")
            self.root.title(f"Annotation Tool - {filename}")
            self.update_bbox_list()
            
            # Try to load existing annotations
            self.load_existing_annotations()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{str(e)}")
    
    def load_existing_annotations(self):
        """Load existing annotations if JSON file exists"""
        if not self.image_path:
            return
        
        base_name = os.path.splitext(self.image_path)[0]
        json_path = f"{base_name}.json"
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, dict) and 'objects' in data:
                    objects = data['objects']
                elif isinstance(data, list):
                    objects = data
                else:
                    return
                
                # Load bounding boxes
                for obj in objects:
                    class_name = obj.get('class', 'unknown')
                    bbox = obj.get('bbox', [])
                    if len(bbox) == 4:
                        # Scale coordinates if needed
                        x1, y1, x2, y2 = bbox
                        scaled_bbox = BoundingBox(
                            int(x1 * self.scale_factor),
                            int(y1 * self.scale_factor),
                            int(x2 * self.scale_factor),
                            int(y2 * self.scale_factor),
                            class_name
                        )
                        self.bboxes.append(scaled_bbox)
                
                self.redraw_canvas()
                self.update_bbox_list()
                self.update_status(f"Loaded {len(self.bboxes)} existing annotations")
                
            except Exception as e:
                print(f"Failed to load existing annotations: {e}")
    
    def zoom_in(self):
        """Zoom in on the image"""
        if self.display_image:
            new_zoom = min(self.zoom_level * 1.2, self.max_zoom)
            self.set_zoom(new_zoom)
    
    def zoom_out(self):
        """Zoom out on the image"""
        if self.display_image:
            new_zoom = max(self.zoom_level / 1.2, self.min_zoom)
            self.set_zoom(new_zoom)
    
    def reset_zoom(self):
        """Reset zoom to 100%"""
        if self.display_image:
            self.set_zoom(1.0)
    
    def set_zoom(self, zoom_level: float):
        """Set zoom level and update display"""
        if not self.display_image:
            return
        
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
        self.zoom_var.set(self.zoom_level * 100)
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
        
        # Update canvas scrollregion
        img_width, img_height = self.display_image.size
        zoomed_width = int(img_width * self.zoom_level)
        zoomed_height = int(img_height * self.zoom_level)
        self.canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))
        
        self.redraw_canvas()
    
    def on_zoom_slider_change(self, value):
        """Handle zoom slider change"""
        zoom_percent = float(value)
        self.set_zoom(zoom_percent / 100.0)
    
    def on_mousewheel_zoom(self, event):
        """Handle mouse wheel zoom"""
        if not self.display_image:
            return
        
        # Get mouse position relative to canvas
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Zoom in or out
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def on_pan_start(self, event):
        """Start panning with right-click"""
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor='fleur')
    
    def on_pan_drag(self, event):
        """Handle panning drag"""
        if self.panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            # Scroll the canvas
            self.canvas.xview_scroll(int(-dx / 10), 'units')
            self.canvas.yview_scroll(int(-dy / 10), 'units')
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
    
    def on_pan_end(self, event):
        """End panning"""
        self.panning = False
        self.canvas.config(cursor='cross')
    
    def resize_image_640x480(self):
        """Resize the original image to 640x480"""
        if not self.original_image:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        result = messagebox.askyesno(
            "Resize Image",
            "This will permanently resize the image to 640x480.\n\n"
            "Bounding boxes will be scaled accordingly.\n\n"
            "Continue?"
        )
        
        if not result:
            return
        
        try:
            # Store old dimensions
            old_width, old_height = self.original_image.size
            new_width, new_height = 640, 480
            
            # Calculate scaling factors for bounding boxes
            width_scale = new_width / old_width
            height_scale = new_height / old_height
            
            # Resize the original image
            self.original_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the resized image
            if self.image_path:
                self.original_image.save(self.image_path)
            
            # Scale all bounding boxes
            for bbox in self.bboxes:
                # Convert to original coordinates
                orig_x1 = bbox.x1 / self.scale_factor
                orig_y1 = bbox.y1 / self.scale_factor
                orig_x2 = bbox.x2 / self.scale_factor
                orig_y2 = bbox.y2 / self.scale_factor
                
                # Apply resize scaling
                new_x1 = orig_x1 * width_scale
                new_y1 = orig_y1 * height_scale
                new_x2 = orig_x2 * width_scale
                new_y2 = orig_y2 * height_scale
                
                # Convert back to display coordinates (will be recalculated)
                bbox.x1 = new_x1
                bbox.y1 = new_y1
                bbox.x2 = new_x2
                bbox.y2 = new_y2
            
            # Reload the image display
            self.load_image(self.image_path)
            
            # Restore bounding boxes with new scale
            temp_bboxes = self.bboxes.copy()
            self.bboxes = []
            for bbox in temp_bboxes:
                scaled_bbox = BoundingBox(
                    int(bbox.x1 * self.scale_factor),
                    int(bbox.y1 * self.scale_factor),
                    int(bbox.x2 * self.scale_factor),
                    int(bbox.y2 * self.scale_factor),
                    bbox.class_name,
                    bbox.confidence
                )
                self.bboxes.append(scaled_bbox)
            
            self.redraw_canvas()
            self.update_bbox_list()
            
            self.update_status(f"Image resized to 640x480")
            messagebox.showinfo("Success", f"Image resized to 640x480\n\n"
                              f"Original size: {old_width}x{old_height}\n"
                              f"New size: {new_width}x{new_height}\n"
                              f"Bounding boxes: {len(self.bboxes)} scaled")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to resize image:\n{str(e)}")
    
    def run_grounding_dino(self):
        """Run GroundingDINO for automatic detection"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not GROUNDING_DINO_AVAILABLE:
            messagebox.showerror("GroundingDINO Not Available",
                               "GroundingDINO is not installed or not available.\n"
                               "The tool will work in manual annotation mode only.")
            return
        
        # Ask user if they want to clear existing boxes
        if self.bboxes:
            result = messagebox.askyesnocancel(
                "Clear Existing Boxes?",
                "Do you want to clear existing annotations before running auto-detection?\n\n"
                "Yes: Clear and run detection\n"
                "No: Add to existing boxes\n"
                "Cancel: Don't run detection"
            )
            if result is None:  # Cancel
                return
            elif result:  # Yes
                self.bboxes = []
        
        self.update_status("Running GroundingDINO... Please wait...")
        self.root.update()
        
        try:
            # Create detector
            detector = GroundingDINODetector(device="auto")
            
            if detector.model is None:
                messagebox.showerror("Error", "Failed to load GroundingDINO model")
                self.update_status("GroundingDINO failed to load")
                return
            
            # Run detection directly without ground truth evaluation
            detections = self._run_detection_only(detector)
            
            # Convert detections to bounding boxes (scaled for display)
            for det in detections:
                bbox_coords = det['bbox']  # [x1, y1, x2, y2]
                class_name = det.get('object', det.get('class', 'unknown'))
                confidence = det.get('confidence', 1.0)
                
                # Scale coordinates for display
                x1 = int(bbox_coords[0] * self.scale_factor)
                y1 = int(bbox_coords[1] * self.scale_factor)
                x2 = int(bbox_coords[2] * self.scale_factor)
                y2 = int(bbox_coords[3] * self.scale_factor)
                
                bbox = BoundingBox(x1, y1, x2, y2, class_name, confidence)
                self.bboxes.append(bbox)
            
            self.redraw_canvas()
            self.update_bbox_list()
            self.update_status(f"GroundingDINO detected {len(detections)} objects")
            
        except Exception as e:
            messagebox.showerror("Error", f"GroundingDINO failed:\n{str(e)}")
            self.update_status("GroundingDINO detection failed")
    
    def _run_detection_only(self, detector):
        """Run GroundingDINO detection without ground truth evaluation"""
        try:
            from groundingdino.util.inference import load_image, predict
        except ImportError:
            raise ImportError("GroundingDINO not available")
        
        # Load image
        image_source, image = load_image(self.image_path)
        h, w, _ = image_source.shape
        
        # Use custom prompts or INDOOR_BUSINESS_CLASSES
        prompts = OBJECT_CLASSES if OBJECT_CLASSES else INDOOR_BUSINESS_CLASSES
        text_query = ". ".join(prompts) + "."
        
        # Run detection
        boxes, confidences, labels = predict(
            model=detector.model,
            image=image,
            caption=text_query,
            box_threshold=detector.box_threshold,
            text_threshold=detector.text_threshold,
            device="cpu"
        )
        
        # Convert coordinates: [cx, cy, w, h] normalized -> [x1, y1, x2, y2] pixels
        detections = []
        for box, confidence, label in zip(boxes, confidences, labels):
            if confidence >= detector.confidence_threshold:
                cx_norm, cy_norm, w_norm, h_norm = box
                
                cx = cx_norm * w
                cy = cy_norm * h
                box_w = w_norm * w
                box_h = h_norm * h
                
                x1 = int(cx - box_w / 2)
                y1 = int(cy - box_h / 2)
                x2 = int(cx + box_w / 2)
                y2 = int(cy + box_h / 2)
                
                # Clamp to image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Validate bbox
                if x2 > x1 and y2 > y1:
                    class_name = label.strip().lower()
                    # Filter to only include objects in our class list
                    if class_name in prompts or not prompts:
                        detections.append({
                            "object": class_name,
                            "confidence": float(confidence),
                            "bbox": [x1, y1, x2, y2]
                        })
        
        return detections
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        if not self.image_path:
            messagebox.showwarning("No Image", "No image loaded")
            return
        
        if not self.bboxes:
            result = messagebox.askyesno("No Annotations",
                                        "No bounding boxes to save. Save empty file?")
            if not result:
                return
        
        # Default save path
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        default_name = f"{base_name}.json"
        default_dir = os.path.dirname(self.image_path)
        
        file_path = filedialog.asksaveasfilename(
            title="Save Annotations",
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Convert bboxes to JSON format (unscale coordinates)
                objects = []
                for bbox in self.bboxes:
                    # Unscale coordinates to original image size
                    x1 = int(bbox.x1 / self.scale_factor)
                    y1 = int(bbox.y1 / self.scale_factor)
                    x2 = int(bbox.x2 / self.scale_factor)
                    y2 = int(bbox.y2 / self.scale_factor)
                    
                    objects.append({
                        "class": bbox.class_name,
                        "bbox": [x1, y1, x2, y2]
                    })
                
                # Create JSON in specified format
                output = {
                    "image": os.path.basename(self.image_path),
                    "objects": objects
                }
                
                with open(file_path, 'w') as f:
                    json.dump(output, f, indent=2)
                
                self.update_status(f"Saved {len(objects)} annotations to {os.path.basename(file_path)}")
                messagebox.showinfo("Success", f"Annotations saved successfully!\n{len(objects)} objects")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save annotations:\n{str(e)}")
    
    def clear_all_boxes(self):
        """Clear all bounding boxes"""
        if self.bboxes:
            result = messagebox.askyesno("Clear All", "Are you sure you want to clear all bounding boxes?")
            if result:
                self.bboxes = []
                self.current_bbox = None
                self.redraw_canvas()
                self.update_bbox_list()
                self.update_status("Cleared all bounding boxes")
    
    def delete_selected_bbox(self, event=None):
        """Delete the currently selected bounding box"""
        selected = [bbox for bbox in self.bboxes if bbox.selected]
        if selected:
            for bbox in selected:
                self.bboxes.remove(bbox)
            self.redraw_canvas()
            self.update_bbox_list()
            self.update_status(f"Deleted {len(selected)} box(es)")
    
    def deselect_all(self, event=None):
        """Deselect all bounding boxes"""
        for bbox in self.bboxes:
            bbox.selected = False
        self.redraw_canvas()
        self.update_bbox_list()
    
    def on_mouse_down(self, event):
        """Handle mouse button press"""
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates (accounting for zoom)
        x = canvas_x / self.zoom_level
        y = canvas_y / self.zoom_level
        
        # Check if clicking on existing bbox
        clicked_bbox = None
        clicked_handle = None
        
        for bbox in reversed(self.bboxes):  # Check from top to bottom
            handle = bbox.get_handle_at_point(x, y)
            if handle:
                clicked_bbox = bbox
                clicked_handle = handle
                break
            elif bbox.contains_point(x, y):
                clicked_bbox = bbox
                break
        
        if clicked_bbox:
            # Deselect all others
            for bbox in self.bboxes:
                bbox.selected = False
            
            clicked_bbox.selected = True
            
            if clicked_handle:
                # Start resizing
                self.resizing = True
                self.resize_handle = clicked_handle
                self.current_bbox = clicked_bbox
            else:
                # Start dragging
                self.dragging = True
                self.current_bbox = clicked_bbox
                self.drag_offset_x = x - clicked_bbox.x1
                self.drag_offset_y = y - clicked_bbox.y1
        else:
            # Start drawing new bbox
            self.deselect_all()
            self.drawing = True
            self.start_x = x
            self.start_y = y
            
            # Get current class
            class_name = self.class_var.get() or "unknown"
            self.current_bbox = BoundingBox(x, y, x, y, class_name)
        
        self.redraw_canvas()
        self.update_bbox_list()
    
    def on_mouse_drag(self, event):
        """Handle mouse drag"""
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates (accounting for zoom)
        x = canvas_x / self.zoom_level
        y = canvas_y / self.zoom_level
        
        if self.drawing and self.current_bbox:
            # Update drawing bbox
            self.current_bbox.x2 = x
            self.current_bbox.y2 = y
            self.redraw_canvas(include_current=True)
            
        elif self.dragging and self.current_bbox:
            # Move bbox
            width = self.current_bbox.x2 - self.current_bbox.x1
            height = self.current_bbox.y2 - self.current_bbox.y1
            
            self.current_bbox.x1 = x - self.drag_offset_x
            self.current_bbox.y1 = y - self.drag_offset_y
            self.current_bbox.x2 = self.current_bbox.x1 + width
            self.current_bbox.y2 = self.current_bbox.y1 + height
            
            self.redraw_canvas()
            
        elif self.resizing and self.current_bbox and self.resize_handle:
            # Resize bbox
            if 'n' in self.resize_handle:
                self.current_bbox.y1 = y
            if 's' in self.resize_handle:
                self.current_bbox.y2 = y
            if 'w' in self.resize_handle:
                self.current_bbox.x1 = x
            if 'e' in self.resize_handle:
                self.current_bbox.x2 = x
            
            # Ensure min/max are correct
            if self.current_bbox.x1 > self.current_bbox.x2:
                self.current_bbox.x1, self.current_bbox.x2 = self.current_bbox.x2, self.current_bbox.x1
            if self.current_bbox.y1 > self.current_bbox.y2:
                self.current_bbox.y1, self.current_bbox.y2 = self.current_bbox.y2, self.current_bbox.y1
            
            self.redraw_canvas()
    
    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if self.drawing and self.current_bbox:
            # Finish drawing
            canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            
            # Convert canvas coordinates to image coordinates (accounting for zoom)
            x = canvas_x / self.zoom_level
            y = canvas_y / self.zoom_level
            
            self.current_bbox.x2 = x
            self.current_bbox.y2 = y
            
            # Normalize coordinates
            if self.current_bbox.x1 > self.current_bbox.x2:
                self.current_bbox.x1, self.current_bbox.x2 = self.current_bbox.x2, self.current_bbox.x1
            if self.current_bbox.y1 > self.current_bbox.y2:
                self.current_bbox.y1, self.current_bbox.y2 = self.current_bbox.y2, self.current_bbox.y1
            
            # Only add if bbox has area
            width = self.current_bbox.x2 - self.current_bbox.x1
            height = self.current_bbox.y2 - self.current_bbox.y1
            if width > 5 and height > 5:
                self.bboxes.append(self.current_bbox)
                self.current_bbox.selected = True
            
            self.current_bbox = None
        
        self.drawing = False
        self.dragging = False
        self.resizing = False
        self.resize_handle = None
        
        self.redraw_canvas()
        self.update_bbox_list()
    
    def on_mouse_move(self, event):
        """Handle mouse movement (for cursor changes)"""
        if self.drawing or self.dragging or self.resizing or self.panning:
            return
        
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Convert canvas coordinates to image coordinates (accounting for zoom)
        x = canvas_x / self.zoom_level
        y = canvas_y / self.zoom_level
        
        # Check for resize handles
        cursor = 'cross'
        for bbox in self.bboxes:
            handle = bbox.get_handle_at_point(x, y)
            if handle:
                # Set appropriate cursor for handle
                cursor_map = {
                    'nw': 'top_left_corner',
                    'ne': 'top_right_corner',
                    'sw': 'bottom_left_corner',
                    'se': 'bottom_right_corner',
                    'n': 'top_side',
                    's': 'bottom_side',
                    'w': 'left_side',
                    'e': 'right_side',
                }
                cursor = cursor_map.get(handle, 'cross')
                break
            elif bbox.contains_point(x, y):
                cursor = 'fleur'  # Move cursor
                break
        
        self.canvas.config(cursor=cursor)
    
    def redraw_canvas(self, include_current: bool = False):
        """Redraw the canvas with image and bboxes"""
        self.canvas.delete('all')
        
        if self.display_image:
            # Apply zoom to image
            if self.zoom_level != 1.0:
                zoomed_width = int(self.display_image.width * self.zoom_level)
                zoomed_height = int(self.display_image.height * self.zoom_level)
                zoomed_image = self.display_image.resize((zoomed_width, zoomed_height), Image.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(zoomed_image)
            else:
                self.photo_image = ImageTk.PhotoImage(self.display_image)
            
            self.canvas.create_image(0, 0, image=self.photo_image, anchor=tk.NW)
            
            # Draw bounding boxes (scaled by zoom)
            for bbox in self.bboxes:
                color = self.selected_color if bbox.selected else self.bbox_color
                width = 3 if bbox.selected else 2
                
                # Apply zoom to bbox coordinates
                x1 = bbox.x1 * self.zoom_level
                y1 = bbox.y1 * self.zoom_level
                x2 = bbox.x2 * self.zoom_level
                y2 = bbox.y2 * self.zoom_level
                
                # Draw rectangle
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=color, width=width, tags='bbox'
                )
                
                # Draw label
                label_text = f"{bbox.class_name}"
                if bbox.confidence < 1.0:
                    label_text += f" ({bbox.confidence:.2f})"
                
                # Label background
                label_height = int(20 * self.zoom_level)
                label_width = int(len(label_text) * 7 * self.zoom_level)
                self.canvas.create_rectangle(
                    x1, y1 - label_height, x1 + label_width, y1,
                    fill=color, outline=color, tags='label'
                )
                self.canvas.create_text(
                    x1 + 2, y1 - label_height // 2,
                    text=label_text, anchor=tk.W, fill='white', tags='label'
                )
                
                # Draw resize handles if selected
                if bbox.selected:
                    handle_size = max(6, int(6 * self.zoom_level))
                    handles = [
                        (x1, y1),  # NW
                        (x2, y1),  # NE
                        (x1, y2),  # SW
                        (x2, y2),  # SE
                        ((x1 + x2) / 2, y1),  # N
                        ((x1 + x2) / 2, y2),  # S
                        (x1, (y1 + y2) / 2),  # W
                        (x2, (y1 + y2) / 2),  # E
                    ]
                    for hx, hy in handles:
                        self.canvas.create_rectangle(
                            hx - handle_size / 2, hy - handle_size / 2,
                            hx + handle_size / 2, hy + handle_size / 2,
                            fill='white', outline=color, width=2, tags='handle'
                        )
            
            # Draw current bbox being drawn
            if include_current and self.current_bbox and self.drawing:
                x1 = self.current_bbox.x1 * self.zoom_level
                y1 = self.current_bbox.y1 * self.zoom_level
                x2 = self.current_bbox.x2 * self.zoom_level
                y2 = self.current_bbox.y2 * self.zoom_level
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=self.drawing_color, width=2, dash=(5, 5), tags='current'
                )
        
        self.update_stats()
    
    def update_bbox_list(self):
        """Update the bounding boxes listbox"""
        self.bbox_listbox.delete(0, tk.END)
        
        for i, bbox in enumerate(self.bboxes):
            marker = "► " if bbox.selected else "  "
            text = f"{marker}{i+1}. {bbox.class_name}"
            if bbox.confidence < 1.0:
                text += f" ({bbox.confidence:.2f})"
            self.bbox_listbox.insert(tk.END, text)
            
            # Highlight selected
            if bbox.selected:
                self.bbox_listbox.itemconfig(i, bg='lightblue')
    
    def update_stats(self):
        """Update statistics label"""
        total = len(self.bboxes)
        selected = sum(1 for bbox in self.bboxes if bbox.selected)
        
        # Count by class
        class_counts = {}
        for bbox in self.bboxes:
            class_counts[bbox.class_name] = class_counts.get(bbox.class_name, 0) + 1
        
        stats_text = f"Boxes: {total}"
        if selected > 0:
            stats_text += f" | Selected: {selected}"
        if class_counts:
            top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            class_str = ", ".join([f"{c}: {n}" for c, n in top_classes])
            stats_text += f" | {class_str}"
        
        self.stats_label.config(text=stats_text)
    
    def on_listbox_select(self, event):
        """Handle listbox selection"""
        selection = self.bbox_listbox.curselection()
        if selection:
            idx = selection[0]
            # Deselect all
            for bbox in self.bboxes:
                bbox.selected = False
            # Select clicked
            if 0 <= idx < len(self.bboxes):
                self.bboxes[idx].selected = True
                self.redraw_canvas()
                self.update_bbox_list()
    
    def on_class_changed(self, event):
        """Handle class selection change"""
        new_class = self.class_var.get()
        
        # Update selected bbox
        selected = [bbox for bbox in self.bboxes if bbox.selected]
        if selected:
            for bbox in selected:
                bbox.class_name = new_class
            self.redraw_canvas()
            self.update_bbox_list()
            self.update_status(f"Changed class to: {new_class}")
    
    def filter_classes(self, *args):
        """Filter class list based on search text"""
        search_text = self.class_search_var.get().lower()
        
        if not search_text:
            # Show all classes if search is empty
            self.class_combobox['values'] = self.all_classes
        else:
            # Filter classes that contain the search text
            filtered = [cls for cls in self.all_classes if search_text in cls.lower()]
            self.class_combobox['values'] = filtered
            
            # If current selection is not in filtered list, clear it
            if self.class_var.get() and self.class_var.get() not in filtered:
                if filtered:
                    self.class_combobox.set(filtered[0])
    
    def configure_api(self):
        """Open dialog to configure API endpoint"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure API Endpoint")
        dialog.geometry("500x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # API URL input
        main_frame = tk.Frame(dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="API Base URL:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        url_var = tk.StringVar(value=self.api_url)
        url_entry = tk.Entry(main_frame, textvariable=url_var, font=('Arial', 10))
        url_entry.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(main_frame, text="Example: http://localhost:5000 or https://api.example.com",
                font=('Arial', 8), fg='gray').pack(anchor=tk.W, pady=(0, 10))
        
        # Info label
        info_text = "The API should provide:\n• /health endpoint for health checks\n• /detect endpoint for object detection (to be implemented)"
        tk.Label(main_frame, text=info_text, justify=tk.LEFT, font=('Arial', 9),
                fg='blue').pack(anchor=tk.W, pady=(0, 15))
        
        # Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        def save_and_close():
            new_url = url_var.get().strip()
            if new_url:
                # Remove trailing slash
                self.api_url = new_url.rstrip('/')
                self.update_status(f"API URL updated: {self.api_url}")
                dialog.destroy()
            else:
                messagebox.showwarning("Invalid URL", "Please enter a valid URL", parent=dialog)
        
        tk.Button(button_frame, text="Save", command=save_and_close,
                 bg='lightgreen', width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                 width=15).pack(side=tk.LEFT, padx=5)
        
        # Focus on entry
        url_entry.focus_set()
        url_entry.select_range(0, tk.END)
    
    def test_api_health(self):
        """Test API connection with health check endpoint"""
        self.update_status("Testing API connection...")
        self.root.update()
        
        try:
            import requests
            
            health_url = f"{self.api_url}/health"
            response = requests.get(health_url, timeout=5)
            
            if response.status_code == 200:
                messagebox.showinfo("API Health Check",
                                   f"✓ API is healthy!\n\n"
                                   f"Endpoint: {health_url}\n"
                                   f"Status: {response.status_code}\n"
                                   f"Response: {response.text[:100]}")
                self.update_status(f"API health check successful: {self.api_url}")
            else:
                messagebox.showwarning("API Health Check",
                                      f"API responded with status {response.status_code}\n\n"
                                      f"Endpoint: {health_url}")
                self.update_status(f"API health check failed: Status {response.status_code}")
                
        except ImportError:
            messagebox.showerror("Missing Dependency",
                               "The 'requests' library is not installed.\n\n"
                               "Install it with: pip install requests")
            self.update_status("Error: requests library not found")
        except requests.exceptions.Timeout:
            messagebox.showerror("Connection Timeout",
                               f"Connection to API timed out.\n\n"
                               f"Endpoint: {self.api_url}/health\n\n"
                               f"Please check the URL and try again.")
            self.update_status("API health check timeout")
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Connection Error",
                               f"Could not connect to API.\n\n"
                               f"Endpoint: {self.api_url}/health\n\n"
                               f"Please check:\n"
                               f"• The API is running\n"
                               f"• The URL is correct\n"
                               f"• Network connectivity")
            self.update_status("API connection failed")
        except Exception as e:
            messagebox.showerror("Error", f"Health check failed:\n{str(e)}")
            self.update_status(f"API health check error: {str(e)}")
    
    def run_api_detection(self):
        """Run object detection using API endpoint"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        # Ask user if they want to clear existing boxes
        if self.bboxes:
            result = messagebox.askyesnocancel(
                "Clear Existing Boxes?",
                "Do you want to clear existing annotations before running API detection?\n\n"
                "Yes: Clear and run detection\n"
                "No: Add to existing boxes\n"
                "Cancel: Don't run detection"
            )
            if result is None:  # Cancel
                return
            elif result:  # Yes
                self.bboxes = []
        
        self.update_status("Running API detection... Please wait...")
        self.root.update()
        
        try:
            import requests
            
            detect_url = f"{self.api_url}/detect"
            
            # Prepare form data
            data = {
                'classes': ','.join(OBJECT_CLASSES),
                'use_gpu': 'true'
            }
            
            # Send image to API
            with open(self.image_path, 'rb') as f:
                files = {'image': (os.path.basename(self.image_path), f, 'image/jpeg')}
                response = requests.post(detect_url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                detections = result.get('objects', [])
                
                # Convert detections to bounding boxes (scaled for display)
                for det in detections:
                    bbox_coords = det['bbox']  # [x1, y1, x2, y2]
                    class_name = det.get('class', 'unknown')
                    confidence = det.get('confidence', 1.0)
                    
                    # Scale coordinates for display
                    x1 = int(bbox_coords[0] * self.scale_factor)
                    y1 = int(bbox_coords[1] * self.scale_factor)
                    x2 = int(bbox_coords[2] * self.scale_factor)
                    y2 = int(bbox_coords[3] * self.scale_factor)
                    
                    bbox = BoundingBox(x1, y1, x2, y2, class_name, confidence)
                    self.bboxes.append(bbox)
                
                self.redraw_canvas()
                self.update_bbox_list()
                
                exec_time = result.get('execution_time_seconds', 0)
                self.update_status(f"API detected {len(detections)} objects ({exec_time:.2f}s)")
                
                messagebox.showinfo("API Detection Complete",
                                  f"Successfully detected {len(detections)} objects\n\n"
                                  f"Execution time: {exec_time:.2f}s")
            else:
                error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                messagebox.showerror("API Error",
                                   f"API returned status {response.status_code}\n\n"
                                   f"Details: {error_detail}")
                self.update_status(f"API detection failed: Status {response.status_code}")
            
        except ImportError:
            messagebox.showerror("Missing Dependency",
                               "The 'requests' library is not installed.\n\n"
                               "Install it with: pip install requests")
            self.update_status("Error: requests library not found")
        except requests.exceptions.Timeout:
            messagebox.showerror("Connection Timeout",
                               f"Request to API timed out.\n\n"
                               f"Endpoint: {self.api_url}/detect\n\n"
                               f"The detection may take longer than expected.")
            self.update_status("API detection timeout")
        except requests.exceptions.ConnectionError:
            messagebox.showerror("Connection Error",
                               f"Could not connect to API.\n\n"
                               f"Endpoint: {self.api_url}/detect\n\n"
                               f"Please check:\n"
                               f"• The API is running\n"
                               f"• The URL is correct\n"
                               f"• Network connectivity")
            self.update_status("API connection failed")
        except Exception as e:
            messagebox.showerror("Error", f"API detection failed:\n{str(e)}")
            self.update_status(f"API detection error: {str(e)}")
    
    def batch_process_directory(self):
        """Batch process all images in a directory using API endpoint"""
        # Get directory to process
        if self.image_path:
            initial_dir = os.path.dirname(self.image_path)
        else:
            initial_dir = os.getcwd()
        
        directory = filedialog.askdirectory(
            title="Select Directory to Batch Process",
            initialdir=initial_dir
        )
        
        if not directory:
            return
        
        # Find all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        image_files = []
        
        for file in os.listdir(directory):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(directory, file)
                json_path = os.path.splitext(image_path)[0] + '.json'
                
                # Only add if JSON doesn't exist
                if not os.path.exists(json_path):
                    image_files.append(image_path)
        
        if not image_files:
            messagebox.showinfo("Batch Process",
                              f"No images found to process in:\n{directory}\n\n"
                              f"All images either have existing JSON files or no images were found.")
            return
        
        # Confirm with user
        result = messagebox.askyesno(
            "Batch Process Confirmation",
            f"Found {len(image_files)} images without JSON annotations.\n\n"
            f"Directory: {directory}\n\n"
            f"This will send each image to the API endpoint:\n{self.api_url}\n\n"
            f"Continue?"
        )
        
        if not result:
            return
        
        # Create progress dialog
        progress_dialog = tk.Toplevel(self.root)
        progress_dialog.title("Batch Processing")
        progress_dialog.geometry("600x300")
        progress_dialog.transient(self.root)
        progress_dialog.grab_set()
        
        # Center the dialog
        progress_dialog.update_idletasks()
        x = (progress_dialog.winfo_screenwidth() // 2) - (progress_dialog.winfo_width() // 2)
        y = (progress_dialog.winfo_screenheight() // 2) - (progress_dialog.winfo_height() // 2)
        progress_dialog.geometry(f"+{x}+{y}")
        
        # Progress dialog contents
        main_frame = tk.Frame(progress_dialog, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Batch Processing Images", 
                font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        progress_label = tk.Label(main_frame, text="Starting...", font=('Arial', 10))
        progress_label.pack(pady=(0, 10))
        
        # Progress bar
        from tkinter import ttk as tkttk
        progress_bar = tkttk.Progressbar(main_frame, length=500, mode='determinate')
        progress_bar.pack(pady=(0, 10))
        
        # Log text area
        log_frame = tk.Frame(main_frame)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        log_scrollbar = tk.Scrollbar(log_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        log_text = tk.Text(log_frame, height=10, yscrollcommand=log_scrollbar.set,
                          font=('Courier', 9))
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=log_text.yview)
        
        # Cancel button
        cancel_button = tk.Button(main_frame, text="Close", command=progress_dialog.destroy,
                                 state=tk.DISABLED)
        cancel_button.pack()
        
        # Process images
        def process_images():
            try:
                import requests
                
                total = len(image_files)
                processed = 0
                successful = 0
                failed = 0
                
                progress_bar['maximum'] = total
                
                for i, image_path in enumerate(image_files):
                    filename = os.path.basename(image_path)
                    progress_label.config(text=f"Processing {i+1}/{total}: {filename}")
                    progress_bar['value'] = i
                    progress_dialog.update()
                    
                    log_text.insert(tk.END, f"\n[{i+1}/{total}] Processing: {filename}\n")
                    log_text.see(tk.END)
                    progress_dialog.update()
                    
                    try:
                        # Call API detection endpoint
                        detect_url = f"{self.api_url}/detect"
                        
                        # Prepare form data
                        data = {
                            'classes': ','.join(OBJECT_CLASSES),
                            'use_gpu': 'true'
                        }
                        
                        # Send image to API
                        with open(image_path, 'rb') as f:
                            files = {'image': (filename, f, 'image/jpeg')}
                            response = requests.post(detect_url, files=files, data=data, timeout=60)
                        
                        if response.status_code == 200:
                            result = response.json()
                            detections = result.get('objects', [])
                            
                            # Save JSON annotation
                            json_path = os.path.splitext(image_path)[0] + '.json'
                            output = {
                                "image": filename,
                                "objects": detections
                            }
                            
                            with open(json_path, 'w') as f:
                                json.dump(output, f, indent=2)
                            
                            successful += 1
                            exec_time = result.get('execution_time_seconds', 0)
                            log_text.insert(tk.END, 
                                          f"  ✓ Success: {len(detections)} objects ({exec_time:.2f}s)\n", 
                                          'success')
                        else:
                            failed += 1
                            log_text.insert(tk.END, 
                                          f"  ✗ Failed: API returned status {response.status_code}\n", 
                                          'error')
                        
                    except requests.exceptions.Timeout:
                        failed += 1
                        log_text.insert(tk.END, f"  ✗ Timeout: Request timed out\n", 'error')
                    except requests.exceptions.ConnectionError:
                        failed += 1
                        log_text.insert(tk.END, f"  ✗ Connection Error: Could not reach API\n", 'error')
                    except Exception as e:
                        failed += 1
                        log_text.insert(tk.END, f"  ✗ Error: {str(e)}\n", 'error')
                    
                    processed += 1
                    log_text.see(tk.END)
                    progress_dialog.update()
                
                # Complete
                progress_bar['value'] = total
                progress_label.config(text=f"Complete: {processed} processed, {successful} successful, {failed} failed")
                log_text.insert(tk.END, f"\n{'='*60}\n")
                log_text.insert(tk.END, f"Batch processing complete!\n", 'success')
                log_text.insert(tk.END, f"Total: {processed} | Success: {successful} | Failed: {failed}\n")
                log_text.see(tk.END)
                
                cancel_button.config(state=tk.NORMAL)
                self.update_status(f"Batch processing complete: {successful}/{processed} successful")
                
            except ImportError:
                log_text.insert(tk.END, "\n✗ Error: 'requests' library not installed\n", 'error')
                log_text.insert(tk.END, "Install with: pip install requests\n")
                cancel_button.config(state=tk.NORMAL)
                self.update_status("Error: requests library not found")
        
        # Configure text tags for colored output
        log_text.tag_config('success', foreground='green')
        log_text.tag_config('error', foreground='red')
        log_text.tag_config('warning', foreground='orange')
        log_text.tag_config('info', foreground='blue')
        
        # Start processing after dialog is shown
        progress_dialog.after(100, process_images)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = AnnotationTool(root)
    
    # Check command line arguments for image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            app.load_image(image_path)
    
    root.mainloop()


if __name__ == "__main__":
    main()

