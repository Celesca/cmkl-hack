import re
import random
import requests
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers.image_utils import load_image
from rapidfuzz import fuzz, process
from langdetect import detect
from aift import setting
from aift.nlp import text_cleansing, text_sum
from aift.nlp.translation import th2en
import os
from urllib.parse import urlparse
import io
import base64
from typing import List, Dict, Any, Union, Tuple, Optional

# ============================================
# === UTILS ===
# ============================================
setting.set_api_key('KIeFbAUzBG4A3Zrvo9gp1fV6bTwICIAG')

def process_and_translate_list(items_to_check):
    """
    Detects language of items in a list and translates Thai items to English using aift.

    Args:
        items_to_check (list): A list of strings to process.

    Returns:
        list: A new list with Thai items translated to English.
    """
    processed_items = []
    for item in items_to_check:
        if len(item) > 10000:
            item = text_sum.summarize(item)
        try:
            detected_language = detect(item)
            if detected_language == 'th':
                try:
                    # Assuming API key is set globally or handled by setting.set_api_key
                    cleaned_text = text_cleansing.clean(item)['cleansing_text']
                    translated_item = th2en.translate(cleaned_text)['translated_text']
                    processed_items.append(translated_item)
                except Exception as e:
                    # print(f"Error during aift translation of '{item}': {e}") # Removed print
                    processed_items.append(item) # Append original item on translation error
            else:
                processed_items.append(item)
        except Exception as e:
            # print(f"Error during language detection for '{item}': {e}") # Removed print
            processed_items.append(item) # Append original item in case of detection error

    return processed_items

def clean_and_format_label(label: str) -> str:
    label = label.strip()
    label = re.sub(r'\s+', ' ', label)
    return label.title().replace(" ", "-").replace(".", "")


def match_labels_fuzzy(detected_labels, reference_labels, threshold=70):
    final_labels = []
    for detected in detected_labels:
        detected_clean = detected.strip()
        detected_clean = re.sub(r'\s+', ' ', detected_clean)
        detected_clean = detected_clean.title()

        best_match, score, _ = process.extractOne(
            detected_clean,
            reference_labels,
            scorer=fuzz.token_sort_ratio
        )
        
        final_labels.append(best_match)

    print(final_labels)
    return final_labels


def generate_colors(labels):
    unique_labels = list(set(labels))
    random.seed(888)
    colors = {label: (random.random(), random.random(), random.random()) for label in unique_labels}
    return colors


class DynamicGroundingDINO:
    """
    Grounding DINO model for zero-shot object detection with text queries.
    Updated with Thai language support and fuzzy matching.
    """

    def __init__(self, model_id: str = "rziga/mm_grounding_dino_large_all", device: str = "auto"):
        """
        Initialize the Grounding DINO model

        Args:
            model_id: Model identifier from HuggingFace
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model on {self.device}...")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        else:
            print("Running on CPU - this may be slower and require more memory")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)
            
            # Load model with CPU-optimized settings
            if self.device == "cpu":
                print("Loading model optimized for CPU...")
                # Load with lower precision for CPU to reduce memory usage
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,  # Ensure float32 for CPU compatibility
                    low_cpu_mem_usage=True,     # Optimize memory usage
                    device_map=None             # Don't use device mapping for CPU
                )
            else:
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            
            # Move model to device with error handling
            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    self.model = self.model.to(self.device)
                    print(f"Model successfully moved to {self.device}")
                except Exception as e:
                    print(f"Failed to move model to CUDA, falling back to CPU: {e}")
                    self.device = "cpu"
                    self.model = self.model.to(self.device)
            else:
                self.model = self.model.to(self.device)
                # Set model to eval mode for CPU inference
                self.model.eval()
                
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def load_image(self, image_source: Union[str, Image.Image]) -> Image.Image:
        """
        Load image from various sources

        Args:
            image_source: Can be URL, local file path, or PIL Image

        Returns:
            PIL Image in RGB format
        """
        if isinstance(image_source, str):
            if self._is_url(image_source):
                # Load from URL
                try:
                    response = requests.get(image_source, stream=True, timeout=30)
                    response.raise_for_status()
                    image = Image.open(response.raw)
                    print(f"Loaded image from URL: {image_source}")
                except Exception as e:
                    raise ValueError(f"Failed to load image from URL: {e}")
            else:
                # Try different path variations for local files
                possible_paths = [
                    image_source,  # Exact path
                    os.path.join("/app", image_source),  # Container app path
                    os.path.join("/app/images", image_source),  # Container images path
                    os.path.join(".", image_source),  # Current directory
                    os.path.join("images", image_source)  # Local images directory
                ]
                
                image = None
                for path in possible_paths:
                    if os.path.exists(path):
                        try:
                            image = Image.open(path)
                            print(f"Loaded local image from: {path}")
                            break
                        except Exception as e:
                            print(f"Failed to load from {path}: {e}")
                            continue
                
                if image is None:
                    raise FileNotFoundError(f"Image file not found. Tried paths: {possible_paths}")
                    
        elif isinstance(image_source, Image.Image):
            image = image_source
            print("Using provided PIL Image")
        else:
            raise ValueError("image_source must be URL, file path, or PIL Image")

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize image if too large for CPU processing
        if self.device == "cpu":
            max_size = 1024  # Maximum dimension for CPU processing
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image for CPU processing: {image.size}")

        return image

    def load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """
        Load image from bytes data

        Args:
            image_bytes: Image data in bytes format

        Returns:
            PIL Image in RGB format
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {e}")

    def _is_url(self, string: str) -> bool:
        """Check if string is a valid URL"""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except:
            return False

    def detect_objects(self, image_source: Union[str, Image.Image, bytes], 
                      text_queries: Union[str, List[str]], 
                      box_threshold: float = 0.35, 
                      text_threshold: float = 0.35) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Detect objects in image based on text queries using the new implementation

        Args:
            image_source: Image source (URL, file path, PIL Image, or bytes)
            text_queries: List of text descriptions to search for
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching

        Returns:
            Tuple of (PIL Image, detection results)
        """
        try:
            # Load image
            if isinstance(image_source, bytes):
                image = self.load_image_from_bytes(image_source)
            else:
                image = self.load_image(image_source)

            # Prepare text queries - ensure proper format
            if isinstance(text_queries, str):
                text_queries = [text_queries]
            
            # Process and translate text labels
            translated_text_labels = process_and_translate_list(text_queries)
            
            # Clean labels
            processed_text_labels = [clean_and_format_label(label) for label in translated_text_labels]
            print(f"Processed labels: {processed_text_labels}")

            print(f"Searching for: {', '.join(text_queries)}")
            print(f"Thresholds - Box: {box_threshold}, Text: {text_threshold}")

            # Process inputs with proper error handling
            try:
                inputs = self.processor(images=image, text=[processed_text_labels], return_tensors="pt")
                
                # Move inputs to device safely with proper dtype handling
                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        # Ensure proper dtype for CPU
                        if self.device == "cpu" and v.dtype == torch.float16:
                            v = v.to(torch.float32)
                        processed_inputs[k] = v.to(self.device)
                    else:
                        processed_inputs[k] = v
                
                inputs = processed_inputs
                
            except Exception as e:
                print(f"Error during preprocessing: {e}")
                raise ValueError(f"Failed to preprocess inputs: {e}")

            # Run inference with error handling and CPU optimizations
            try:
                # Set model to eval mode
                self.model.eval()
                
                with torch.no_grad():
                    # Use torch.inference_mode for better CPU performance if available
                    if hasattr(torch, 'inference_mode') and self.device == "cpu":
                        with torch.inference_mode():
                            outputs = self.model(**inputs)
                    else:
                        outputs = self.model(**inputs)
                        
            except RuntimeError as e:
                if "could not create a primitive" in str(e):
                    print(f"Primitive creation error - likely CPU/memory issue: {e}")
                    print("This might be due to insufficient memory or CPU optimization issues")
                    print("Try using a smaller image or different model")
                    raise ValueError(f"Model inference failed due to resource constraints: {e}")
                else:
                    print(f"Runtime error during model inference: {e}")
                    raise ValueError(f"Model inference failed: {e}")
            except Exception as e:
                print(f"Error during model inference: {e}")
                raise ValueError(f"Model inference failed: {e}")

            # Post-process results
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    text_labels=processed_text_labels,
                    target_sizes=[(image.height, image.width)]
                )
                
                # Fuzzy match the labels
                if len(results[0]["labels"]) > 0:
                    final_labels = match_labels_fuzzy(results[0]["labels"], processed_text_labels)
                    results[0]["labels"] = final_labels
                    
            except Exception as e:
                print(f"Error during post-processing: {e}")
                raise ValueError(f"Post-processing failed: {e}")

            return image, results[0]
            
        except Exception as e:
            print(f"Detection error: {e}")
            raise e

    def generate_colors(self, labels: List[str]) -> Dict[str, np.ndarray]:
        """Generate distinct colors for different labels"""
        unique_labels = list(set(labels))
        random.seed(888)
        colors = {label: (random.random(), random.random(), random.random()) for label in unique_labels}
        return colors

    def create_visualization(self, image: Image.Image, results: Dict[str, Any], 
                           figsize: Tuple[int, int] = (12, 8),
                           show_confidence: bool = True, 
                           font_size: int = 12) -> Image.Image:
        """
        Create visualization with bounding boxes and return as PIL Image

        Args:
            image: PIL Image
            results: Detection results from the model
            figsize: Figure size for matplotlib
            show_confidence: Whether to show confidence scores
            font_size: Font size for labels

        Returns:
            PIL Image with visualized detection results
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(image)

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        if len(boxes) == 0:
            ax.set_title('No Objects Detected', fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
        else:
            # Generate colors for labels
            color_map = self.generate_colors(labels)

            # Draw bounding boxes
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                box = box.tolist() if hasattr(box, "tolist") else box
                x_min, y_min, x_max, y_max = box
                confidence = round(score.item() if hasattr(score, "item") else score, 3)

                # Calculate dimensions
                width = x_max - x_min
                height = y_max - y_min

                # Get color
                color = color_map[label]

                # Format label for display
                pretty_label = label.replace("-", " ")

                # Create rectangle
                rect = patches.Rectangle(
                    (x_min, y_min), width, height,
                    linewidth=3, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)

                # Add label
                if show_confidence:
                    text = f'{pretty_label}: {confidence}'
                else:
                    text = pretty_label

                ax.text(
                    x_min, max(y_min - 10, 0),
                    text,
                    color='white',
                    fontsize=font_size,
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8)
                )

            ax.set_xlim(0, image.size[0])
            ax.set_ylim(image.size[1], 0)
            ax.axis('off')
            ax.set_title(f'Detected Objects: {", ".join(set([l.replace("-", " ") for l in labels]))}',
                        fontsize=14, fontweight='bold')

            # Add legend
            legend_elements = [patches.Patch(color=color, label=label.replace("-", " ")) 
                              for label, color in color_map.items()]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

        plt.tight_layout()

        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)

        return result_image

    def process_detection(self, image_source: Union[str, Image.Image, bytes], 
                         text_queries: Union[str, List[str]], 
                         box_threshold: float = 0.35,
                         text_threshold: float = 0.35,
                         return_visualization: bool = True) -> Dict[str, Any]:
        """
        Complete detection pipeline with structured output for API

        Args:
            image_source: Image source (URL, file path, PIL Image, or bytes)
            text_queries: Text descriptions to search for
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
            return_visualization: Whether to return visualization image

        Returns:
            Dictionary containing detection results and optional visualization
        """
        try:
            # Validate inputs
            if not text_queries or (isinstance(text_queries, list) and len(text_queries) == 0):
                return {
                    "success": False,
                    "error": "No text queries provided",
                    "num_detections": 0,
                    "detections": []
                }

            # Run detection
            image, results = self.detect_objects(
                image_source, text_queries, box_threshold, text_threshold
            )

            # Check if we have valid results
            if not results or "boxes" not in results:
                return {
                    "success": False,
                    "error": "No valid detection results returned",
                    "num_detections": 0,
                    "detections": []
                }

            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]

            # Format detection results
            detections = []
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                try:
                    box = box.tolist() if hasattr(box, 'tolist') else box
                    x_min, y_min, x_max, y_max = box
                    confidence = round(score.item() if hasattr(score, 'item') else float(score), 3)

                    detections.append({
                        "id": i + 1,
                        "label": label,
                        "confidence": confidence,
                        "bounding_box": {
                            "x_min": round(float(x_min), 2),
                            "y_min": round(float(y_min), 2),
                            "x_max": round(float(x_max), 2),
                            "y_max": round(float(y_max), 2),
                            "width": round(float(x_max - x_min), 2),
                            "height": round(float(y_max - y_min), 2)
                        }
                    })
                except Exception as e:
                    print(f"Error processing detection {i}: {e}")
                    continue

            response_data = {
                "success": True,
                "num_detections": len(detections),
                "detections": detections,
                "image_size": {
                    "width": image.size[0],
                    "height": image.size[1]
                },
                "queries": text_queries if isinstance(text_queries, list) else [text_queries],
                "thresholds": {
                    "box_threshold": box_threshold,
                    "text_threshold": text_threshold
                }
            }

            # Add visualization if requested
            if return_visualization:
                try:
                    viz_image = self.create_visualization(image, results)
                    
                    # Convert to base64 for API response
                    buf = io.BytesIO()
                    viz_image.save(buf, format='PNG')
                    buf.seek(0)
                    viz_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    
                    response_data["visualization"] = {
                        "image_base64": viz_base64,
                        "format": "png"
                    }
                except Exception as e:
                    print(f"Error creating visualization: {e}")
                    response_data["visualization"] = None

            return response_data

        except Exception as e:
            print(f"Process detection error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e),
                "num_detections": 0,
                "detections": []
            }


class ModelManager:
    """
    Singleton class to manage the model instance
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_id: str = "rziga/mm_grounding_dino_large_all", device: str = "auto") -> DynamicGroundingDINO:
        """Get or create model instance"""
        if self._model is None:
            self._model = DynamicGroundingDINO(model_id=model_id, device=device)
        return self._model

    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None
