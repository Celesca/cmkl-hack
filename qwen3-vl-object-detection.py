"""
Person Object Detection using Qwen3-VL via Ollama
Detects people in images/video frames and returns bounding box coordinates
"""

import cv2
import numpy as np
import base64
import requests
import json
import os
import re
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from PIL import Image
import io


@dataclass
class PersonDetection:
    """Single person detection result"""
    person_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    description: str
    attributes: Dict[str, str]  # clothing, pose, etc.


@dataclass
class FrameDetectionResult:
    """Detection result for a single frame"""
    frame_idx: int
    timestamp: float
    persons_detected: int
    detections: List[PersonDetection]
    raw_response: str
    processing_time: float


class Qwen3VLPersonDetector:
    """
    Person Detector using Qwen3-VL model via Ollama
    
    Features:
    - Detect persons in images/video frames
    - Return approximate bounding boxes
    - Describe person attributes (clothing, pose, action)
    - Process video frames for tracking
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5-vl",
        confidence_threshold: float = 0.5,
        max_image_size: int = 768
    ):
        """
        Initialize the Qwen3-VL Person Detector
        
        Args:
            ollama_url: Base URL for Ollama API
            model_name: Name of the Qwen VL model in Ollama
            confidence_threshold: Minimum confidence to report detection
            max_image_size: Maximum dimension for image resizing
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.max_image_size = max_image_size
        
        # Verify Ollama connection
        self._verify_ollama_connection()
        
        print(f"✅ Qwen3VLPersonDetector initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Ollama URL: {self.ollama_url}")
    
    def _verify_ollama_connection(self) -> bool:
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '').split(':')[0] for m in models]
                if self.model_name.split(':')[0] not in model_names:
                    print(f"⚠️ Warning: Model '{self.model_name}' not found in Ollama.")
                    print(f"   Available models: {model_names}")
                    print(f"   Run: ollama pull {self.model_name}")
                    return False
                return True
            else:
                print(f"⚠️ Warning: Could not verify Ollama models")
                return False
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Warning: Cannot connect to Ollama at {self.ollama_url}")
            print("   Make sure Ollama is running: ollama serve")
            return False
    
    def _image_to_base64(
        self,
        image: Union[np.ndarray, str, Image.Image]
    ) -> Tuple[str, Tuple[int, int]]:
        """
        Convert image to base64 encoded string
        
        Args:
            image: OpenCV BGR frame, file path, or PIL Image
            
        Returns:
            Tuple of (base64 string, (width, height))
        """
        if isinstance(image, str):
            # Load from file path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert OpenCV BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get original size
        orig_width, orig_height = pil_image.size
        
        # Resize if needed
        if max(orig_width, orig_height) > self.max_image_size:
            scale = self.max_image_size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8'), (orig_width, orig_height)
    
    def _parse_bbox_from_text(
        self,
        text: str,
        image_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Parse bounding box coordinates from model response
        
        Args:
            text: Model response text
            image_size: (width, height) of original image
            
        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        width, height = image_size
        bboxes = []
        
        # Pattern 1: [x1, y1, x2, y2] format (normalized 0-1000 or 0-1)
        pattern1 = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        matches = re.findall(pattern1, text)
        
        for match in matches:
            x1, y1, x2, y2 = map(float, match)
            
            # Check if normalized (0-1000 range from Qwen)
            if max(x1, y1, x2, y2) > 10:
                # Qwen uses 0-1000 normalization
                x1 = int(x1 * width / 1000)
                y1 = int(y1 * height / 1000)
                x2 = int(x2 * width / 1000)
                y2 = int(y2 * height / 1000)
            else:
                # 0-1 normalization
                x1 = int(x1 * width)
                y1 = int(y1 * height)
                x2 = int(x2 * width)
                y2 = int(y2 * height)
            
            # Ensure valid bbox
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
        
        # Pattern 2: percentage format "x: 10%, y: 20%, width: 30%, height: 40%"
        pattern2 = r'x[:\s]*(\d+(?:\.\d+)?)\s*%.*?y[:\s]*(\d+(?:\.\d+)?)\s*%.*?(?:width|w)[:\s]*(\d+(?:\.\d+)?)\s*%.*?(?:height|h)[:\s]*(\d+(?:\.\d+)?)\s*%'
        matches = re.findall(pattern2, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            x_pct, y_pct, w_pct, h_pct = map(float, match)
            x1 = int(x_pct * width / 100)
            y1 = int(y_pct * height / 100)
            x2 = int((x_pct + w_pct) * width / 100)
            y2 = int((y_pct + h_pct) * height / 100)
            
            if x2 > x1 and y2 > y1:
                bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def detect_persons(
        self,
        image: Union[np.ndarray, str, Image.Image],
        return_attributes: bool = True
    ) -> List[PersonDetection]:
        """
        Detect persons in an image
        
        Args:
            image: OpenCV BGR frame, file path, or PIL Image
            return_attributes: Whether to include person attributes
            
        Returns:
            List of PersonDetection objects
        """
        import time
        start_time = time.time()
        
        # Convert image to base64
        image_base64, image_size = self._image_to_base64(image)
        width, height = image_size
        
        # Construct detection prompt
        if return_attributes:
            prompt = """Analyze this image and detect all people/persons visible.

For each person detected, provide:
1. Bounding box coordinates as [x1, y1, x2, y2] in 0-1000 scale (where 1000 is full width/height)
2. Confidence score (0.0 to 1.0)
3. Brief description (clothing, pose, action)

Respond in JSON format:
{
  "persons": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95,
      "description": "person in blue shirt standing",
      "clothing": "blue shirt, jeans",
      "pose": "standing",
      "action": "looking at camera"
    }
  ],
  "total_count": 1
}

If no persons detected, return {"persons": [], "total_count": 0}

JSON response:"""
        else:
            prompt = """Detect all people/persons in this image.
Return bounding boxes as [x1, y1, x2, y2] in 0-1000 scale.

JSON format: {"persons": [{"bbox": [x1,y1,x2,y2], "confidence": 0.9}], "total_count": N}

JSON response:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"❌ Ollama API error: {response.status_code}")
                return []
            
            result = response.json()
            raw_response = result.get('response', '')
            
            # Parse JSON response
            detections = []
            try:
                # Extract JSON from response
                json_str = raw_response
                if '```json' in json_str:
                    json_str = json_str.split('```json')[1].split('```')[0]
                elif '```' in json_str:
                    json_str = json_str.split('```')[1].split('```')[0]
                
                # Find JSON object
                start_idx = json_str.find('{')
                end_idx = json_str.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = json_str[start_idx:end_idx]
                
                parsed = json.loads(json_str)
                persons = parsed.get('persons', [])
                
                for i, person in enumerate(persons):
                    bbox_raw = person.get('bbox', [])
                    if len(bbox_raw) == 4:
                        x1, y1, x2, y2 = bbox_raw
                        
                        # Convert from 0-1000 scale to pixels
                        if max(x1, y1, x2, y2) > 10:
                            x1 = int(x1 * width / 1000)
                            y1 = int(y1 * height / 1000)
                            x2 = int(x2 * width / 1000)
                            y2 = int(y2 * height / 1000)
                        else:
                            x1 = int(x1 * width)
                            y1 = int(y1 * height)
                            x2 = int(x2 * width)
                            y2 = int(y2 * height)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, width))
                        y1 = max(0, min(y1, height))
                        x2 = max(0, min(x2, width))
                        y2 = max(0, min(y2, height))
                        
                        confidence = float(person.get('confidence', 0.8))
                        
                        if confidence >= self.confidence_threshold and x2 > x1 and y2 > y1:
                            detection = PersonDetection(
                                person_id=i + 1,
                                confidence=confidence,
                                bbox=(x1, y1, x2, y2),
                                center=((x1 + x2) // 2, (y1 + y2) // 2),
                                description=person.get('description', ''),
                                attributes={
                                    'clothing': person.get('clothing', ''),
                                    'pose': person.get('pose', ''),
                                    'action': person.get('action', '')
                                }
                            )
                            detections.append(detection)
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️ JSON parse error: {e}")
                # Fallback: try to extract bboxes from raw text
                bboxes = self._parse_bbox_from_text(raw_response, image_size)
                for i, bbox in enumerate(bboxes):
                    detections.append(PersonDetection(
                        person_id=i + 1,
                        confidence=0.7,
                        bbox=bbox,
                        center=((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                        description="person detected",
                        attributes={}
                    ))
            
            processing_time = time.time() - start_time
            print(f"✅ Detected {len(detections)} person(s) in {processing_time:.2f}s")
            
            return detections
            
        except requests.exceptions.Timeout:
            print("⚠️ Timeout during person detection")
            return []
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return []
    
    def detect_in_video(
        self,
        video_path: str,
        frame_sample_rate: int = 1,
        max_frames: Optional[int] = None
    ) -> List[FrameDetectionResult]:
        """
        Detect persons in video frames
        
        Args:
            video_path: Path to video file
            frame_sample_rate: Frames to process per second
            max_frames: Maximum number of frames to process
            
        Returns:
            List of FrameDetectionResult for each processed frame
        """
        import time
        
        print(f"📹 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Total frames: {total_frames}")
        
        frame_interval = max(1, int(fps / frame_sample_rate))
        results = []
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                if max_frames and processed_count >= max_frames:
                    break
                
                timestamp = frame_idx / fps
                print(f"   Processing frame {processed_count + 1} (t={timestamp:.2f}s)...", end='\r')
                
                start_time = time.time()
                detections = self.detect_persons(frame, return_attributes=True)
                processing_time = time.time() - start_time
                
                result = FrameDetectionResult(
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    persons_detected=len(detections),
                    detections=detections,
                    raw_response="",
                    processing_time=processing_time
                )
                results.append(result)
                processed_count += 1
            
            frame_idx += 1
        
        cap.release()
        
        print(f"\n✅ Processed {len(results)} frames")
        total_persons = sum(r.persons_detected for r in results)
        print(f"   Total person detections: {total_persons}")
        
        return results
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[PersonDetection],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: OpenCV BGR image
            detections: List of PersonDetection objects
            color: BGR color for bounding boxes
            thickness: Line thickness
            show_labels: Whether to show labels
            
        Returns:
            Image with drawn bounding boxes
        """
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw rectangle
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            if show_labels:
                # Prepare label
                label = f"Person {det.person_id}: {det.confidence:.0%}"
                if det.attributes.get('action'):
                    label += f" - {det.attributes['action']}"
                
                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    output,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 5, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
        
        return output
    
    def save_detection_results(
        self,
        results: List[FrameDetectionResult],
        output_path: str
    ):
        """Save detection results to JSON file"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total_frames': len(results),
            'frames': []
        }
        
        for result in results:
            frame_data = {
                'frame_idx': result.frame_idx,
                'timestamp': result.timestamp,
                'persons_detected': result.persons_detected,
                'processing_time': result.processing_time,
                'detections': [asdict(d) for d in result.detections]
            }
            output_data['frames'].append(frame_data)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


def main():
    """Example usage of Qwen3VLPersonDetector"""
    
    # Initialize detector
    detector = Qwen3VLPersonDetector(
        ollama_url="http://localhost:11434",
        model_name="qwen2.5-vl",
        confidence_threshold=0.5
    )
    
    # Example 1: Detect in single image
    print("\n" + "=" * 50)
    print("Example: Single Image Detection")
    print("=" * 50)
    
    test_image = "test.jpg"
    if os.path.exists(test_image):
        detections = detector.detect_persons(test_image)
        
        for det in detections:
            print(f"\n👤 Person {det.person_id}:")
            print(f"   Confidence: {det.confidence:.0%}")
            print(f"   BBox: {det.bbox}")
            print(f"   Description: {det.description}")
            if det.attributes.get('action'):
                print(f"   Action: {det.attributes['action']}")
        
        # Draw and save result
        img = cv2.imread(test_image)
        output = detector.draw_detections(img, detections)
        cv2.imwrite("detection_result.jpg", output)
        print("\n💾 Saved: detection_result.jpg")
    else:
        print(f"Test image not found: {test_image}")
    
    # Example 2: Detect in video
    print("\n" + "=" * 50)
    print("Example: Video Detection")
    print("=" * 50)
    
    test_video = "test.mp4"
    if os.path.exists(test_video):
        results = detector.detect_in_video(
            test_video,
            frame_sample_rate=1,
            max_frames=10
        )
        
        # Print summary
        for result in results:
            print(f"\nFrame {result.frame_idx} (t={result.timestamp:.2f}s): {result.persons_detected} person(s)")
            for det in result.detections:
                print(f"  - Person {det.person_id}: {det.description}")
        
        # Save results
        detector.save_detection_results(results, "video_detection_results.json")
    else:
        print(f"Test video not found: {test_video}")
    
    print("\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    print("""
# Initialize detector
detector = Qwen3VLPersonDetector(model_name="qwen2.5-vl")

# Detect in image
detections = detector.detect_persons("image.jpg")

# Detect in video
results = detector.detect_in_video("video.mp4", frame_sample_rate=1)

# Draw bounding boxes
img = cv2.imread("image.jpg")
output = detector.draw_detections(img, detections)
cv2.imwrite("output.jpg", output)
""")


if __name__ == "__main__":
    main()
