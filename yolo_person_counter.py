"""
YOLO Person Counter Module
Uses YOLOv8 nano for fast person detection and counting in videos
"""

import cv2
import numpy as np
import tempfile
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    logger.info("YOLO (ultralytics) imported successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    YOLO = None
    logger.warning(f"YOLO not available: {e}. Install with: pip install ultralytics")


@dataclass
class PersonDetection:
    """Single person detection result"""
    frame_idx: int
    timestamp: float
    person_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: Tuple[int, int]


@dataclass
class FrameCount:
    """Person count for a single frame"""
    frame_idx: int
    timestamp: float
    person_count: int
    detections: List[PersonDetection]


@dataclass
class CountingResult:
    """Complete counting result for a video"""
    video_path: str
    total_frames: int
    fps: float
    duration: float
    frame_counts: List[FrameCount]
    max_persons: int
    min_persons: int
    avg_persons: float
    output_video_path: Optional[str] = None
    output_video_base64: Optional[str] = None


class YOLOPersonCounter:
    """
    YOLO-based person counter for videos
    Uses YOLOv8 nano for fast inference
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",  # nano model for speed
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize YOLO person counter
        
        Args:
            model_name: YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('auto', 'cpu', 'cuda', '0', etc.)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLO not available. Install with: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load YOLO model
        logger.info(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        
        # Person class ID in COCO dataset is 0
        self.person_class_id = 0
        
        logger.info(f"YOLO model loaded successfully")
    
    def detect_persons_in_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float
    ) -> FrameCount:
        """
        Detect persons in a single frame
        
        Args:
            frame: BGR image as numpy array
            frame_idx: Frame index
            timestamp: Timestamp in seconds
            
        Returns:
            FrameCount with detection results
        """
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.person_class_id],  # Only detect persons
            verbose=False
        )
        
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for idx, box in enumerate(boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Calculate center point
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    detection = PersonDetection(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        person_id=idx,
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        center=(center_x, center_y)
                    )
                    detections.append(detection)
        
        return FrameCount(
            frame_idx=frame_idx,
            timestamp=timestamp,
            person_count=len(detections),
            detections=detections
        )
    
    def draw_detections_on_frame(
        self,
        frame: np.ndarray,
        frame_count: FrameCount,
        show_count: bool = True,
        show_confidence: bool = True,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        count_bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Draw detection boxes and count on frame
        
        Args:
            frame: BGR image
            frame_count: FrameCount with detections
            show_count: Whether to show person count overlay
            show_confidence: Whether to show confidence scores
            box_color: BGR color for bounding boxes
            text_color: BGR color for text
            count_bg_color: BGR color for count background
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw bounding boxes
        for detection in frame_count.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label
            if show_confidence:
                label = f"Person {detection.person_id + 1}: {detection.confidence:.2f}"
            else:
                label = f"Person {detection.person_id + 1}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 5, y1),
                box_color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                1
            )
            
            # Draw center point
            cv2.circle(annotated, detection.center, 4, (0, 0, 255), -1)
        
        # Draw person count overlay
        if show_count:
            count_text = f"Persons: {frame_count.person_count}"
            time_text = f"Time: {frame_count.timestamp:.2f}s"
            
            # Count background
            cv2.rectangle(annotated, (10, 10), (200, 70), count_bg_color, -1)
            cv2.rectangle(annotated, (10, 10), (200, 70), box_color, 2)
            
            # Count text
            cv2.putText(
                annotated,
                count_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2
            )
            
            # Time text
            cv2.putText(
                annotated,
                time_text,
                (20, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )
        
        return annotated
    
    def process_video(
        self,
        video_path: str,
        output_video: bool = True,
        frame_sample_rate: int = 1,
        output_format: str = "mp4",
        return_base64: bool = False,
        output_dir: str = None,
        output_filename: str = None
    ) -> CountingResult:
        """
        Process video and count persons in each frame
        
        Args:
            video_path: Path to input video
            output_video: Whether to generate annotated output video
            frame_sample_rate: Process every Nth frame (1 = all frames)
            output_format: Output video format
            return_base64: Whether to return video as base64
            output_dir: Custom output directory for video (None = temp dir)
            output_filename: Custom output filename (None = auto-generated)
            
        Returns:
            CountingResult with all counting data
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Video: {width}x{height}, {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        
        # Setup output video writer if needed
        output_path = None
        writer = None
        
        if output_video:
            suffix = f".{output_format}"
            
            # Use custom output directory if provided
            if output_dir:
                # Ensure directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename if not provided
                if output_filename:
                    filename = output_filename if output_filename.endswith(suffix) else f"{output_filename}{suffix}"
                else:
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"person_count_{timestamp}{suffix}"
                
                output_path = os.path.join(output_dir, filename)
            else:
                # Create temporary output file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                output_path = temp_file.name
                temp_file.close()
            
            # Use mp4v codec for MP4
            if output_format.lower() == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif output_format.lower() == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            logger.info(f"Output video will be saved to: {output_path}")
        
        # Process frames
        frame_counts = []
        frame_idx = 0
        last_frame_count = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps if fps > 0 else 0
            
            # Process frame if it matches sample rate
            if frame_idx % frame_sample_rate == 0:
                frame_count = self.detect_persons_in_frame(frame, frame_idx, timestamp)
                frame_counts.append(frame_count)
                last_frame_count = frame_count
                
                if frame_idx % 30 == 0:
                    logger.info(f"Frame {frame_idx}/{total_frames}: {frame_count.person_count} persons")
            else:
                # Use last count for frames we skip
                if last_frame_count:
                    frame_count = FrameCount(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        person_count=last_frame_count.person_count,
                        detections=last_frame_count.detections
                    )
                else:
                    frame_count = FrameCount(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        person_count=0,
                        detections=[]
                    )
            
            # Write annotated frame to output video
            if writer is not None:
                annotated_frame = self.draw_detections_on_frame(frame, frame_count)
                writer.write(annotated_frame)
            
            frame_idx += 1
        
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        
        # Calculate statistics
        counts = [fc.person_count for fc in frame_counts]
        max_persons = max(counts) if counts else 0
        min_persons = min(counts) if counts else 0
        avg_persons = sum(counts) / len(counts) if counts else 0
        
        logger.info(f"Processing complete. Max: {max_persons}, Min: {min_persons}, Avg: {avg_persons:.2f}")
        
        # Convert output video to base64 if requested
        output_base64 = None
        if return_base64 and output_path and os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                output_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        return CountingResult(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            frame_counts=frame_counts,
            max_persons=max_persons,
            min_persons=min_persons,
            avg_persons=avg_persons,
            output_video_path=output_path,
            output_video_base64=output_base64
        )
    
    def get_person_count_timeline(
        self,
        result: CountingResult
    ) -> Dict[str, Any]:
        """
        Generate timeline data for visualization
        
        Args:
            result: CountingResult from process_video
            
        Returns:
            Timeline data dict
        """
        timeline = {
            "timestamps": [],
            "counts": [],
            "max_count": result.max_persons,
            "min_count": result.min_persons,
            "avg_count": result.avg_persons,
            "duration": result.duration
        }
        
        for fc in result.frame_counts:
            timeline["timestamps"].append(fc.timestamp)
            timeline["counts"].append(fc.person_count)
        
        return timeline


# Singleton instance
_yolo_counter = None

def get_yolo_counter(
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.5
) -> Optional[YOLOPersonCounter]:
    """
    Get or create YOLO counter singleton
    
    Args:
        model_name: YOLO model name
        confidence_threshold: Detection confidence threshold
        
    Returns:
        YOLOPersonCounter instance or None if not available
    """
    global _yolo_counter
    
    if not YOLO_AVAILABLE:
        return None
    
    if _yolo_counter is None:
        try:
            _yolo_counter = YOLOPersonCounter(
                model_name=model_name,
                confidence_threshold=confidence_threshold
            )
        except Exception as e:
            logger.error(f"Failed to initialize YOLO counter: {e}")
            return None
    
    return _yolo_counter


if __name__ == "__main__":
    # Test the counter
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python yolo_person_counter.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    counter = YOLOPersonCounter()
    result = counter.process_video(
        video_path,
        output_video=True,
        frame_sample_rate=1
    )
    
    print(f"\n=== Results ===")
    print(f"Total frames: {result.total_frames}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Max persons: {result.max_persons}")
    print(f"Min persons: {result.min_persons}")
    print(f"Avg persons: {result.avg_persons:.2f}")
    print(f"Output video: {result.output_video_path}")
