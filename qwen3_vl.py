"""
Video Action Detection using Qwen3-VL via Ollama
Detects actions (like running) in video and identifies time segments
"""

import cv2
import numpy as np
import base64
import requests
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io


@dataclass
class ActionDetection:
    """Single frame action detection result"""
    timestamp: float
    frame_idx: int
    confidence: float
    description: str
    action_detected: bool
    raw_response: str


@dataclass
class ActionSegment:
    """Continuous segment where action is detected"""
    start_time: float
    end_time: float
    duration: float
    confidence: float
    frame_count: int
    action_label: str
    detections: List[ActionDetection]


class Qwen3VLActionDetector:
    """
    Video Action Detector using Qwen3-VL model via Ollama
    
    Features:
    - Frame extraction from video
    - Action classification per frame using Qwen3-VL
    - Temporal segmentation of action occurrences
    - Confidence scoring
    """
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen3-vl",  # or "qwen2-vl" depending on your Ollama version
        confidence_threshold: float = 0.6,
        frame_sample_rate: int = 1,  # Extract 1 frame per second
        max_workers: int = 4
    ):
        """
        Initialize the Qwen3-VL Action Detector
        
        Args:
            ollama_url: Base URL for Ollama API
            model_name: Name of the Qwen VL model in Ollama
            confidence_threshold: Minimum confidence to consider action detected
            frame_sample_rate: Frames to extract per second
            max_workers: Number of parallel workers for frame processing
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.frame_sample_rate = frame_sample_rate
        self.max_workers = max_workers
        
        # Verify Ollama is running
        self._verify_ollama_connection()
        
        print(f"✅ Qwen3VLActionDetector initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Ollama URL: {self.ollama_url}")
        print(f"   Confidence threshold: {self.confidence_threshold}")
    
    def _verify_ollama_connection(self):
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
            else:
                print(f"⚠️ Warning: Could not verify Ollama models")
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Warning: Cannot connect to Ollama at {self.ollama_url}")
            print("   Make sure Ollama is running: ollama serve")
    
    def _frame_to_base64(self, frame: np.ndarray, max_size: int = 512) -> str:
        """
        Convert OpenCV frame to base64 encoded image
        
        Args:
            frame: OpenCV BGR frame
            max_size: Maximum dimension for resizing (to reduce API payload)
            
        Returns:
            Base64 encoded JPEG image
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed to reduce payload size
        height, width = rgb_frame.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        # Convert to PIL Image and then to base64
        pil_image = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _analyze_frame_with_qwen(
        self,
        frame_base64: str,
        action_prompt: str,
        frame_idx: int,
        timestamp: float
    ) -> ActionDetection:
        """
        Analyze a single frame using Qwen3-VL via Ollama
        
        Args:
            frame_base64: Base64 encoded image
            action_prompt: Action to detect (e.g., "running")
            frame_idx: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            ActionDetection result
        """
        # Construct the prompt for action detection
        prompt = f"""Analyze this image and determine if a person is {action_prompt}.

Respond in JSON format with these fields:
- "action_detected": true or false
- "confidence": number between 0.0 and 1.0
- "description": brief description of what you see

Focus specifically on detecting if someone is {action_prompt}. Be precise and accurate.

JSON response:"""

        try:
            # Call Ollama API with vision model
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [frame_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistent results
                        "num_predict": 200
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                # Parse the JSON response
                try:
                    # Try to extract JSON from the response
                    json_str = raw_response
                    if '```json' in json_str:
                        json_str = json_str.split('```json')[1].split('```')[0]
                    elif '```' in json_str:
                        json_str = json_str.split('```')[1].split('```')[0]
                    
                    # Find JSON object in response
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx]
                    
                    parsed = json.loads(json_str)
                    
                    action_detected = parsed.get('action_detected', False)
                    if isinstance(action_detected, str):
                        action_detected = action_detected.lower() == 'true'
                    
                    confidence = float(parsed.get('confidence', 0.5))
                    description = parsed.get('description', raw_response[:100])
                    
                except (json.JSONDecodeError, ValueError, KeyError):
                    # Fallback: parse response heuristically
                    raw_lower = raw_response.lower()
                    action_detected = (
                        f'{action_prompt.lower()}' in raw_lower and
                        ('yes' in raw_lower or 'true' in raw_lower or 'is ' + action_prompt.lower() in raw_lower)
                    )
                    confidence = 0.7 if action_detected else 0.3
                    description = raw_response[:200]
                
                return ActionDetection(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    confidence=confidence,
                    description=description,
                    action_detected=action_detected and confidence >= self.confidence_threshold,
                    raw_response=raw_response
                )
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return ActionDetection(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    confidence=0.0,
                    description=f"API error: {response.status_code}",
                    action_detected=False,
                    raw_response=""
                )
                
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout analyzing frame {frame_idx}")
            return ActionDetection(
                timestamp=timestamp,
                frame_idx=frame_idx,
                confidence=0.0,
                description="Timeout",
                action_detected=False,
                raw_response=""
            )
        except Exception as e:
            print(f"❌ Error analyzing frame {frame_idx}: {e}")
            return ActionDetection(
                timestamp=timestamp,
                frame_idx=frame_idx,
                confidence=0.0,
                description=str(e),
                action_detected=False,
                raw_response=""
            )
    
    def extract_frames(self, video_path: str) -> List[Dict]:
        """
        Extract frames from video at specified sample rate
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame data dictionaries
        """
        print(f"📹 Extracting frames from: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   FPS: {fps:.2f}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {duration:.2f} seconds")
        
        # Calculate frame interval based on sample rate
        frame_interval = max(1, int(fps / self.frame_sample_rate))
        
        frames_data = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                frames_data.append({
                    'frame': frame.copy(),
                    'frame_idx': frame_idx,
                    'timestamp': timestamp,
                    'base64': self._frame_to_base64(frame)
                })
            
            frame_idx += 1
        
        cap.release()
        
        print(f"   Extracted {len(frames_data)} frames for analysis")
        return frames_data
    
    def _analyze_frame_wrapper(self, args: Tuple) -> ActionDetection:
        """Wrapper for parallel processing"""
        frame_data, action_prompt = args
        return self._analyze_frame_with_qwen(
            frame_data['base64'],
            action_prompt,
            frame_data['frame_idx'],
            frame_data['timestamp']
        )
    
    def analyze_frames(
        self,
        frames_data: List[Dict],
        action_prompt: str,
        parallel: bool = False
    ) -> List[ActionDetection]:
        """
        Analyze all frames for specified action
        
        Args:
            frames_data: List of frame data from extract_frames
            action_prompt: Action to detect (e.g., "running")
            parallel: Whether to process frames in parallel
            
        Returns:
            List of ActionDetection results
        """
        print(f"🔍 Analyzing {len(frames_data)} frames for action: '{action_prompt}'")
        
        detections = []
        
        if parallel and self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                args_list = [(fd, action_prompt) for fd in frames_data]
                detections = list(executor.map(self._analyze_frame_wrapper, args_list))
        else:
            # Sequential processing with progress
            for i, frame_data in enumerate(frames_data):
                print(f"   Processing frame {i+1}/{len(frames_data)} (t={frame_data['timestamp']:.2f}s)", end='\r')
                detection = self._analyze_frame_with_qwen(
                    frame_data['base64'],
                    action_prompt,
                    frame_data['frame_idx'],
                    frame_data['timestamp']
                )
                detections.append(detection)
        
        print(f"\n✅ Analysis complete")
        
        # Count detections
        positive_count = sum(1 for d in detections if d.action_detected)
        print(f"   Action detected in {positive_count}/{len(detections)} frames")
        
        return detections
    
    def group_into_segments(
        self,
        detections: List[ActionDetection],
        action_label: str,
        max_gap_seconds: float = 2.0
    ) -> List[ActionSegment]:
        """
        Group consecutive positive detections into segments
        
        Args:
            detections: List of ActionDetection results
            action_label: Label for the action
            max_gap_seconds: Maximum gap between detections to consider same segment
            
        Returns:
            List of ActionSegment results
        """
        # Filter positive detections
        positive = [d for d in detections if d.action_detected]
        
        if not positive:
            return []
        
        # Sort by timestamp
        positive.sort(key=lambda x: x.timestamp)
        
        segments = []
        current_segment_detections = [positive[0]]
        
        for i in range(1, len(positive)):
            current = positive[i]
            prev = positive[i-1]
            
            # Check if this detection is part of current segment
            if current.timestamp - prev.timestamp <= max_gap_seconds:
                current_segment_detections.append(current)
            else:
                # Save current segment and start new one
                if current_segment_detections:
                    segments.append(self._create_segment(current_segment_detections, action_label))
                current_segment_detections = [current]
        
        # Don't forget the last segment
        if current_segment_detections:
            segments.append(self._create_segment(current_segment_detections, action_label))
        
        return segments
    
    def _create_segment(
        self,
        detections: List[ActionDetection],
        action_label: str
    ) -> ActionSegment:
        """Create an ActionSegment from a list of detections"""
        start_time = detections[0].timestamp
        end_time = detections[-1].timestamp
        
        return ActionSegment(
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            confidence=np.mean([d.confidence for d in detections]),
            frame_count=len(detections),
            action_label=action_label,
            detections=detections
        )
    
    def process_video(
        self,
        video_path: str,
        action_prompt: str = "running",
        parallel: bool = False,
        save_results: bool = False,
        output_dir: str = "./results"
    ) -> Dict:
        """
        Main method to process video and detect actions
        
        Args:
            video_path: Path to video file
            action_prompt: Action to detect (e.g., "running", "walking", "jumping")
            parallel: Whether to process frames in parallel
            save_results: Whether to save results to JSON file
            output_dir: Directory to save results
            
        Returns:
            Dictionary with detection results
        """
        job_id = f"qwen_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"🎬 Video Action Detection using Qwen3-VL")
        print(f"{'='*60}")
        print(f"📁 Video: {video_path}")
        print(f"🎯 Action: {action_prompt}")
        print(f"🆔 Job ID: {job_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract frames
        frames_data = self.extract_frames(video_path)
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        cap.release()
        
        # Step 2: Analyze frames
        detections = self.analyze_frames(frames_data, action_prompt, parallel)
        
        # Step 3: Group into segments
        segments = self.group_into_segments(detections, action_prompt)
        
        # Prepare results
        results = {
            'job_id': job_id,
            'video_path': video_path,
            'action_prompt': action_prompt,
            'timestamp': datetime.now().isoformat(),
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': video_duration
            },
            'stats': {
                'frames_analyzed': len(detections),
                'frames_with_action': sum(1 for d in detections if d.action_detected),
                'segments_found': len(segments),
                'detection_rate': sum(1 for d in detections if d.action_detected) / len(detections) * 100 if detections else 0
            },
            'segments': [asdict(s) for s in segments],
            'all_detections': [asdict(d) for d in detections]
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Video duration: {video_duration:.2f} seconds")
        print(f"Frames analyzed: {len(detections)}")
        print(f"Action detected in: {results['stats']['frames_with_action']} frames ({results['stats']['detection_rate']:.1f}%)")
        print(f"Segments found: {len(segments)}")
        
        if segments:
            print(f"\n🏃 ACTION SEGMENTS ({action_prompt}):")
            print("-" * 40)
            for i, segment in enumerate(segments):
                print(f"  Segment {i+1}:")
                print(f"    ⏱️  Time: {segment.start_time:.2f}s - {segment.end_time:.2f}s")
                print(f"    ⏳ Duration: {segment.duration:.2f}s")
                print(f"    📊 Confidence: {segment.confidence:.2%}")
                print(f"    🎞️  Frames: {segment.frame_count}")
        else:
            print(f"\n❌ No '{action_prompt}' action detected in the video")
        
        print(f"{'='*60}\n")
        
        # Save results if requested
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{job_id}_results.json")
            
            # Remove raw_response from saved detections to reduce file size
            save_results_data = results.copy()
            for det in save_results_data['all_detections']:
                det.pop('raw_response', None)
            for seg in save_results_data['segments']:
                for det in seg.get('detections', []):
                    det.pop('raw_response', None)
            
            with open(output_file, 'w') as f:
                json.dump(save_results_data, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
        return results


def main():
    """Example usage of Qwen3VLActionDetector"""
    
    # Initialize detector
    detector = Qwen3VLActionDetector(
        ollama_url="http://localhost:11434",
        model_name="qwen3-vl",  # Change to your model name
        confidence_threshold=0.5,
        frame_sample_rate=1  # 1 frame per second
    )
    
    # Process video
    video_path = "test.mp4"  # Replace with your video path
    
    if os.path.exists(video_path):
        results = detector.process_video(
            video_path=video_path,
            action_prompt="running",  # Action to detect
            parallel=False,  # Set True for faster processing
            save_results=True
        )
        
        # Access segments
        for segment in results.get('segments', []):
            print(f"Running detected from {segment['start_time']:.2f}s to {segment['end_time']:.2f}s")
    else:
        print(f"Video file not found: {video_path}")
        print("\nExample usage:")
        print("  detector = Qwen3VLActionDetector()")
        print("  results = detector.process_video('video.mp4', action_prompt='running')")


if __name__ == "__main__":
    main()
