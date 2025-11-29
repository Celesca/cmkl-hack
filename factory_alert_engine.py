"""
Factory Alert Engine
Analyzes video footage and generates structured alerts for smart factory optimization
"""

import cv2
import numpy as np
import requests
import json
import base64
import io
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from PIL import Image


class AlertType(str, Enum):
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    MAINTENANCE = "maintenance"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FactoryAlert:
    """Structured factory alert"""
    alert_id: str
    alert_type: str
    severity: str
    zone_id: str
    title: str
    description: str
    person_count: int
    action_detected: str
    confidence: float
    detected_at: str
    frame_timestamp: float
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete video analysis result"""
    job_id: str
    video_duration: float
    frames_analyzed: int
    total_persons_detected: int
    alerts: List[FactoryAlert]
    summary: Dict[str, Any]
    timestamp: str


class FactoryAlertEngine:
    """
    Factory Alert Engine using Qwen3-VL for action detection
    Analyzes video and generates structured alerts for factory optimization
    """
    
    # Alert rules configuration
    ALERT_RULES = {
        "running": {
            "type": AlertType.SAFETY,
            "severity": AlertSeverity.CRITICAL,
            "title": "🚨 Person Running Detected",
            "recommended_action": "Verify worker safety and check for emergency situation"
        },
        "falling": {
            "type": AlertType.SAFETY,
            "severity": AlertSeverity.CRITICAL,
            "title": "🚨 Person Falling Detected",
            "recommended_action": "Immediate medical attention may be required"
        },
        "idle": {
            "type": AlertType.EFFICIENCY,
            "severity": AlertSeverity.MEDIUM,
            "title": "⚠️ Worker Idle Detected",
            "recommended_action": "Check for bottlenecks or reassign tasks"
        },
        "standing": {
            "type": AlertType.EFFICIENCY,
            "severity": AlertSeverity.LOW,
            "title": "📊 Worker Standing",
            "recommended_action": "Monitor for extended idle periods"
        },
        "sitting": {
            "type": AlertType.EFFICIENCY,
            "severity": AlertSeverity.LOW,
            "title": "📊 Worker Sitting",
            "recommended_action": "Verify if break time or work task"
        },
        "fighting": {
            "type": AlertType.SAFETY,
            "severity": AlertSeverity.CRITICAL,
            "title": "🚨 Physical Altercation Detected",
            "recommended_action": "Security response required immediately"
        },
        "sleeping": {
            "type": AlertType.COMPLIANCE,
            "severity": AlertSeverity.HIGH,
            "title": "⚠️ Worker Sleeping Detected",
            "recommended_action": "Supervisor intervention required"
        },
        "crowded": {
            "type": AlertType.CAPACITY,
            "severity": AlertSeverity.MEDIUM,
            "title": "👥 Area Overcrowded",
            "recommended_action": "Redistribute workers to other zones"
        },
        "no_ppe": {
            "type": AlertType.COMPLIANCE,
            "severity": AlertSeverity.HIGH,
            "title": "⚠️ Missing Safety Equipment",
            "recommended_action": "Ensure worker wears required PPE"
        },
        "working": {
            "type": AlertType.EFFICIENCY,
            "severity": AlertSeverity.LOW,
            "title": "✅ Normal Work Activity",
            "recommended_action": "No action required"
        }
    }
    
    # Keywords to detect actions from model response
    ACTION_KEYWORDS = {
        "running": ["running", "run", "rushing", "sprinting", "jogging"],
        "falling": ["falling", "fall", "fallen", "tripped", "collapsed"],
        "idle": ["idle", "doing nothing", "not working", "inactive", "waiting"],
        "standing": ["standing", "stand", "stationary"],
        "sitting": ["sitting", "sit", "seated"],
        "fighting": ["fighting", "fight", "hitting", "punching", "altercation", "physical conflict"],
        "sleeping": ["sleeping", "sleep", "asleep", "napping", "dozed"],
        "working": ["working", "work", "operating", "assembling", "typing", "lifting", "carrying", "manufacturing"]
    }
    
    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5-vl",
        confidence_threshold: float = 0.5,
        frame_sample_rate: int = 1,
        person_count_threshold: int = 5,
        zone_id: str = "default_zone"
    ):
        """
        Initialize the Factory Alert Engine
        
        Args:
            ollama_url: Ollama API URL
            model_name: Qwen VL model name
            confidence_threshold: Minimum confidence for alerts
            frame_sample_rate: Frames to analyze per second
            person_count_threshold: Max persons before crowding alert
            zone_id: Default zone identifier
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.frame_sample_rate = frame_sample_rate
        self.person_count_threshold = person_count_threshold
        self.zone_id = zone_id
        
        print(f"✅ FactoryAlertEngine initialized")
        print(f"   Model: {self.model_name}")
        print(f"   Ollama URL: {self.ollama_url}")
    
    def _frame_to_base64(self, frame: np.ndarray, max_size: int = 512) -> str:
        """Convert OpenCV frame to base64"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        height, width = rgb_frame.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        pil_image = Image.fromarray(rgb_frame)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _analyze_frame_with_qwen(
        self,
        frame_base64: str,
        frame_idx: int,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Analyze a single frame using Qwen3-VL
        Returns structured analysis with person count and actions
        """
        prompt = """Analyze this factory/workplace image and provide a detailed safety and efficiency assessment.

Respond in JSON format with these exact fields:
{
    "person_count": <number of people visible>,
    "actions": [
        {
            "person_id": <number>,
            "action": "<what the person is doing: working, idle, running, standing, sitting, etc.>",
            "confidence": <0.0 to 1.0>,
            "description": "<brief description of the person and their activity>"
        }
    ],
    "safety_concerns": "<any safety issues observed or 'none'>",
    "overall_description": "<brief description of the scene>"
}

Focus on:
1. Count all people accurately
2. Identify what each person is doing (working, idle, running, standing, sitting, etc.)
3. Note any safety concerns (running, no PPE, dangerous behavior)
4. Be precise and factual

JSON response:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [frame_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '')
                
                # Parse JSON from response
                try:
                    json_str = raw_response
                    if '```json' in json_str:
                        json_str = json_str.split('```json')[1].split('```')[0]
                    elif '```' in json_str:
                        json_str = json_str.split('```')[1].split('```')[0]
                    
                    start_idx = json_str.find('{')
                    end_idx = json_str.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = json_str[start_idx:end_idx]
                    
                    parsed = json.loads(json_str)
                    
                    return {
                        "success": True,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "person_count": parsed.get("person_count", 0),
                        "actions": parsed.get("actions", []),
                        "safety_concerns": parsed.get("safety_concerns", "none"),
                        "overall_description": parsed.get("overall_description", ""),
                        "raw_response": raw_response
                    }
                    
                except (json.JSONDecodeError, ValueError):
                    # Fallback: extract info heuristically
                    return {
                        "success": True,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "person_count": self._extract_person_count(raw_response),
                        "actions": self._extract_actions(raw_response),
                        "safety_concerns": "unknown",
                        "overall_description": raw_response[:200],
                        "raw_response": raw_response
                    }
            else:
                return {
                    "success": False,
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "error": str(e)
            }
    
    def _extract_person_count(self, text: str) -> int:
        """Extract person count from text heuristically"""
        import re
        
        # Look for patterns like "2 people", "three persons", etc.
        patterns = [
            r'(\d+)\s*(?:people|persons|workers|individuals)',
            r'(?:one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:people|persons|workers)',
        ]
        
        word_to_num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                val = match.group(1) if match.group(1).isdigit() else word_to_num.get(match.group(1), 0)
                return int(val)
        
        # Check for "person" singular
        if re.search(r'\b(a person|one person|1 person)\b', text.lower()):
            return 1
        
        return 0
    
    def _extract_actions(self, text: str) -> List[Dict]:
        """Extract actions from text heuristically"""
        actions = []
        text_lower = text.lower()
        
        for action_type, keywords in self.ACTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    actions.append({
                        "person_id": 1,
                        "action": action_type,
                        "confidence": 0.6,
                        "description": f"Detected {action_type} activity"
                    })
                    break
        
        return actions if actions else [{"person_id": 1, "action": "unknown", "confidence": 0.3, "description": "Activity unclear"}]
    
    def _classify_action(self, action_text: str) -> str:
        """Classify action text into standard action type"""
        action_lower = action_text.lower()
        
        for action_type, keywords in self.ACTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in action_lower:
                    return action_type
        
        return "working"  # Default to working if unclear
    
    def _generate_alert(
        self,
        action: str,
        person_count: int,
        confidence: float,
        timestamp: float,
        description: str,
        frame_idx: int
    ) -> Optional[FactoryAlert]:
        """Generate a factory alert based on detected action"""
        
        # Get alert rule for this action
        rule = self.ALERT_RULES.get(action)
        
        if not rule:
            return None
        
        # Skip low-severity alerts if confidence is low
        if confidence < self.confidence_threshold and rule["severity"] != AlertSeverity.CRITICAL:
            return None
        
        alert = FactoryAlert(
            alert_id=str(uuid.uuid4())[:8],
            alert_type=rule["type"].value,
            severity=rule["severity"].value,
            zone_id=self.zone_id,
            title=rule["title"],
            description=description,
            person_count=person_count,
            action_detected=action,
            confidence=round(confidence, 2),
            detected_at=datetime.now().isoformat(),
            frame_timestamp=round(timestamp, 2),
            recommended_action=rule["recommended_action"],
            metadata={
                "frame_idx": frame_idx,
                "model": self.model_name
            }
        )
        
        return alert
    
    def _check_capacity_alert(
        self,
        person_count: int,
        timestamp: float,
        frame_idx: int
    ) -> Optional[FactoryAlert]:
        """Check if area is overcrowded"""
        
        if person_count >= self.person_count_threshold:
            return FactoryAlert(
                alert_id=str(uuid.uuid4())[:8],
                alert_type=AlertType.CAPACITY.value,
                severity=AlertSeverity.MEDIUM.value,
                zone_id=self.zone_id,
                title="👥 Area Overcrowded",
                description=f"Detected {person_count} people in zone, exceeding threshold of {self.person_count_threshold}",
                person_count=person_count,
                action_detected="crowded",
                confidence=0.95,
                detected_at=datetime.now().isoformat(),
                frame_timestamp=round(timestamp, 2),
                recommended_action="Redistribute workers to other zones",
                metadata={
                    "frame_idx": frame_idx,
                    "threshold": self.person_count_threshold
                }
            )
        
        return None
    
    def analyze_video(
        self,
        video_path: str,
        zone_id: Optional[str] = None,
        actions_to_detect: Optional[List[str]] = None
    ) -> AnalysisResult:
        """
        Analyze video and generate factory alerts
        
        Args:
            video_path: Path to video file
            zone_id: Zone identifier for alerts
            actions_to_detect: Specific actions to look for (default: all)
            
        Returns:
            AnalysisResult with all alerts and summary
        """
        if zone_id:
            self.zone_id = zone_id
        
        if actions_to_detect is None:
            actions_to_detect = list(self.ACTION_KEYWORDS.keys())
        
        job_id = f"factory_job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"🏭 Factory Video Analysis")
        print(f"{'='*60}")
        print(f"📁 Video: {video_path}")
        print(f"🏷️  Zone: {self.zone_id}")
        print(f"🆔 Job ID: {job_id}")
        print(f"{'='*60}\n")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        print(f"📹 Video Info:")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {video_duration:.2f}s")
        print(f"   Total frames: {total_frames}")
        
        frame_interval = max(1, int(fps / self.frame_sample_rate))
        
        all_alerts: List[FactoryAlert] = []
        frames_analyzed = 0
        total_persons = 0
        action_counts: Dict[str, int] = {}
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                print(f"   Analyzing frame {frames_analyzed + 1} (t={timestamp:.2f}s)...", end='\r')
                
                # Convert frame and analyze
                frame_base64 = self._frame_to_base64(frame)
                analysis = self._analyze_frame_with_qwen(frame_base64, frame_idx, timestamp)
                
                if analysis.get("success"):
                    person_count = analysis.get("person_count", 0)
                    total_persons += person_count
                    
                    # Check capacity alert
                    capacity_alert = self._check_capacity_alert(person_count, timestamp, frame_idx)
                    if capacity_alert:
                        all_alerts.append(capacity_alert)
                    
                    # Process each detected action
                    for action_data in analysis.get("actions", []):
                        action_text = action_data.get("action", "unknown")
                        action_type = self._classify_action(action_text)
                        confidence = action_data.get("confidence", 0.5)
                        description = action_data.get("description", "")
                        
                        # Count actions
                        action_counts[action_type] = action_counts.get(action_type, 0) + 1
                        
                        # Generate alert for non-normal actions
                        if action_type in actions_to_detect and action_type != "working":
                            alert = self._generate_alert(
                                action=action_type,
                                person_count=person_count,
                                confidence=confidence,
                                timestamp=timestamp,
                                description=description or analysis.get("overall_description", ""),
                                frame_idx=frame_idx
                            )
                            if alert:
                                all_alerts.append(alert)
                
                frames_analyzed += 1
            
            frame_idx += 1
        
        cap.release()
        
        # Deduplicate similar alerts (same type within 2 seconds)
        deduplicated_alerts = self._deduplicate_alerts(all_alerts)
        
        # Generate summary
        summary = {
            "total_frames": total_frames,
            "frames_analyzed": frames_analyzed,
            "average_person_count": round(total_persons / max(frames_analyzed, 1), 2),
            "action_distribution": action_counts,
            "alert_counts": self._count_alerts_by_type(deduplicated_alerts),
            "critical_alerts": sum(1 for a in deduplicated_alerts if a.severity == "critical"),
            "high_alerts": sum(1 for a in deduplicated_alerts if a.severity == "high"),
            "medium_alerts": sum(1 for a in deduplicated_alerts if a.severity == "medium"),
            "low_alerts": sum(1 for a in deduplicated_alerts if a.severity == "low")
        }
        
        print(f"\n\n{'='*60}")
        print(f"📊 ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Frames analyzed: {frames_analyzed}")
        print(f"Total alerts: {len(deduplicated_alerts)}")
        print(f"Critical: {summary['critical_alerts']}, High: {summary['high_alerts']}, Medium: {summary['medium_alerts']}, Low: {summary['low_alerts']}")
        print(f"{'='*60}\n")
        
        return AnalysisResult(
            job_id=job_id,
            video_duration=round(video_duration, 2),
            frames_analyzed=frames_analyzed,
            total_persons_detected=total_persons,
            alerts=deduplicated_alerts,
            summary=summary,
            timestamp=datetime.now().isoformat()
        )
    
    def _deduplicate_alerts(self, alerts: List[FactoryAlert], time_window: float = 2.0) -> List[FactoryAlert]:
        """Remove duplicate alerts within time window"""
        if not alerts:
            return []
        
        # Sort by timestamp
        sorted_alerts = sorted(alerts, key=lambda a: a.frame_timestamp)
        
        deduplicated = [sorted_alerts[0]]
        
        for alert in sorted_alerts[1:]:
            last_alert = deduplicated[-1]
            
            # Check if same type and within time window
            if (alert.alert_type == last_alert.alert_type and 
                alert.action_detected == last_alert.action_detected and
                abs(alert.frame_timestamp - last_alert.frame_timestamp) < time_window):
                # Keep the one with higher confidence
                if alert.confidence > last_alert.confidence:
                    deduplicated[-1] = alert
            else:
                deduplicated.append(alert)
        
        return deduplicated
    
    def _count_alerts_by_type(self, alerts: List[FactoryAlert]) -> Dict[str, int]:
        """Count alerts by type"""
        counts: Dict[str, int] = {}
        for alert in alerts:
            counts[alert.alert_type] = counts.get(alert.alert_type, 0) + 1
        return counts


def main():
    """Example usage"""
    engine = FactoryAlertEngine(
        ollama_url="http://localhost:11434",
        model_name="qwen2.5-vl",
        confidence_threshold=0.5,
        frame_sample_rate=1,
        person_count_threshold=5,
        zone_id="assembly_line_1"
    )
    
    # Test with a video file
    video_path = "test.mp4"
    
    if os.path.exists(video_path):
        result = engine.analyze_video(
            video_path=video_path,
            zone_id="production_floor",
            actions_to_detect=["running", "idle", "falling", "fighting", "sleeping", "working"]
        )
        
        print("\n📋 ALERTS:")
        for alert in result.alerts:
            print(f"\n{alert.title}")
            print(f"   Type: {alert.alert_type}")
            print(f"   Severity: {alert.severity}")
            print(f"   Time: {alert.frame_timestamp}s")
            print(f"   Description: {alert.description}")
        
        # Save to JSON
        output_data = {
            "job_id": result.job_id,
            "video_duration": result.video_duration,
            "frames_analyzed": result.frames_analyzed,
            "total_persons_detected": result.total_persons_detected,
            "summary": result.summary,
            "timestamp": result.timestamp,
            "alerts": [asdict(a) for a in result.alerts]
        }
        
        with open("factory_alerts.json", "w") as f:
            json.dump(output_data, f, indent=2)
        print("\n💾 Saved to factory_alerts.json")
    else:
        print(f"Test video not found: {video_path}")


if __name__ == "__main__":
    main()
