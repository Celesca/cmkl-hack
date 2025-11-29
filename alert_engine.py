"""
Smart Factory Alert Processing Engine
Processes detection results and generates structured alerts
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from collections import defaultdict


class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    SAFETY = "safety"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    MAINTENANCE = "maintenance"


@dataclass
class ZoneConfig:
    """Configuration for a factory zone"""
    zone_id: str
    zone_name: str
    max_persons: int = 10
    min_persons: int = 0
    restricted: bool = False
    required_ppe: List[str] = field(default_factory=list)  # ["helmet", "vest", "gloves"]
    allowed_actions: List[str] = field(default_factory=lambda: ["working", "walking", "standing"])
    prohibited_actions: List[str] = field(default_factory=lambda: ["running", "fighting", "sleeping"])
    idle_threshold_seconds: int = 300  # 5 minutes
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Alert:
    """Structured alert for factory notification system"""
    alert_id: str
    alert_type: str
    severity: str
    zone_id: str
    camera_id: str
    title: str
    description: str
    person_count: int
    action_detected: Optional[str]
    confidence: float
    detected_at: str
    recommended_action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


@dataclass
class DetectionEvent:
    """Detection event from camera"""
    event_id: str
    camera_id: str
    zone_id: str
    timestamp: str
    person_count: int
    detections: List[Dict[str, Any]]  # Bounding boxes with labels
    actions: List[Dict[str, Any]]  # Action classifications
    raw_response: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AlertEngine:
    """
    Alert Processing Engine for Smart Factory
    
    Processes detection events and generates alerts based on configurable rules
    """
    
    def __init__(self):
        self.zones: Dict[str, ZoneConfig] = {}
        self.alerts: List[Alert] = []
        self.events: List[DetectionEvent] = []
        self.alert_history: Dict[str, List[Alert]] = defaultdict(list)  # zone_id -> alerts
        self.last_action_time: Dict[str, Dict[str, datetime]] = defaultdict(dict)  # zone_id -> {person_id: last_active_time}
        self.alert_cooldown: Dict[str, datetime] = {}  # alert_key -> last_alert_time
        self.cooldown_seconds = 60  # Don't repeat same alert within 60 seconds
        
        # Initialize default factory zones
        self._init_default_zones()
    
    def _init_default_zones(self):
        """Initialize default factory zone configurations"""
        default_zones = [
            ZoneConfig(
                zone_id="assembly_line_1",
                zone_name="Assembly Line 1",
                max_persons=8,
                min_persons=2,
                restricted=False,
                required_ppe=["helmet", "safety_vest"],
                allowed_actions=["working", "walking", "standing", "assembling"],
                prohibited_actions=["running", "fighting", "sleeping"],
                idle_threshold_seconds=300
            ),
            ZoneConfig(
                zone_id="warehouse_a",
                zone_name="Warehouse Section A",
                max_persons=15,
                min_persons=0,
                restricted=False,
                required_ppe=["safety_vest"],
                allowed_actions=["working", "walking", "carrying", "lifting"],
                prohibited_actions=["running", "fighting"],
                idle_threshold_seconds=600
            ),
            ZoneConfig(
                zone_id="machinery_zone",
                zone_name="Heavy Machinery Zone",
                max_persons=3,
                min_persons=0,
                restricted=True,
                required_ppe=["helmet", "safety_vest", "gloves", "safety_glasses"],
                allowed_actions=["working", "walking", "operating"],
                prohibited_actions=["running", "fighting", "unauthorized_access"],
                idle_threshold_seconds=180
            ),
            ZoneConfig(
                zone_id="packaging_area",
                zone_name="Packaging Area",
                max_persons=10,
                min_persons=3,
                restricted=False,
                required_ppe=["safety_vest"],
                allowed_actions=["working", "walking", "packing", "carrying"],
                prohibited_actions=["running", "fighting"],
                idle_threshold_seconds=420
            ),
            ZoneConfig(
                zone_id="quality_control",
                zone_name="Quality Control Station",
                max_persons=4,
                min_persons=1,
                restricted=False,
                required_ppe=[],
                allowed_actions=["working", "inspecting", "standing", "walking"],
                prohibited_actions=["running", "fighting"],
                idle_threshold_seconds=240
            ),
            ZoneConfig(
                zone_id="restricted_storage",
                zone_name="Restricted Storage Area",
                max_persons=2,
                min_persons=0,
                restricted=True,
                required_ppe=["helmet", "safety_vest", "gloves"],
                allowed_actions=["walking", "carrying"],
                prohibited_actions=["running", "unauthorized_access", "loitering"],
                idle_threshold_seconds=120
            )
        ]
        
        for zone in default_zones:
            self.zones[zone.zone_id] = zone
    
    def add_zone(self, zone: ZoneConfig):
        """Add or update a zone configuration"""
        self.zones[zone.zone_id] = zone
    
    def get_zone(self, zone_id: str) -> Optional[ZoneConfig]:
        """Get zone configuration by ID"""
        return self.zones.get(zone_id)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        return f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
    
    def _check_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key in self.alert_cooldown:
            last_time = self.alert_cooldown[alert_key]
            if datetime.now() - last_time < timedelta(seconds=self.cooldown_seconds):
                return True  # Still in cooldown
        return False
    
    def _set_cooldown(self, alert_key: str):
        """Set cooldown for alert"""
        self.alert_cooldown[alert_key] = datetime.now()
    
    def _get_severity_emoji(self, severity: str) -> str:
        """Get emoji for severity level"""
        emojis = {
            "critical": "🚨",
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢"
        }
        return emojis.get(severity, "⚪")
    
    def process_detection(
        self,
        camera_id: str,
        zone_id: str,
        detections: List[Dict[str, Any]],
        actions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[DetectionEvent, List[Alert]]:
        """
        Process detection results and generate alerts
        
        Args:
            camera_id: Camera identifier
            zone_id: Zone identifier
            detections: List of object detections with bounding boxes
            actions: Optional list of action classifications
            
        Returns:
            Tuple of (DetectionEvent, List[Alert])
        """
        timestamp = datetime.now().isoformat()
        
        # Count persons
        person_count = sum(1 for d in detections if d.get("label", "").lower() in ["person", "human", "worker", "man", "woman"])
        
        # Create detection event
        event = DetectionEvent(
            event_id=self._generate_event_id(),
            camera_id=camera_id,
            zone_id=zone_id,
            timestamp=timestamp,
            person_count=person_count,
            detections=detections,
            actions=actions or []
        )
        self.events.append(event)
        
        # Generate alerts based on rules
        alerts = []
        zone = self.zones.get(zone_id)
        
        if zone:
            # Check capacity alerts
            capacity_alerts = self._check_capacity_rules(zone, person_count, camera_id, timestamp)
            alerts.extend(capacity_alerts)
            
            # Check action-based alerts
            if actions:
                action_alerts = self._check_action_rules(zone, actions, person_count, camera_id, timestamp)
                alerts.extend(action_alerts)
            
            # Check PPE compliance (if PPE detection is in detections)
            ppe_alerts = self._check_ppe_rules(zone, detections, person_count, camera_id, timestamp)
            alerts.extend(ppe_alerts)
            
            # Check restricted zone access
            if zone.restricted and person_count > 0:
                restricted_alerts = self._check_restricted_zone(zone, person_count, camera_id, timestamp)
                alerts.extend(restricted_alerts)
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.alert_history[zone_id].append(alert)
        
        return event, alerts
    
    def _check_capacity_rules(
        self,
        zone: ZoneConfig,
        person_count: int,
        camera_id: str,
        timestamp: str
    ) -> List[Alert]:
        """Check capacity-related rules"""
        alerts = []
        
        # Over capacity
        if person_count > zone.max_persons:
            alert_key = f"capacity_over_{zone.zone_id}"
            if not self._check_cooldown(alert_key):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.CAPACITY.value,
                    severity=AlertSeverity.HIGH.value if person_count > zone.max_persons * 1.5 else AlertSeverity.MEDIUM.value,
                    zone_id=zone.zone_id,
                    camera_id=camera_id,
                    title=f"🔴 Overcrowding in {zone.zone_name}",
                    description=f"{person_count} persons detected in {zone.zone_name}, exceeding maximum capacity of {zone.max_persons}. This may cause safety hazards and reduce work efficiency.",
                    person_count=person_count,
                    action_detected=None,
                    confidence=0.95,
                    detected_at=timestamp,
                    recommended_action=f"Redirect {person_count - zone.max_persons} workers to other zones or pause incoming personnel.",
                    metadata={
                        "max_capacity": zone.max_persons,
                        "excess_count": person_count - zone.max_persons
                    }
                )
                alerts.append(alert)
                self._set_cooldown(alert_key)
        
        # Under capacity (potential understaffing)
        if zone.min_persons > 0 and person_count < zone.min_persons:
            alert_key = f"capacity_under_{zone.zone_id}"
            if not self._check_cooldown(alert_key):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.EFFICIENCY.value,
                    severity=AlertSeverity.MEDIUM.value,
                    zone_id=zone.zone_id,
                    camera_id=camera_id,
                    title=f"⚠️ Understaffed: {zone.zone_name}",
                    description=f"Only {person_count} persons detected in {zone.zone_name}, below minimum staffing of {zone.min_persons}. Production efficiency may be impacted.",
                    person_count=person_count,
                    action_detected=None,
                    confidence=0.90,
                    detected_at=timestamp,
                    recommended_action=f"Assign {zone.min_persons - person_count} additional workers to this zone.",
                    metadata={
                        "min_required": zone.min_persons,
                        "shortage_count": zone.min_persons - person_count
                    }
                )
                alerts.append(alert)
                self._set_cooldown(alert_key)
        
        return alerts
    
    def _check_action_rules(
        self,
        zone: ZoneConfig,
        actions: List[Dict[str, Any]],
        person_count: int,
        camera_id: str,
        timestamp: str
    ) -> List[Alert]:
        """Check action-based rules"""
        alerts = []
        
        for action_data in actions:
            action = action_data.get("action", "").lower()
            confidence = action_data.get("confidence", 0.5)
            
            # Check prohibited actions
            if action in [a.lower() for a in zone.prohibited_actions]:
                alert_key = f"prohibited_{action}_{zone.zone_id}"
                if not self._check_cooldown(alert_key):
                    # Determine severity based on action
                    if action in ["running", "fighting"]:
                        severity = AlertSeverity.CRITICAL.value
                        emoji = "🚨"
                    elif action in ["sleeping", "unauthorized_access"]:
                        severity = AlertSeverity.HIGH.value
                        emoji = "🔴"
                    else:
                        severity = AlertSeverity.MEDIUM.value
                        emoji = "⚠️"
                    
                    alert = Alert(
                        alert_id=self._generate_alert_id(),
                        alert_type=AlertType.SAFETY.value if action in ["running", "fighting"] else AlertType.COMPLIANCE.value,
                        severity=severity,
                        zone_id=zone.zone_id,
                        camera_id=camera_id,
                        title=f"{emoji} Prohibited Action: {action.title()} in {zone.zone_name}",
                        description=f"A person was detected {action} in {zone.zone_name} at {timestamp}. This action is prohibited in this zone for safety reasons.",
                        person_count=person_count,
                        action_detected=action,
                        confidence=confidence,
                        detected_at=timestamp,
                        recommended_action=self._get_action_recommendation(action, zone),
                        metadata={
                            "prohibited_actions": zone.prohibited_actions,
                            "action_confidence": confidence
                        }
                    )
                    alerts.append(alert)
                    self._set_cooldown(alert_key)
            
            # Check idle workers
            if action in ["idle", "standing", "waiting"] and confidence > 0.7:
                # This would need time tracking - simplified version
                alert_key = f"idle_{zone.zone_id}"
                if not self._check_cooldown(alert_key):
                    idle_count = sum(1 for a in actions if a.get("action", "").lower() in ["idle", "standing", "waiting"])
                    if idle_count >= 2:  # Alert if 2+ workers idle
                        alert = Alert(
                            alert_id=self._generate_alert_id(),
                            alert_type=AlertType.EFFICIENCY.value,
                            severity=AlertSeverity.MEDIUM.value,
                            zone_id=zone.zone_id,
                            camera_id=camera_id,
                            title=f"⚠️ Low Activity in {zone.zone_name}",
                            description=f"{idle_count} workers detected idle or standing in {zone.zone_name}. Current staffing: {person_count} persons.",
                            person_count=person_count,
                            action_detected="idle",
                            confidence=confidence,
                            detected_at=timestamp,
                            recommended_action="Investigate potential bottleneck or reassign workers to active tasks.",
                            metadata={
                                "idle_count": idle_count,
                                "idle_threshold_seconds": zone.idle_threshold_seconds
                            }
                        )
                        alerts.append(alert)
                        self._set_cooldown(alert_key)
        
        return alerts
    
    def _check_ppe_rules(
        self,
        zone: ZoneConfig,
        detections: List[Dict[str, Any]],
        person_count: int,
        camera_id: str,
        timestamp: str
    ) -> List[Alert]:
        """Check PPE compliance rules"""
        alerts = []
        
        if not zone.required_ppe or person_count == 0:
            return alerts
        
        # Check for detected PPE items
        detected_ppe = set()
        for det in detections:
            label = det.get("label", "").lower()
            if label in ["helmet", "hard_hat", "safety_helmet"]:
                detected_ppe.add("helmet")
            elif label in ["vest", "safety_vest", "high_vis"]:
                detected_ppe.add("safety_vest")
            elif label in ["gloves", "safety_gloves"]:
                detected_ppe.add("gloves")
            elif label in ["glasses", "safety_glasses", "goggles"]:
                detected_ppe.add("safety_glasses")
        
        # Check for missing PPE
        missing_ppe = set(zone.required_ppe) - detected_ppe
        
        if missing_ppe and person_count > 0:
            alert_key = f"ppe_missing_{zone.zone_id}"
            if not self._check_cooldown(alert_key):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.COMPLIANCE.value,
                    severity=AlertSeverity.HIGH.value,
                    zone_id=zone.zone_id,
                    camera_id=camera_id,
                    title=f"🔴 PPE Violation in {zone.zone_name}",
                    description=f"Missing required PPE detected in {zone.zone_name}: {', '.join(missing_ppe)}. {person_count} person(s) may not be properly equipped.",
                    person_count=person_count,
                    action_detected="ppe_violation",
                    confidence=0.85,
                    detected_at=timestamp,
                    recommended_action=f"Ensure all personnel in {zone.zone_name} are equipped with: {', '.join(zone.required_ppe)}",
                    metadata={
                        "required_ppe": zone.required_ppe,
                        "detected_ppe": list(detected_ppe),
                        "missing_ppe": list(missing_ppe)
                    }
                )
                alerts.append(alert)
                self._set_cooldown(alert_key)
        
        return alerts
    
    def _check_restricted_zone(
        self,
        zone: ZoneConfig,
        person_count: int,
        camera_id: str,
        timestamp: str
    ) -> List[Alert]:
        """Check restricted zone access"""
        alerts = []
        
        alert_key = f"restricted_access_{zone.zone_id}"
        if not self._check_cooldown(alert_key):
            alert = Alert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.COMPLIANCE.value,
                severity=AlertSeverity.HIGH.value if person_count > zone.max_persons else AlertSeverity.MEDIUM.value,
                zone_id=zone.zone_id,
                camera_id=camera_id,
                title=f"🔐 Restricted Zone Access: {zone.zone_name}",
                description=f"{person_count} person(s) detected in restricted zone {zone.zone_name}. Verify authorization status.",
                person_count=person_count,
                action_detected="zone_access",
                confidence=0.95,
                detected_at=timestamp,
                recommended_action="Verify personnel authorization. Contact security if unauthorized access.",
                metadata={
                    "restricted": True,
                    "max_authorized": zone.max_persons
                }
            )
            alerts.append(alert)
            self._set_cooldown(alert_key)
        
        return alerts
    
    def _get_action_recommendation(self, action: str, zone: ZoneConfig) -> str:
        """Get recommended action for prohibited behavior"""
        recommendations = {
            "running": "Stop all nearby machinery immediately. Verify worker safety and investigate cause.",
            "fighting": "Alert security immediately. Separate involved personnel and document incident.",
            "sleeping": "Wake worker and assess fitness for duty. Consider fatigue management review.",
            "unauthorized_access": "Contact security. Verify credentials and escort from restricted area.",
            "loitering": "Direct personnel to assigned work area. Review task assignments."
        }
        return recommendations.get(action, f"Address prohibited action ({action}) according to safety protocols.")
    
    def get_alerts(
        self,
        zone_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        unacknowledged_only: bool = False
    ) -> List[Alert]:
        """Get alerts with optional filtering"""
        filtered = self.alerts.copy()
        
        if zone_id:
            filtered = [a for a in filtered if a.zone_id == zone_id]
        
        if alert_type:
            filtered = [a for a in filtered if a.alert_type == alert_type]
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        
        if unacknowledged_only:
            filtered = [a for a in filtered if a.acknowledged_at is None]
        
        # Sort by timestamp descending (most recent first)
        filtered.sort(key=lambda x: x.detected_at, reverse=True)
        
        return filtered[:limit]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Optional[Alert]:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged_at = datetime.now().isoformat()
                alert.acknowledged_by = acknowledged_by
                return alert
        return None
    
    def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved_at = datetime.now().isoformat()
                return alert
        return None
    
    def get_zone_summary(self, zone_id: str) -> Dict[str, Any]:
        """Get summary of alerts for a zone"""
        zone = self.zones.get(zone_id)
        zone_alerts = self.alert_history.get(zone_id, [])
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in zone_alerts:
            severity_counts[alert.severity] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in zone_alerts:
            type_counts[alert.alert_type] += 1
        
        # Recent events
        zone_events = [e for e in self.events if e.zone_id == zone_id]
        recent_person_counts = [e.person_count for e in zone_events[-10:]]
        
        return {
            "zone_id": zone_id,
            "zone_name": zone.zone_name if zone else "Unknown",
            "zone_config": zone.to_dict() if zone else None,
            "total_alerts": len(zone_alerts),
            "unacknowledged_alerts": sum(1 for a in zone_alerts if a.acknowledged_at is None),
            "alerts_by_severity": dict(severity_counts),
            "alerts_by_type": dict(type_counts),
            "total_events": len(zone_events),
            "avg_person_count": sum(recent_person_counts) / len(recent_person_counts) if recent_person_counts else 0,
            "last_event": zone_events[-1].to_dict() if zone_events else None
        }
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """Get overall factory summary"""
        total_alerts = len(self.alerts)
        unacknowledged = sum(1 for a in self.alerts if a.acknowledged_at is None)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert.severity] += 1
        
        # Count by zone
        zone_counts = defaultdict(int)
        for alert in self.alerts:
            zone_counts[alert.zone_id] += 1
        
        return {
            "total_zones": len(self.zones),
            "total_alerts": total_alerts,
            "unacknowledged_alerts": unacknowledged,
            "alerts_by_severity": dict(severity_counts),
            "alerts_by_zone": dict(zone_counts),
            "total_events": len(self.events),
            "zones": {zone_id: self.get_zone_summary(zone_id) for zone_id in self.zones.keys()}
        }


# Global alert engine instance
alert_engine = AlertEngine()


def get_alert_engine() -> AlertEngine:
    """Get the global alert engine instance"""
    return alert_engine
