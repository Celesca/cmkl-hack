import torch
import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Required imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BlipProcessor, BlipForConditionalGeneration

# Optional spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

@dataclass
class Detection:
    timestamp: float
    frame_idx: int
    confidence: float
    blip_description: str
    similarity_scores: Dict[str, float]
    passed: bool

@dataclass
class ActionSegment:
    start_time: float
    end_time: float
    confidence: float
    frame_count: int
    action_label: str
    detections: List[Detection]

class ActionDetector:
    def __init__(self, person_weight=0.2, action_weight=0.7, context_weight=0.1,
                 similarity_threshold=0.5, action_threshold=0.4):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Normalize weights
        total = person_weight + action_weight + context_weight
        self.person_weight = person_weight / total
        self.action_weight = action_weight / total
        self.context_weight = context_weight / total
        
        self.similarity_threshold = similarity_threshold
        self.action_threshold = action_threshold
        
        self._init_models()
        print(f"✅ ActionDetector ready - Action weight: {self.action_weight:.2f}")
    
    def _init_models(self):
        # BLIP
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        self.blip_model.to(self.device)
        
        # Similarity
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # NER
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                pass
    
    def extract_action_verb(self, prompt: str) -> str:
        """🎯 Extract main action verb from prompt using NER (no corpus!)"""
        if self.nlp:
            doc = self.nlp(prompt)
            # Find main action verbs
            for token in doc:
                if (token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp', 'xcomp']) or \
                   (token.tag_ in ['VBG'] and token.dep_ in ['amod', 'acl', 'ROOT']):
                    return token.lemma_.lower()
        
        # Fallback: simple pattern matching
        words = prompt.lower().split()
        for word in words:
            if word.endswith('ing') and len(word) > 4:
                return word
        
        return "action"  # Ultimate fallback
    
    def parse_ner(self, text: str) -> Dict[str, List[str]]:
        """Extract NER components."""
        components = {'persons': [], 'actions': [], 'contexts': []}
        
        if self.nlp:
            doc = self.nlp(text)
            for token in doc:
                text_lower = token.text.lower()
                if text_lower in ['person', 'people', 'man', 'woman', 'child', 'someone', 'individual'] or \
                   (token.pos_ == 'NOUN' and token.dep_ == 'nsubj'):
                    components['persons'].append(text_lower)
                elif (token.pos_ == 'VERB' and token.dep_ in ['ROOT', 'ccomp', 'xcomp']) or \
                     (token.tag_ in ['VBG', 'VBN', 'VBD']):
                    components['actions'].append(token.lemma_.lower())
                elif token.pos_ in ['ADJ', 'ADV', 'NOUN'] and len(text_lower) > 2:
                    components['contexts'].append(text_lower)
        else:
            # Fallback
            words = text.lower().split()
            components['persons'] = [w for w in words if w in ['person', 'people', 'man', 'woman', 'child', 'someone']]
            components['actions'] = [w for w in words if (w.endswith('ing') or w.endswith('ed')) and len(w) > 3]
            components['contexts'] = [w for w in words if w not in components['persons'] + components['actions'] and len(w) > 2]
        
        return components
    
    def calculate_similarity(self, comp1: List[str], comp2: List[str]) -> float:
        """Calculate component similarity."""
        if not comp1 or not comp2:
            return 0.0
        text1, text2 = ' '.join(comp1), ' '.join(comp2)
        embeddings = self.similarity_model.encode([text1, text2])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    
    def weighted_similarity(self, prompt: str, blip_desc: str) -> Dict[str, float]:
        """Calculate NER-weighted similarity."""
        prompt_ner = self.parse_ner(prompt)
        blip_ner = self.parse_ner(blip_desc)
        
        person_sim = self.calculate_similarity(prompt_ner['persons'], blip_ner['persons'])
        action_sim = self.calculate_similarity(prompt_ner['actions'], blip_ner['actions'])
        context_sim = self.calculate_similarity(prompt_ner['contexts'], blip_ner['contexts'])
        
        weighted = (self.person_weight * person_sim + 
                   self.action_weight * action_sim + 
                   self.context_weight * context_sim)
        
        return {'person': person_sim, 'action': action_sim, 'context': context_sim, 'weighted': weighted}
    
    def detect_in_frame(self, frame_data: Dict, prompt: str) -> Detection:
        """Detect action in single frame."""
        pil_image = Image.fromarray(frame_data['frame'])
        inputs = self.blip_processor(pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            blip_desc = self.blip_processor.decode(generated_ids[0], skip_special_tokens=True)
        
        scores = self.weighted_similarity(prompt, blip_desc)
        passed = scores['weighted'] >= self.similarity_threshold and scores['action'] >= self.action_threshold
        
        return Detection(
            timestamp=frame_data['timestamp'],
            frame_idx=frame_data['frame_idx'],
            confidence=scores['weighted'],
            blip_description=blip_desc,
            similarity_scores=scores,
            passed=passed
        )
    
    def process_frames_parallel(self, frames: List[Dict], prompt: str) -> List[Detection]:
        """Process frames in parallel batches."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.detect_in_frame, frame, prompt) for frame in frames]
            return [future.result() for future in futures]
    
    def group_segments(self, detections: List[Detection], action_label: str) -> List[ActionSegment]:
        """Group passed detections into segments."""
        passed = [d for d in detections if d.passed]
        if not passed:
            return []
        
        passed.sort(key=lambda x: x.timestamp)
        segments = []
        current_group = []
        
        for detection in passed:
            if current_group and detection.timestamp - current_group[-1].timestamp > 3.0:
                segments.append(self._create_segment(current_group, action_label))
                current_group = []
            current_group.append(detection)
        
        if current_group:
            segments.append(self._create_segment(current_group, action_label))
        
        return segments
    
    def _create_segment(self, detections: List[Detection], action_label: str) -> ActionSegment:
        """Create action segment."""
        return ActionSegment(
            start_time=detections[0].timestamp,
            end_time=detections[-1].timestamp,
            confidence=np.mean([d.confidence for d in detections]),
            frame_count=len(detections),
            action_label=action_label,
            detections=detections
        )
    
    def create_timeline_viz_data(self, frames_data: List[Dict], segments: List[ActionSegment], 
                                prompt: str, job_id: str, video_duration: float) -> Dict:
        """🎬 Create timeline visualization data (without saving files)."""
        if not segments:
            return None
        
        # 🎯 Extract action verb dynamically from prompt
        action_verb = self.extract_action_verb(prompt)
        
        timeline_data = {
            'job_id': job_id,
            'prompt': prompt,
            'action_verb': action_verb,
            'video_duration': video_duration,
            'segments': []
        }
        
        for i, segment in enumerate(segments):
            duration = segment.end_time - segment.start_time
            timeline_data['segments'].append({
                'index': i,
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': duration,
                'confidence': segment.confidence,
                'action_label': action_verb,
                'frame_count': segment.frame_count
            })
        
        print(f"📊 Timeline data created with {len(segments)} segments")
        return timeline_data
    
    def create_timeline_viz(self, frames_data: List[Dict], segments: List[ActionSegment], 
                           prompt: str, output_dir: str, job_id: str, video_duration: float):
        """🎬 Create timeline visualization with dynamic NER labels."""
        if not segments:
            return None
        
        # 🎯 Extract action verb dynamically from prompt
        action_verb = self.extract_action_verb(prompt)
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 6, figure=fig, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.2)
        
        # Timeline
        ax = fig.add_subplot(gs[0, :])
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        
        for i, segment in enumerate(segments):
            duration = segment.end_time - segment.start_time
            ax.barh(0, duration, left=segment.start_time, height=0.6, 
                   color=colors[i], alpha=0.8, edgecolor='black', linewidth=1)
            
            mid_time = segment.start_time + duration / 2
            # 🎯 Use dynamic action verb label
            ax.text(mid_time, 0, f"{action_verb}\n{segment.confidence:.2f}", 
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, video_duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title(f'Action Timeline for prompt: "{prompt}"', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yticks([])
        
        # 🎯 Dynamic legend with extracted verb
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[0], alpha=0.8, label=action_verb)]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Sample frames
        for i, segment in enumerate(segments[:10]):
            if i < 10:  # Max 10 samples
                mid_detection = segment.detections[len(segment.detections)//2]
                frame_data = next((f for f in frames_data if f['frame_idx'] == mid_detection.frame_idx), None)
                
                if frame_data:
                    row, col = 1 + (i // 5), i % 5
                    if row < 3:
                        ax_frame = fig.add_subplot(gs[row, col])
                        ax_frame.imshow(frame_data['frame'])
                        ax_frame.axis('off')
                        
                        # 🎯 Dynamic title with extracted verb
                        title = f"{action_verb}\n{segment.start_time:.1f}-{segment.end_time:.1f}s"
                        ax_frame.set_title(title, fontsize=10, fontweight='bold')
                        
                        for spine in ax_frame.spines.values():
                            spine.set_edgecolor(colors[i])
                            spine.set_linewidth(3)
                            spine.set_visible(True)
        
        # Save
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        viz_file = f"{output_dir}/visualizations/{job_id}_timeline.png"
        plt.savefig(viz_file, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Timeline saved: {viz_file}")
        return viz_file
    
    def process_video(self, video_path: str, prompt: str, save_files: bool = False) -> Dict:
        """🎬 Process video with parallel batching - returns results directly without saving."""
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"🎬 Processing: {video_path}")
        print(f"🎯 Prompt: '{prompt}'")
        print(f"� Save files: {save_files}")
        
        # Extract frames
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        frames_data = []
        for frame_idx in range(0, total_frames, max(1, int(fps))):  # 1 FPS
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames_data.append({
                    'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps
                })
        cap.release()
        
        print(f"📹 Processing {len(frames_data)} frames")
        
        # Process in parallel
        all_detections = self.process_frames_parallel(frames_data, prompt)
        
        # Extract action verb and group segments
        action_verb = self.extract_action_verb(prompt)
        segments = self.group_segments(all_detections, action_verb)
        passed_detections = [d for d in all_detections if d.passed]
        
        # 🎯 Optimized results - only store passed detections
        results = {
            'job_id': job_id,
            'video_path': video_path,
            'prompt': prompt,
            'action_verb': action_verb,  # Store extracted verb
            'timestamp': datetime.now().isoformat(),
            'video_duration': video_duration,
            'stats': {
                'total_frames': len(frames_data),
                'total_detections': len(all_detections),
                'passed_detections': len(passed_detections),
                'success_rate': len(passed_detections) / len(all_detections) * 100,
                'segments_found': len(segments)
            },
            # 🎯 Only store passed detections to reduce JSON size
            'passed_detections_only': [asdict(d) for d in passed_detections],
            'segments': [asdict(s) for s in segments]
        }
        
        # Optional: Create timeline visualization data (in-memory)
        timeline_data = None
        if segments and save_files:
            try:
                timeline_data = self.create_timeline_viz_data(frames_data, segments, prompt, job_id, video_duration)
                results['timeline_data'] = timeline_data
                print(f"📊 Timeline visualization data created")
            except Exception as e:
                print(f"⚠️ Warning: Could not create timeline data: {e}")
        
        # Print summary
        print(f"\n📊 RESULTS")
        print(f"Extracted Action Verb: '{action_verb}'")
        print(f"Passed Detections: {len(passed_detections)}/{len(all_detections)} ({len(passed_detections)/len(all_detections)*100:.1f}%)")
        print(f"Segments Found: {len(segments)}")
        
        if segments:
            for i, segment in enumerate(segments):
                duration = segment.end_time - segment.start_time
                print(f"  {i+1}: {segment.start_time:.1f}-{segment.end_time:.1f}s ({duration:.1f}s) - {segment.confidence:.3f}")
        
        print(f"✅ Video processing completed - returning results directly")
        
        return results


# Main usage - same as before
if __name__ == "__main__":
    # Initialize detector
    detector = ActionDetector(
        person_weight=0.2,   # Low weight
        action_weight=0.7,   # HIGH weight on actions
        context_weight=0.1,  # Low weight
        similarity_threshold=0.5,
        action_threshold=0.4
    )
    
    # Process video
    video_path = "/content/videoplayback.mp4"  # Replace with your video
    prompt = "person running"       # Replace with your action
    
    results = detector.process_video(video_path, prompt)
    
    print(f"✅ Complete!")
    print(f"📄 JSON (optimized): ./results/json/{results['job_id']}_results.json")
    print(f"📊 Timeline: ./results/visualizations/{results['job_id']}_timeline.png")
    print(f"🎯 Detected verb: '{results['action_verb']}'")