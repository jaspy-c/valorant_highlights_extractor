import cv2
import os
import glob
import subprocess
from difflib import SequenceMatcher
import numpy as np
import easyocr
import time
import argparse
from datetime import timedelta, datetime, timezone

def is_similar(a, b, threshold=0.5):
    """
    Return True if the similarity ratio between strings a and b is at least threshold.
    """
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold

class HighlightDetector:
    def __init__(self, video_path, output_folder="highlights", 
                 kill_roi=(1300, 90, 1900, 130),
                 replay_roi=(1600, 980, 1830, 1080),
                 defusing_roi=(832, 133, 1087, 179),
                 team1_roi=(720, 20, 780, 52),
                 team2_roi=(1140, 20, 1200, 52),
                 score1_roi=(800, 16, 850, 69),
                 score2_roi=(1070, 16, 1120, 69),
                 team1="sen", team2="100t"):
        """Initialize the highlight detector with configuration parameters."""
        self.start_time = time.time()
        self.timings = {}
        
        print("Initializing detector...")
        self.video_path = video_path
        self.output_folder = output_folder
        self.kill_x1, self.kill_y1, self.kill_x2, self.kill_y2 = kill_roi
        self.replay_x1, self.replay_y1, self.replay_x2, self.replay_y2 = replay_roi
        self.defusing_x1, self.defusing_y1, self.defusing_x2, self.defusing_y2 = defusing_roi
        self.team1_roi = team1_roi
        self.team2_roi = team2_roi
        self.score1_roi = score1_roi
        self.score2_roi = score2_roi
        self.team1 = team1.lower()
        self.team2 = team2.lower()
        self.map_transitions = []
        
        self.problematic_ocr_mapping = {
            't1': ['n', 'ii', 'h', 'tl']
            # Add more as needed
        }
        
        # Add a state variable to track if we're in a valid game period
        self.in_game_period = True  # Start assuming we're in a game
        self.map_transitions = []
        
        init_start = time.time()
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.timings['easyocr_init'] = time.time() - init_start
        
        # Get video duration using ffprobe
        duration_start = time.time()
        duration_cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        self.video_duration = float(subprocess.check_output(duration_cmd).decode().strip())
        self.timings['duration_check'] = time.time() - duration_start
        
        # Create necessary folders
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs("frames", exist_ok=True)

    def preprocess_image(self, image):
        """Convert image to grayscale and apply thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY_INV)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def preprocess_score_roi(self, roi, method='default'):
        """
        Apply various preprocessing methods to improve OCR accuracy for score regions.
        Returns multiple versions of the processed image for OCR.
        """
        # Make a copy to avoid modifying the original
        img = roi.copy()
        
        if method == 'default':
            # Basic grayscale
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif method == 'threshold':
            # Binary thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            return thresh
        elif method == 'adaptive':
            # Adaptive thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        elif method == 'otsu':
            # Otsu's thresholding
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == 'canny':
            # Edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.Canny(gray, 100, 200)
        elif method == 'sharpen':
            # Sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(img, -1, kernel)
        elif method == 'dilate':
            # Dilation to enhance text
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,2), np.uint8)
            return cv2.dilate(thresh, kernel, iterations=1)
        elif method == 'erode':
            # Erosion to enhance text
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,2), np.uint8)
            return cv2.erode(thresh, kernel, iterations=1)
        elif method == 'contrast':
            # Increase contrast
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            return img
    
    def quick_get_score_from_roi(self, roi):
        """
        Perform a fast OCR on the score ROI using a single preprocessing method.
        This method is used to decide whether to run the full multi-method OCR.
        """
        # Use a basic threshold for speed.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        results = self.reader.readtext(thresh, allowlist='0123456789')
        for (bbox, text, prob) in results:
            try:
                score = int(text.strip())
                return score
            except ValueError:
                continue
        return 0

    def get_score_from_roi(self, roi, debug_name="score"):
        """
        Extract score from ROI using multiple preprocessing methods for better accuracy.
        Returns the most consistent score detected.
        """
        methods = ['default', 'threshold', 'adaptive', 'otsu', 'contrast', 'sharpen', 'dilate', 'erode']
        
        # Dictionary to count occurrences of each detected score
        score_counts = {}
        original_roi = roi.copy()
        
        # Track preprocessed images for debugging
        debug_images = {}
        
        # Try each preprocessing method
        for method in methods:
            processed_roi = self.preprocess_score_roi(roi, method)
            debug_images[method] = processed_roi.copy()
            
            # Very strict allowlist - only numbers, no letters or symbols
            results = self.reader.readtext(processed_roi, allowlist='0123456789')
            
            for (bbox, text, prob) in results:
                if prob > 0.2:  # Lower threshold to catch more potential matches
                    try:
                        score = int(text.strip())
                        # Only consider reasonable scores in Valorant (0-15 typically)
                        if 0 <= score <= 26:
                            score_counts[score] = score_counts.get(score, 0) + 1
                            # Save this detection for debugging
                            if score not in debug_images:
                                debug_images[f"detected_{score}_{method}"] = processed_roi
                    except ValueError:
                        continue
        
        # Debug save images for verification (commented out)
        # frame_count = int(time.time()) % 10000  # Simple unique identifier
        # for method, img in debug_images.items():
        #     cv2.imwrite(f"frames/ocr_debug_{debug_name}_{method}_{frame_count}.jpg", img)
        
        # No scores detected
        if not score_counts:
            return 0
        
        # Find the most common score
        most_common_score = max(score_counts.items(), key=lambda x: x[1])
        score_value = most_common_score[0]
        occurrences = most_common_score[1]
        
        # Only accept the score if it was detected multiple times
        if occurrences >= 2:
            print(f"Score {score_value} detected {occurrences} times with different methods")
            return score_value
        else:
            # If no consistent detection, try one more attempt with a larger ROI
            h, w = original_roi.shape[:2]
            expanded_roi = cv2.copyMakeBorder(original_roi, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            results = self.reader.readtext(expanded_roi, allowlist='0123456789')
            for (bbox, text, prob) in results:
                if prob > 0.3:
                    try:
                        score = int(text.strip())
                        if 0 <= score <= 20:
                            print(f"Score {score} detected in expanded ROI with prob {prob}")
                            return score
                    except ValueError:
                        continue
                        
            if score_counts:
                most_likely = max(score_counts.items(), key=lambda x: x[1])[0]
                print(f"Using most likely score {most_likely} (low confidence)")
                return most_likely
        
        return 0
    
    def detect_map_end(self, frame):
        """
        Detect if the current frame shows the end of a map by checking score conditions.
        Using an initial quick OCR check to avoid expensive processing until scores are near 13.
        """
        score1_x1, score1_y1, score1_x2, score1_y2 = self.score1_roi
        score2_x1, score2_y1, score2_x2, score2_y2 = self.score2_roi
        
        # Extract both score regions
        score1_roi = frame[score1_y1:score1_y2, score1_x1:score1_x2]
        score2_roi = frame[score2_y1:score2_y2, score2_x1:score2_x2]
        
        # Quick OCR check
        quick_score1 = self.quick_get_score_from_roi(score1_roi)
        quick_score2 = self.quick_get_score_from_roi(score2_roi)
        
        # If both scores are far from a potential map end (e.g., below 10), skip full OCR
        if quick_score1 < 10 and quick_score2 < 10:
            return False, score1_roi, score2_roi

        # Proceed with full OCR if one team is close to 13
        score1 = self.get_score_from_roi(score1_roi, "team1")
        score2 = self.get_score_from_roi(score2_roi, "team2")
        
        if score1 > 0 or score2 > 0:
            print(f"Detected scores: {score1}-{score2}")
        
        # If OCR completely failed, do not proceed
        if score1 == 0 and score2 == 0:
            return False, score1_roi, score2_roi
        
        # Check win conditions:
        if score1 >= 13 and score1 - score2 >= 2:
            print(f"Team 1 wins with score {score1}-{score2}")
            return True, score1_roi, score2_roi
        
        if score2 >= 13 and score2 - score1 >= 2:
            print(f"Team 2 wins with score {score1}-{score2}")
            return True, score1_roi, score2_roi

        # Overtime win conditions
        if score1 >= 13 and score2 >= 12 and score1 - score2 == 2:
            print(f"Team 1 wins in overtime with score {score1}-{score2}")
            return True, score1_roi, score2_roi
        
        if score2 >= 13 and score1 >= 12 and score2 - score1 == 2:
            print(f"Team 2 wins in overtime with score {score1}-{score2}")
            return True, score1_roi, score2_roi
        
        return False, score1_roi, score2_roi
    
    def detect_new_map(self, frame):
        """
        Detect if this is the start of a new map by checking for a 0-0 score.
        A quick OCR check is done first to speed things up.
        """
        score1_x1, score1_y1, score1_x2, score1_y2 = self.score1_roi
        score2_x1, score2_y1, score2_x2, score2_y2 = self.score2_roi
        
        # Extract both score regions
        score1_roi = frame[score1_y1:score1_y2, score1_x1:score1_x2]
        score2_roi = frame[score2_y1:score2_y2, score2_x1:score2_x2]
        
        # Quick check: if either score is nonzero, it's not a new map.
        quick_score1 = self.quick_get_score_from_roi(score1_roi)
        quick_score2 = self.quick_get_score_from_roi(score2_roi)
        
        print(f"New map quick check - Detected scores: {quick_score1}-{quick_score2}")
        if quick_score1 != 0 or quick_score2 != 0:
            return False, score1_roi, score2_roi
        
        # Optionally, you could run the full OCR here if needed.
        return True, score1_roi, score2_roi
    
    def save_map_transitions(self):
        """Save map transition timestamps to a separate file."""
        if not self.map_transitions:
            return
            
        try:
            with open("map_transitions.txt", 'w') as f:
                for timestamp in self.map_transitions:
                    f.write(f"{timestamp:.3f}\n")
            print("Saved map transitions to map_transitions.txt")
        except IOError as e:
            print(f"Error saving map transitions: {e}")

    def is_replay(self, frame):
        """Detect if the current frame is a replay by looking for 'REPLAY' text."""
        replay_roi = frame[self.replay_y1:self.replay_y2, 
                           self.replay_x1:self.replay_x2]
        results = self.reader.readtext(replay_roi)
        for (bbox, text, prob) in results:
            if ('replay' in text.lower() or 'thrifty' in text.lower()) and prob > 0.5:
                return True, replay_roi
        return False, replay_roi
    
    def is_defusing(self, frame):
        """Detect if the current frame is showing a defusing event by looking for 'defusing' text."""
        defusing_roi = frame[self.defusing_y1:self.defusing_y2, self.defusing_x1:self.defusing_x2]
        results = self.reader.readtext(defusing_roi)
        for (bbox, text, prob) in results:
            if 'defusing' in text.lower() and prob > 0.2:
                return True, defusing_roi
        return False, defusing_roi
    
    def is_team_playing(self, frame, roi_coords, team1, team2):
        """
        Check if either team name is present in the ROI.
        First try sequence matching, then fall back to direct mapping for difficult cases.
        """
        x1, y1, x2, y2 = roi_coords
        team_roi = frame[y1:y2, x1:x2]
        results = self.reader.readtext(team_roi)
        
        # Define problematic OCR mappings that fail sequence matching
        problematic_ocr_mapping = {
            't1': ['n', 'ii', 'h', 'tl'],    # OCR results that don't match T1 well
            'eg': ['ec', 'co', 'eo'],         # OCR results that don't match EG well
            'c9': ['cg', 'o9', 'co'],         # OCR results that don't match C9 well
            # Add more as needed
        }
        
        # First try with sequence matching (your original approach)
        for (bbox, text, prob) in results:
            text_lower = text.lower().strip()
            ratio_team1 = SequenceMatcher(None, text_lower, team1).ratio()
            ratio_team2 = SequenceMatcher(None, text_lower, team2).ratio()
            print(f"Comparing OCR text '{text_lower}' with teams '{team1}' (ratio={ratio_team1:.2f}) and '{team2}' (ratio={ratio_team2:.2f}), prob={prob:.2f}")
            
            if (ratio_team1 >= 0.33 or ratio_team2 >= 0.33) and prob > 0.02:
                return True, team_roi
        
        # If sequence matching failed, try the direct mapping approach
        for (bbox, text, prob) in results:
            text_lower = text.lower().strip()
            
            # Check if team1 is in our mapping and the OCR text matches known problematic readings
            if team1 in problematic_ocr_mapping and text_lower in problematic_ocr_mapping[team1]:
                print(f"Matched '{team1}' via problematic OCR mapping with text '{text_lower}'")
                return True, team_roi
                
            # Same for team2
            if team2 in problematic_ocr_mapping and text_lower in problematic_ocr_mapping[team2]:
                print(f"Matched '{team2}' via problematic OCR mapping with text '{text_lower}'")
                return True, team_roi
        
        return False, team_roi

    def extract_frames_and_timestamps(self):
        """Extract frames and detect kills or defusing events, ignoring replay segments and analysis periods."""
        start_time = time.time()
        timestamps = []
        frame_count = 0
        in_replay_segment = False
        replay_cooldown = 0
        map_start_cooldown = 0  # Cooldown for new map (map start) detection
        map_end_cooldown = 0    # Cooldown for detecting map end

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video file: {self.video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps * 1)
            replay_cooldown_frames = int(fps * 3)
            map_cooldown_frames = int(fps * 600)

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                if current_time >= self.video_duration:
                    break

                if frame_count % frame_interval == 0:

                    left_team_present, team1_ = self.is_team_playing(frame, self.team1_roi, self.team1, self.team2)
                    right_team_present, team2_ = self.is_team_playing(frame, self.team2_roi, self.team1, self.team2)

                    if left_team_present and right_team_present:
                        # New Map (Map Start) Detection: only check every 3 frames when cooldown expired.
                        if map_start_cooldown == 0 and frame_count % 3 == 0:
                            is_new_map, score1_roi, score2_roi = self.detect_new_map(frame)
                            # cv2.imwrite(f"frames/score1_{frame_count}_{current_time:.2f}.jpg", score1_roi)
                            # cv2.imwrite(f"frames/score2_{frame_count}_{current_time:.2f}.jpg", score2_roi)
                            if is_new_map:
                                print(f"New map detected at {current_time:.2f}s - Resuming highlight extraction")
                                self.map_transitions.append(current_time)
                                self.save_map_transitions()
                                map_start_cooldown = map_cooldown_frames
                                self.in_game_period = True  # Resume extraction
                                cv2.imwrite(f"frames/map_transition_{frame_count}_{current_time:.2f}.jpg", frame)
                        else:
                            map_start_cooldown = max(0, map_start_cooldown - frame_interval)

                        # Map End Detection: Only when in a game period, and check every 3 frames with cooldown.
                        if self.in_game_period:
                            if map_end_cooldown == 0 and frame_count % 3 == 0:
                                is_map_end, score1_roi, score2_roi = self.detect_map_end(frame)
                                if is_map_end:
                                    print(f"Map end detected at {current_time:.2f}s - Pausing highlight extraction")
                                    cv2.imwrite(f"frames/map_end_{frame_count}_{current_time:.2f}.jpg", frame)
                                    self.in_game_period = False
                                    map_end_cooldown = map_cooldown_frames
                            else:
                                map_end_cooldown = max(0, map_end_cooldown - frame_interval)
                    else:
                        # Decrement cooldowns if teams are not both present.
                        map_start_cooldown = max(0, map_start_cooldown - frame_interval)
                        map_end_cooldown = max(0, map_end_cooldown - frame_interval)

                    # Only process highlights if we're in a valid game period
                    if self.in_game_period and left_team_present and right_team_present:
                        if replay_cooldown == 0:
                            in_replay_segment, _ = self.is_replay(frame)
                            if in_replay_segment:
                                replay_cooldown = replay_cooldown_frames
                        else:
                            replay_cooldown = max(0, replay_cooldown - frame_interval)

                        if not in_replay_segment:
                            kill_roi = frame[self.kill_y1:self.kill_y2, self.kill_x1:self.kill_x2]
                            results_kill = self.reader.readtext(kill_roi)
                            valid_kills = [text for _, text, prob in results_kill if prob > 0.1]

                            defusing_detected, defusing_frame = self.is_defusing(frame)

                            if len(valid_kills) >= 2 or defusing_detected:
                                rounded_time = round(current_time, 2)
                                print(f"Frame {frame_count} at {current_time:.2f}s: Kill texts: {valid_kills}")
                                if defusing_detected:
                                    print("Defusing detected at this frame.")
                                timestamps.append((rounded_time, defusing_detected))

                frame_count += 1

        finally:
            cap.release()

        print(timestamps)
        self.timings['frame_extraction'] = time.time() - start_time
        return timestamps


    def extract_highlight_timestamps(self, timestamps, before_kill=6, after_kill=1.4):
        """Extract highlight clips around timestamps."""
        start_time = time.time()
        highlights = []
        DEFUSE_EXTENSION = -1

        for (timestamp, defusing) in timestamps:
            if not 0 <= timestamp <= self.video_duration:
                print(f"Warning: Skipping invalid timestamp {timestamp}")
                continue
                
            current_start = max(0, timestamp - before_kill)
            effective_after = after_kill + (DEFUSE_EXTENSION if defusing else 0)
            current_end = min(self.video_duration, timestamp + effective_after)
            
            if current_start < current_end:
                highlights.append((current_start, current_end))
            
        self.timings['highlight_extraction'] = time.time() - start_time
        return highlights

    def merge_highlight_timestamps(self, highlights, buffer_time=15):
        """Merge overlapping or close highlight clips and save to timestamps file."""
        if not highlights:
            return []
        
        if buffer_time < 0:
            raise ValueError("buffer_time must be non-negative")
            
        for start, end in highlights:
            if end < start:
                raise ValueError(f"Invalid timestamp: end ({end}) before start ({start})")
        
        highlights.sort(key=lambda x: x[0])
        start_time = time.time()
        merged = []
        current_start, current_end = highlights[0]

        for start, end in highlights[1:]:
            if start - current_end <= buffer_time:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        
        timestamp_file = "highlight_timestamps.txt"
        
        try:
            with open(timestamp_file, 'w') as f:
                for start, end in merged:
                    f.write(f"{start:.3f},{end:.3f}\n")
            print(f"Saved timestamps to {timestamp_file}")
        except IOError as e:
            print(f"Error saving timestamps: {e}")
        
        self.timings['highlight_merging'] = time.time() - start_time
        return merged

    def save_highlights(self, timestamp_file):
        """Save highlight clips using FFmpeg with timestamps from file."""
        if not os.path.exists(timestamp_file):
            raise FileNotFoundError(f"Timestamp file not found: {timestamp_file}")
            
        start_time = time.time()
        clip_times = []
        
        # Load highlight timestamps
        with open(timestamp_file, 'r') as f:
            highlight_times = [tuple(map(float, line.strip().split(','))) for line in f]
        
        # Load map transition timestamps if available
        map_transition_times = []
        if os.path.exists("map_transitions.txt"):
            with open("map_transitions.txt", 'r') as f:
                map_transition_times = [float(line.strip()) for line in f]
            print(f"Loaded {len(map_transition_times)} map transition times: {map_transition_times}")
        
        # Pre-process to identify the last highlight before each map transition
        last_highlights_before_transitions = set()
        
        if map_transition_times:
            # For each map transition, find the highlight that ends most recently before it
            for transition_time in map_transition_times:
                # Filter highlights that end before this transition
                highlights_before_transition = [(i, start, end) for i, (start, end) in enumerate(highlight_times) 
                                            if end < transition_time]
                
                if highlights_before_transition:
                    # Get the highlight with the latest end time (closest to the transition)
                    last_highlight_idx, _, _ = max(highlights_before_transition, key=lambda x: x[2])
                    last_highlights_before_transitions.add(last_highlight_idx)
                    print(f"Identified highlight #{last_highlight_idx+1} as the last before transition at {transition_time}")
        
        total_highlights = len(highlight_times)
        for i, (start, end) in enumerate(highlight_times):
            # Check if this is the last highlight of the entire video
            is_last_overall = (i == total_highlights - 1)  # Use zero-based index
            
            # Check if this is the last highlight before a map transition
            is_last_before_transition = (i in last_highlights_before_transitions)
            
            # Extend the clip if it's either the last overall or last before a map transition
            if is_last_overall:
                # Extended by 4 seconds for the very last highlight
                extended_end = min(self.video_duration, end + 7)
                print(f"Extending last highlight #{i+1} from {end} to {extended_end} (last overall)")
                end = extended_end
            elif is_last_before_transition:
                # Keep the original 3 second extension for map transitions
                extended_end = min(self.video_duration, end + 3)
                print(f"Extending highlight #{i+1} from {end} to {extended_end} (last before map transition)")
                end = extended_end

            current_utc = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(self.output_folder, f"highlight_{current_utc}_{start:.1f}_{end:.1f}.mp4")
            
            try:
                print(f"Debug - Highlight #{i+1}: start={start:.3f}, end={end:.3f}")
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-ss', str(start),
                    '-i', self.video_path,
                    '-t', str(end - start),
                    '-c', 'copy', '-copyinkf',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    output_path
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode != 0 or not os.path.getsize(output_path) > 0:
                    print("Stream copy failed, falling back to re-encode...")
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-ss', str(start),
                        '-i', self.video_path,
                        '-t', str(end - start),
                        '-c:v', 'libx264',
                        '-c:a', 'aac',
                        '-preset', 'slow',
                        '-crf', '23',
                        '-copyts',
                        '-avoid_negative_ts', 'make_zero',
                        '-max_muxing_queue_size', '1024',
                        '-y',
                        output_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                clip_time = time.time() - start_time
                clip_times.append(clip_time)
                print(f"Successfully saved highlight at {current_utc}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error saving highlight at {current_utc}: {str(e)}")
                print(f"FFmpeg stderr: {e.stderr}")
            except Exception as e:
                print(f"Error saving highlight at {current_utc}: {str(e)}")
        
        self.timings['highlight_saving'] = time.time() - start_time
        if clip_times:
            self.timings['average_clip_time'] = sum(clip_times) / len(clip_times)
            
    def print_timings(self):
        """Print detailed timing information."""
        total_time = time.time() - self.start_time
        print("\n=== Performance Report ===")
        print(f"Total execution time: {timedelta(seconds=int(total_time))}")
        print("\nBreakdown:")
        for step, duration in self.timings.items():
            percentage = (duration / total_time) * 100
            print(f"- {step}: {timedelta(seconds=int(duration))} ({percentage:.1f}%)")
        
        if 'average_clip_time' in self.timings:
            print(f"\nAverage time per clip: {timedelta(seconds=int(self.timings['average_clip_time']))}")

    def run(self):
        """Run the complete highlight detection pipeline."""
        try:
            print("Starting highlight detection...")
            print(f"Video duration: {self.video_duration:.2f} seconds")
            
            timestamp_file = "highlight_timestamps.txt"
            if os.path.exists(timestamp_file):
                print(f"Found existing {timestamp_file}, skipping extraction process")
                return [], [], timestamp_file
                
            timestamps = self.extract_frames_and_timestamps()
            print(f"Found {len(timestamps)} potential highlights at: {timestamps}")
            
            if not timestamps:
                print("No highlights detected.")
                return [], []
            
            highlights = self.extract_highlight_timestamps(timestamps)
            print(f"Extracted {len(highlights)} highlight clips")
            
            if not highlights:
                print("Failed to extract any highlight clips.")
                self.print_timings()
                return timestamps, []
            
            merged_highlights = self.merge_highlight_timestamps(highlights)
            print(f"Merged into {len(merged_highlights)} final highlights")
            
            print(f"Timestamps saved to: {timestamp_file}")
            
            self.print_timings()
            return timestamps, merged_highlights, timestamp_file
            
        except Exception as e:
            print(f"Error in highlight detection pipeline: {str(e)}")
            self.print_timings()
            raise

def main():
    parser = argparse.ArgumentParser(description="Highlight Detector for Valorant")
    parser.add_argument('--video', type=str, required=True, help='Path to the video file')
    parser.add_argument('--team1', type=str, required=True, help='Team1 name (e.g., SEN)')
    parser.add_argument('--team2', type=str, required=True, help='Team2 name (e.g., 100T)')
    parser.add_argument('--create-videos', action='store_true', help='Create video highlights from timestamps')
    args = parser.parse_args()
    
    total_start_time = time.time()

    team1 = args.team1.lower()
    team2 = args.team2.lower()
    
    print(team1, team2)

    detector = HighlightDetector(video_path=args.video, team1=team1, team2=team2)
    
    timestamp_file = "highlight_timestamps.txt"
    if os.path.exists(timestamp_file) and args.create_videos:
        print(f"Found existing {timestamp_file}, proceeding directly to highlight creation")
        detector.save_highlights(timestamp_file)
    else:
        _, _, timestamp_file = detector.run()
        if args.create_videos and os.path.exists(timestamp_file):
            print("Creating video highlights from timestamps...")
            detector.save_highlights(timestamp_file)
    
    total_time = time.time() - total_start_time
    print(f"\nTotal script execution time: {timedelta(seconds=int(total_time))}")
    
if __name__=="__main__":
    main()
