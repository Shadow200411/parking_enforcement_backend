from datetime import datetime, date

class DetectionCache:
    def __init__(self):
        #Dictionary fromat: {"ABC-123:5":{"flagged_at": datetime, "confidence": 0.9, "was_auto_flagged":True}}
        self.cache = {}
        self._last_cleared = datetime.now().date()
        
    def _check_midnight_reset(self):
        """Clears the cache automatically if it's a new day"""
        today = datetime.now().date()
        if today > self._last_cleared:
            self._cache.clear()
            self._last_cleared = today
            
    def should_process_detection(self, plate: str, parking_id: int, confidence: float) -> bool:
        """Returns True if we should process this, False if we should ignore it as a duplicate"""
        self._check_midnight_reset()
        cache_key = f"{plate}:{parking_id}"
        
        #If we haven't seen this car today, we process it.
        if cache_key not in self.cache:
            return True
        
        cached_data = self.cache[cache_key]
        
        #If it was previously flagged for human review (low confidence),
        #but THIS new detection is high confidence, we should process it to auto-flag it.
        if not cached_data["was_auto_flagged"] and confidence >= 0.85:
            return True
        
        #Otherwise, we've already dealt with this car today so we ignore it.
        return False
    
    def mark_as_flagged(self, plate: str, parking_id: int, confidence: float, was_auto_flagged: bool):
        """Save the result to the cache"""
        self._check_midnight_reset()
        cache_key = f"{plate}:{parking_id}"
        self.cache[cache_key] = {
            "flagged_at": datetime.now(),
            "confidence": confidence,
            "was_auto_flagged": was_auto_flagged
        }

#Single instance shared across the app
detection_cache = DetectionCache()