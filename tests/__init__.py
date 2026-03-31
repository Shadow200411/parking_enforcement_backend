from app.services.cache import DetectionCache

def test_cache_blocks_duplicates():
    # 1. ARRANGE: Create a fresh cache instance
    cache = DetectionCache()
    plate = "TEST-999"
    parking_id = 1
    confidence = 0.95
    
    # 2. ACT & ASSERT (First Pass): The cache has never seen this car, so it should return True
    should_process = cache.should_process_detection(plate, parking_id, confidence)
    assert should_process == True
    
    # Simulate the Decision Engine marking it as flagged
    cache.mark_as_flagged(plate, parking_id, confidence, was_auto_flagged=True)
    
    # 3. ACT & ASSERT (Second Pass): The camera detects the exact same car a second later
    should_process_again = cache.should_process_detection(plate, parking_id, confidence)
    
    # It should return False to block the spam!
    assert should_process_again == False