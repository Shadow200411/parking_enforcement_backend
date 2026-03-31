from app.services.cache import DetectionCache

def test_cache_blocks_duplicates():
    #Create a fresh cache instance
    cache = DetectionCache()
    plate = "TEST-999"
    parking_id = 1
    confidence = 0.95
    
    should_process = cache.should_process_detection(plate, parking_id, confidence)
    assert should_process == True
    
    cache.mark_as_flagged(plate, parking_id, confidence, was_auto_flagged=True)
    
    should_process_again = cache.should_process_detection(plate, parking_id, confidence)
    
    assert should_process_again == False