def extract_comprehensive_features(trace_events):
    features = {}
    
    # Filter by type
    keystrokes = [e for e in trace_events if e['type'] in ['keydown', 'keypress']]
    clicks = [e for e in trace_events if e['type'] == 'click']
    scrolls = [e for e in trace_events if e['type'] == 'scroll']
    navigates = [e for e in trace_events if e['type'] == 'navigate']
    
    # KEYSTROKE FEATURES
    latencies = [e.get('latency', 0) for e in keystrokes if e.get('latency')]
    if latencies:
        features['mean_keystroke_latency'] = np.mean(latencies)
        features['std_keystroke_latency'] = np.std(latencies)
        features['max_keystroke_latency'] = np.max(latencies)
        features['min_keystroke_latency'] = np.min(latencies)
    
    # CLICK FEATURES
    click_latencies = [e.get('latency', 0) for e in clicks if e.get('latency')]
    if click_latencies:
        features['mean_click_interval'] = np.mean(click_latencies)
        features['std_click_interval'] = np.std(click_latencies)
    
    click_positions_x = [e.get('x', 0) for e in clicks]
    if click_positions_x:
        features['click_std_x'] = np.std(click_positions_x)
    
    # SCROLL FEATURES
    scroll_distances = [abs(e.get('scrollX', 0)) + abs(e.get('scrollY', 0)) for e in scrolls]
    if scroll_distances:
        features['mean_scroll_distance'] = np.mean(scroll_distances)
        features['scroll_frequency'] = len(scrolls) / (trace_events[-1]['relativeTime'] / 1000 / 60)  # per minute
    
    # TIMING FEATURES
    features['total_duration'] = trace_events[-1]['relativeTime']
    features['num_actions'] = len(keystrokes) + len(clicks) + len(scrolls)
    
    # ACTION RATIOS
    total_actions = features['num_actions']
    if total_actions > 0:
        features['keystroke_ratio'] = len(keystrokes) / total_actions
        features['click_ratio'] = len(clicks) / total_actions
        features['scroll_ratio'] = len(scrolls) / total_actions
    
    return features