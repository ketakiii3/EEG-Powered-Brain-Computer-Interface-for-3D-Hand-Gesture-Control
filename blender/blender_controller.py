import bpy
import os

# --- Configuration ---
# Path to the file where the ML script writes predictions
PREDICTION_FILE = "/path/to/your/project/prediction.txt"

# Mapping IDs to Frame Numbers in Blender Timeline
# Ensure you have animations set at these frames
GESTURE_MAP = {
    0: 1,   # Open
    1: 10,  # Fist/Close
    2: 30,  # Index
    3: 20   # Victory
}

def read_prediction():
    """Reads the latest prediction from the text file."""
    if not os.path.exists(PREDICTION_FILE):
        return None
    try:
        with open(PREDICTION_FILE, 'r') as f:
            content = f.read().strip()
            if content.isdigit():
                return int(content)
    except Exception as e:
        print(f"Error reading file: {e}")
    return None

def update_hand_pose(scene):
    """Handler function run by Blender's timer."""
    pred_label = read_prediction()
    
    if pred_label is not None and pred_label in GESTURE_MAP:
        target_frame = GESTURE_MAP[pred_label]
        # Set the current frame to match the gesture pose
        scene.frame_set(target_frame)
        print(f"Gesture Detected: {pred_label} -> Frame: {target_frame}")

# Register the timer
# This will check the file 10 times per second
bpy.app.timers.register(lambda: update_hand_pose(bpy.context.scene), first_interval=0.1)

print("BCI Hand Controller Started...")