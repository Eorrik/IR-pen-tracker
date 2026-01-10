# App Function Specification

## Overview
This document describes the functionality of the main application (`src/kinect_pen/main.py`). The app serves as the central hub for the Kinect Pen Tracking system, providing calibration, real-time visualization, and data recording capabilities.

## 1. Startup & Calibration Flow

Upon launching the application:

1.  **Check Calibration**: The app checks for the existence of `desk_calibration.json`.
2.  **Scenario A: No Calibration File Found**:
    *   **Prompt**: "No desk calibration found. Please place the ChArUco board on the desk."
    *   **Action**: Automatically enter **Calibration Mode**.
3.  **Scenario B: Calibration File Exists**:
    *   **Prompt**: "Desk calibration found. Do you want to recalibrate?"
    *   **Options**:
        *   `[Y]es`: Enter **Calibration Mode**.
        *   `[N]o` (Default): Load existing calibration and enter **Main App Mode**.

### Calibration Mode
*   **Visual**: Show Color stream (1080P) with ChArUco board detection overlay.
*   **Process**:
    1.  **Detection**: The system detects the ChArUco board in the **Color Camera** frame.
    2.  **Transformation**: The calculated desk plane in Color coordinates is transformed to **IR/Depth Camera** coordinates using the device's factory Color-to-Depth extrinsics.
*   **Interaction**:
    *   **Prompt**: "Press [SPACE] to capture desk plane."
    *   **Action**: User presses **Spacebar** when the board is clearly visible and detection is stable.
*   **Outcome**:
    *   **Success**: Save `desk_calibration.json` (containing the plane in Depth coordinates) and transition to Main App Mode.
    *   **Failure**: Show error message "Board not detected" and retry.

## 2. Main App Mode

The main interface provides a comprehensive view of the tracking status.

### 2.1 Visualization Layout
The window should display the following views (potentially split screen or overlay):

1.  **Camera View (Color)**:
    *   Raw 1080p Color stream (scaled for display).
    *   **Overlay**:
        *   **Tracking Status**: "TRACKING" (Green) / "LOST" (Red).
        *   **FPS**: Current frame rate.
        *   **Recording Status**: "REC" (Red Circle) + Timer when recording.
    
2.  **Desk Top View (XZ Plane)**:
    *   Orthographic projection of the pen onto the desk surface.
    *   **Grid**: 1cm grid lines.
    *   **Elements**:
        *   **Pen Tip**: Current contact point.
        *   **Trace**: Short history of tip positions (trail).

3.  **Desk Side View (ZY Plane)**:
    *   Orthographic projection from the side.
    *   **Elements**:
        *   **Desk Plane**: Horizontal line at Y=0.
        *   **Pen Body**: Line segment representing the pen.
        *   **Hover Height**: Visual indication of distance to desk.

### 2.2 Visualization Elements
*   **Markers**: Visualize the two IR markers (Red/Blue).
*   **Pen Axis**: Line connecting markers.
*   **Predicted Tip**: The calculated physical tip position ( offset read from config file from lowers marker).
    *   **Color Coding**:
        *   **Green**: Hovering.
        *   **Purple**: Touching (Distance < Threshold or Plane Intersection).

## 3. Recording Functionality

### 3.1 UI Controls
*   **Record Button**: A clickable button or hotkey (Spacebar/'R') to toggle recording.
*   **Output Path**: Display the current recording directory (e.g., `recordings/rec_YYYYMMDD_HHMMSS`).

### 3.2 Recorded Data
When recording is active, the system saves:
1.  **Color Video**: `color.avi` (MJPG, 1080p @ 30fps).
2.  **Raw Depth**: `depth/%06d.png` (16-bit PNG).
3.  **Raw IR**: `ir/%06d.png` (16-bit PNG).
4.  **Pen Data**: `pen_data.jsonl` (JSON Lines).
    *   Fields: `frame_id`, `timestamp`, `has_lock`, `tip` (x,y,z), `direction` (x,y,z), `is_touching`.
5.  **Metadata**: `meta.json` (Saved at start).
    *   Intrinsics (Color/Depth).
    *   Extrinsics (Color->Depth).
    *   Desk Plane Equation.

## 4. Technical Architecture

*   **Entry Point**: `src/kinect_pen/main.py`.
*   **Modules**:
    *   `src/kinect_pen/io/kinect_camera.py`: Camera interface.
    *   `src/kinect_pen/algo/pen_tracker.py`: Tracking logic.
    *   `src/kinect_pen/algo/calibration.py`: Calibration logic (Refactored).
    *   `src/kinect_pen/vis/visualizer.py`: Visualization logic (Refactored).
*   **Libraries**: `pyk4a`, `opencv-python`, `numpy`.
