# Cricket-Ball-Tracking

An AI-powered computer vision system for tracking cricket balls, detecting pitches, and analyzing delivery physics in real-time. This project uses YOLO models to simulate a Hawk-Eye-like experience, analyzing ball trajectories and identifying pitch types, swing, spin, and more.

## Features

- **Ball Tracking**: Detects and tracks the cricket ball frame-by-frame using a custom-trained YOLO model (`best.pt`).
- **Pitch Detection**: Automatically identifies the cricket pitch area using another YOLO model (`pitch.pt`).
- **Physics Engine**: Calculates ball dynamics from the 2D tracking points, classifying the delivery:
  - Bounce detection
  - Swing amount
  - Spin / Curvature calculation
  - Speed approximation
  - Delivery classification (e.g., Fast Ball, Swing Ball, Spin Bouncer).
- **TV-Style Trail**: Generates an augmented reality trail behind the ball, similar to professional broadcasting tools like Hawk-Eye.
- **LBW Analysis Integration**: Analyzes Leg Before Wicket (LBW) scenarios when triggered (press 'l' during execution).

## Requirements

- Python 3.8+
- Ultralytics YOLO (`ultralytics`)
- OpenCV (`opencv-python`)
- NumPy (`numpy`)

## Usage

1. Ensure you have your trained YOLO weights inside the `detect/train/weights/` directory (`best.pt` for the ball, `pitch.pt` for the pitch).
2. Place your video file (e.g., `ball_4.mp4`) in the root directory.
3. Run the main script:
   ```bash
   python main.py
   ```
4. **Controls**:
   - `l`: Trigger LBW Analysis (requires sufficient tracking data).
   - `q`: Quit the application.

## Output
The script will display the tracking system in real-time and save an annotated video output named `output.mp4` in the project root.
