import cv2
import numpy as np


def merge_boxes(rects, thresh=40):
    """
    Merges overlapping or nearby rectangles to prevent duplicate boxes on the same object.
    rects: list of (x, y, w, h)
    thresh: maximum distance in pixels between boxes to trigger a merge
    """
    if len(rects) == 0:
        return []

    # Iterate and merge close boxes
    merged = []
    for rect in rects:
        x1, y1, w1, h1 = rect
        has_merged = False
        
        for idx, m_rect in enumerate(merged):
            x2, y2, w2, h2 = m_rect
            
            # Compute horizontal and vertical distance between boundaries
            dist_x = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
            dist_y = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))
            
            if dist_x < thresh and dist_y < thresh:
                # Merge boxes by taking their bounding envelope
                nx = min(x1, x2)
                ny = min(y1, y2)
                nw = max(x1 + w1, x2 + w2) - nx
                nh = max(y1 + h1, y2 + h2) - ny
                merged[idx] = (nx, ny, nw, nh)
                has_merged = True
                break
                
        if not has_merged:
            merged.append(rect)
            
    return merged


def main():
    # 1. Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam successfully opened.")
    print("Press 'q' to quit.")

    # Virtual Camera Calibration parameters
    focal_length = 600.0

    # Box connection indexes
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # bottom face
        (0, 4), (1, 5), (2, 6), (3, 7)   # pillars
    ]

    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Mirror the frame for intuitive interactive feedback
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        height_img, width_img = frame.shape[:2]
        cx = width_img / 2.0
        cy = height_img / 2.0

        # Convert to grayscale and blur to reduce high-frequency camera noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        # 2. Compute absolute difference between current frame and previous frame
        frame_delta = cv2.absdiff(prev_gray, gray)
        
        # Threshold the delta image to find moving pixels
        _, thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)

        # Apply dilate/erode (morphological operations) to merge nearby motion regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of motion regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Gather bounding boxes of contours that exceed a small noise threshold (e.g. 800 pixels)
        # This is sensitive enough to capture small/minute movements (like curtains) but ignores pixel jitter.
        raw_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 800:
                raw_boxes.append(cv2.boundingRect(contour))

        # 3. Merge close/overlapping boxes to avoid duplicate overlapping bboxes on single moving objects
        final_boxes = merge_boxes(raw_boxes, thresh=50)

        # Draw 3D bounding boxes for all detected moving objects
        for x, y, w_box, h_box in final_boxes:
            # Estimate depth (Z) assuming a standard physical size of 0.25 meters
            physical_h = 0.25  
            z_c = (focal_length * physical_h) / max(h_box, 1)
            # Bound depth range
            z_c = np.clip(z_c, 0.3, 3.5)

            # Calculate physical dimensions (l, w, h) based on 2D box size and depth
            l = (w_box * z_c) / focal_length
            h = (h_box * z_c) / focal_length
            w = l  # Symmetric depth

            # Estimate 3D center in camera coordinates (X_c, Y_c, Z_c)
            u_center = x + w_box / 2.0
            v_center = y + h_box / 2.0
            
            x_c = ((u_center - cx) * z_c) / focal_length
            y_c = ((v_center - cy) * z_c) / focal_length

            # Compute the 8 corners of the 3D box in camera coordinates (X: right, Y: down, Z: forward)
            dx = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            dy = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
            dz = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

            corners_cam = np.column_stack((dx, dy, dz))
            corners_cam[:, 0] += x_c
            corners_cam[:, 1] += y_c
            corners_cam[:, 2] += z_c

            # Project corners to 2D image coordinates
            pts_img = np.zeros((8, 2), dtype=int)
            for idx in range(8):
                pt_z = max(corners_cam[idx, 2], 0.1)
                pts_img[idx, 0] = int((corners_cam[idx, 0] * focal_length) / pt_z + cx)
                pts_img[idx, 1] = int((corners_cam[idx, 1] * focal_length) / pt_z + cy)

            # Draw 3D wireframe box (in Red)
            for start, end in connections:
                # Clip coordinates to avoid line wraps or visual glitches at screen edges
                pt1 = (np.clip(pts_img[start][0], 0, width_img - 1), np.clip(pts_img[start][1], 0, height_img - 1))
                pt2 = (np.clip(pts_img[end][0], 0, width_img - 1), np.clip(pts_img[end][1], 0, height_img - 1))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Draw contact point (green dot) at bottom center of the box
            bottom_z = max(z_c, 0.1)
            u_bottom = int((x_c * focal_length) / bottom_z + cx)
            v_bottom = int(((y_c + h/2) * focal_length) / bottom_z + cy)
            u_bottom = np.clip(u_bottom, 0, width_img - 1)
            v_bottom = np.clip(v_bottom, 0, height_img - 1)
            cv2.circle(frame, (u_bottom, v_bottom), 6, (0, 255, 0), -1)

            # Label text showing distance
            cv2.putText(
                frame,
                f"Dist: {z_c:.2f}m",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Update previous frame
        prev_gray = gray

        # Show visual output
        cv2.imshow("Real-Time 3D Object Detection & Tracking", frame)

        # Exit on 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
