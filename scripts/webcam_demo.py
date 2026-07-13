import cv2
import numpy as np


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

        # Convert to grayscale and blur to reduce high-frequency noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        # 2. Compute absolute difference between current frame and previous frame
        frame_delta = cv2.absdiff(prev_gray, gray)
        
        # Threshold the delta image to find moving pixels
        _, thresh = cv2.threshold(frame_delta, 18, 255, cv2.THRESH_BINARY)

        # Apply dilate/erode (morphological operations) to merge nearby motion regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Find contours of motion regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the largest moving contour
        largest_contour = None
        max_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                largest_contour = contour

        # Minimum motion area (e.g. 1.5% of the frame size) to trigger detection
        min_motion_area = 0.015 * (width_img * height_img)

        # If a significant moving object is found, draw exactly one 3D bounding box around it
        if largest_contour is not None and max_area > min_motion_area:
            # 2D Bounding Box of the moving object
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)

            # Estimate depth (Z) assuming the physical size of a hand (approx 0.2m)
            physical_h = 0.25  # meters
            z_c = (focal_length * physical_h) / max(h_box, 1)
            # Bound depth range to be realistic
            z_c = np.clip(z_c, 0.3, 3.0)

            # Calculate the physical size (l, w, h) of the 3D box based on 2D bounding box size
            l = (w_box * z_c) / focal_length
            h = (h_box * z_c) / focal_length
            w = l  # Assume symmetric depth for the bounding box

            # Estimate 3D center in camera coordinates (X_c, Y_c, Z_c)
            u_center = x + w_box / 2.0
            v_center = y + h_box / 2.0
            
            x_c = ((u_center - cx) * z_c) / focal_length
            y_c = ((v_center - cy) * z_c) / focal_length

            # Compute the 8 corners of the 3D box in camera coordinates
            # X is right, Y is down, Z is forward
            dx = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            dy = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
            dz = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

            corners_cam = np.column_stack((dx, dy, dz))
            corners_cam[:, 0] += x_c
            corners_cam[:, 1] += y_c
            corners_cam[:, 2] += z_c

            # Project corners back to image coordinates
            pts_img = np.zeros((8, 2), dtype=int)
            for idx in range(8):
                pt_z = max(corners_cam[idx, 2], 0.1)
                pts_img[idx, 0] = int((corners_cam[idx, 0] * focal_length) / pt_z + cx)
                pts_img[idx, 1] = int((corners_cam[idx, 1] * focal_length) / pt_z + cy)

            # Draw 3D wireframe box (in Red)
            for start, end in connections:
                # Clip coordinates to image borders to avoid lines wrapping/overflowing
                pt1 = (np.clip(pts_img[start][0], 0, width_img - 1), np.clip(pts_img[start][1], 0, height_img - 1))
                pt2 = (np.clip(pts_img[end][0], 0, width_img - 1), np.clip(pts_img[end][1], 0, height_img - 1))
                cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

            # Draw contact point (green dot) at the bottom center of the box
            bottom_z = max(z_c, 0.1)
            u_bottom = int((x_c * focal_length) / bottom_z + cx)
            v_bottom = int(((y_c + h/2) * focal_length) / bottom_z + cy)
            u_bottom = np.clip(u_bottom, 0, width_img - 1)
            v_bottom = np.clip(v_bottom, 0, height_img - 1)
            cv2.circle(frame, (u_bottom, v_bottom), 6, (0, 255, 0), -1)

            # Annotation text showing distance
            cv2.putText(
                frame,
                f"Tracked Object | Dist: {z_c:.2f}m",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
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
