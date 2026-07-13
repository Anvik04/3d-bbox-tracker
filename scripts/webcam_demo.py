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

    # 2. Initialize Background Subtractor for moving object detection
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=True
    )

    # Virtual Camera Calibration parameters (assumes standard HD/webcam proportions)
    # These will be updated dynamically based on webcam frame size
    focal_length = 600.0
    
    # 3D Box Dimensions for general moving objects (in meters)
    l, w, h = 0.8, 0.8, 0.8

    # Box connection indexes
    # Top face: 0-1, 1-2, 2-3, 3-0
    # Bottom face: 4-5, 5-6, 6-7, 7-4
    # Verticals: 0-4, 1-5, 2-6, 3-7
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Get frame dimensions
        height_img, width_img = frame.shape[:2]
        cx = width_img / 2.0
        cy = height_img / 2.0

        # Apply background subtraction to detect motion
        fg_mask = back_sub.apply(frame)

        # Post-process mask: threshold and dilate to fill holes
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of moving objects
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            # Filter out tiny noise contours
            area = cv2.contourArea(contour)
            if area < 3000:
                continue

            # 2D Bounding Box
            x, y, w_box, h_box = cv2.boundingRect(contour)

            # Estimate depth (Z) based on object height pixels and physical dimensions
            # Z = (focal_length * physical_height) / height_pixels
            z_c = (focal_length * h) / max(h_box, 1)

            # Estimate 3D center in camera coordinates (X_c, Y_c, Z_c)
            # Center of the 2D box in pixels
            u_center = x + w_box / 2.0
            v_center = y + h_box / 2.0
            
            x_c = ((u_center - cx) * z_c) / focal_length
            y_c = ((v_center - cy) * z_c) / focal_length

            # 3D corners in camera coordinates relative to center (X_c, Y_c, Z_c)
            # In camera frame: X is right, Y is down, Z is forward
            dx = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            dy = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
            dz = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

            # Shift corners by 3D center position
            corners_cam = np.column_stack((dx, dy, dz))
            corners_cam[:, 0] += x_c
            corners_cam[:, 1] += y_c
            corners_cam[:, 2] += z_c

            # Project 3D corners back to 2D image plane
            pts_img = np.zeros((8, 2), dtype=int)
            for idx in range(8):
                pt_z = max(corners_cam[idx, 2], 0.1)  # avoid division by zero
                pts_img[idx, 0] = int((corners_cam[idx, 0] * focal_length) / pt_z + cx)
                pts_img[idx, 1] = int((corners_cam[idx, 1] * focal_length) / pt_z + cy)

            # Draw 3D wireframe box (in Red)
            for start, end in connections:
                cv2.line(frame, tuple(pts_img[start]), tuple(pts_img[end]), (0, 0, 255), 2)

            # Draw contact point (in Green) at bottom center of the box
            # Bottom center in camera coordinates is at (x_c, y_c + h/2, z_c)
            bottom_z = max(z_c, 0.1)
            u_bottom = int((x_c * focal_length) / bottom_z + cx)
            v_bottom = int(((y_c + h/2) * focal_length) / bottom_z + cy)
            cv2.circle(frame, (u_bottom, v_bottom), 6, (0, 255, 0), -1)

            # Draw 2D bounding box (optional, thin outline)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (200, 200, 200), 1)
            cv2.putText(
                frame,
                f"Dist: {z_c:.1f}m",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Show visual output
        cv2.imshow("Real-Time 3D Object Detection & Tracking", frame)

        # Exit on 'q'
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
