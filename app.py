import os
import subprocess
import sys

import gradio as gr


def process_video(input_video, camera_height, tilt_deg, fov_deg):
    if not input_video:
        return None

    output_video = os.path.join("outputs", "mono_demo.mp4")
    os.makedirs("outputs", exist_ok=True)

    # Run the demo script via subprocess, passing the interactive slider settings
    cmd = [
        sys.executable,
        "scripts/run_mono_demo.py",
        "--source",
        input_video,
        "--save-video",
        output_video,
        "--camera-height",
        str(camera_height),
        "--tilt-deg",
        str(tilt_deg),
        "--fov-deg",
        str(fov_deg),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")

    try:
        result = subprocess.run(
            cmd, env=env, check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise Exception(
                f"Command failed with code {result.returncode}:\n{result.stderr}"
            )
        if os.path.exists(output_video):
            return output_video
        else:
            raise Exception("Output video not found after processing.")
    except Exception as e:
        error_msg = f"Error processing video: {e}"
        print(error_msg)
        raise gr.Error(error_msg)


with gr.Blocks(title="3D BBox Tracker Demo") as demo:
    gr.Markdown("# 3D Bounding Box Tracker (Monocular Demo)")
    gr.Markdown(
        "Upload a video and calibrate the camera settings below to see the 3D bounding box tracking in action."
    )

    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input Video")

            gr.Markdown("### Camera Geometry Calibration Settings")
            camera_height_slider = gr.Slider(
                minimum=1.0,
                maximum=15.0,
                value=6.0,
                step=0.1,
                label="Camera Height (meters) [Height of camera above the road]",
            )
            tilt_slider = gr.Slider(
                minimum=0.0,
                maximum=45.0,
                value=16.0,
                step=0.5,
                label="Camera Tilt Angle (degrees) [Downward tilt relative to horizontal]",
            )
            fov_slider = gr.Slider(
                minimum=30.0,
                maximum=120.0,
                value=55.0,
                step=1.0,
                label="Horizontal Field of View (degrees)",
            )

            process_btn = gr.Button("Process Video", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Processed Video")

    process_btn.click(
        fn=process_video,
        inputs=[video_in, camera_height_slider, tilt_slider, fov_slider],
        outputs=video_out,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
