import gradio as gr
import subprocess
import os

def process_video(input_video):
    if not input_video:
        return None
    
    output_video = os.path.join("outputs", "mono_demo.mp4")
    os.makedirs("outputs", exist_ok=True)
    
    import sys
    # Run the demo script via subprocess, explicitly using the current Python executable
    cmd = [
        sys.executable, "scripts/run_mono_demo.py",
        "--source", input_video,
        "--save-video", output_video
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(".")
    
    try:
        result = subprocess.run(cmd, env=env, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Command failed with code {result.returncode}:\n{result.stderr}")
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
    gr.Markdown("Upload a video and calibrate the camera angle to see the 3D bounding box tracking in action.")
    
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input Video")
            process_btn = gr.Button("Process Video", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Processed Video")
            
    process_btn.click(
        fn=process_video, 
        inputs=[video_in], 
        outputs=video_out
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
