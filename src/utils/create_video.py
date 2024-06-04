import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

def create_video(input_folder, output_file, fps=25):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print("Input folder does not exist.")
        return
    ffmpeg_cmd = f"ffmpeg -r {fps} -pattern_type glob -i '{input_folder}/*.png' -c:v libx264 -pix_fmt yuv420p {output_file}"
    os.system(ffmpeg_cmd)


for i in range(10):
    input_folder = os.path.join(root_path, f"/data/thesis_eval/2024-04-18_00-49-51_2_diffusion_model_setup/episode_{int(i):03d}/")
    output_file = os.path.join(root_path, f"/data/thesis_eval/2024-04-18_00-49-51_2_diffusion_model_setup/episode_{int(i):03d}_video.mp4")
    
    create_video(input_folder, output_file)
