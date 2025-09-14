# import sys
# import os

# # Add dynamic-master to Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dynamic-master')))

# from echonet.utils.segmentation import run as run_segmentation

# def preprocess(*args, **kwargs):
#     print("[INFO] Segmentation started...")

#     # Get absolute paths
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     echonet_data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'dynamic-master', 'echonet', 'data'))
#     segmentation_output_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'dynamic-master', 'output', 'segmentation_output'))

#     # Ensure output dir exists
#     os.makedirs(segmentation_output_dir, exist_ok=True)

#     print(f"[INFO] Data directory: {echonet_data_dir}")
#     print(f"[INFO] Output directory: {segmentation_output_dir}")

#     # Run segmentation without saving video (for speed/debugging)
#     run_segmentation.main([
#         "--pretrained",
#         "--run_test",
#         "--num_epochs", "0",
#         "--data_dir", echonet_data_dir,
#         "--output", segmentation_output_dir,
#     ])

#     print("[INFO] Segmentation complete.")
#     return 1

import sys
import os

# Add dynamic-master to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dynamic-master'))
sys.path.append(project_root)

from echonet.utils.segmentation import run as run_segmentation

def preprocess(*args, **kwargs):
    print("[INFO] Starting segmentation preprocessing...")

    # Store current directory to restore later
    original_dir = os.getcwd()

    try:
        # ✅ Change working directory to dynamic-master
        os.chdir(project_root)
        print(f"[INFO] Changed working directory to: {project_root}")

        # Call the segmentation script
        run_segmentation.main([
            "--pretrained",
            "--run_test",
            "--save_video",
            "--num_epochs", "0",
            "--data_dir", "echonet/data",  # relative to dynamic-master
        ])

    finally:
        # ✅ Restore original directory
        os.chdir(original_dir)
        print(f"[INFO] Returned to original directory: {original_dir}")
        print("[INFO] Segmentation complete.")
        return 1




# def preprocess(*args, **kwargs):
    
#     return 1