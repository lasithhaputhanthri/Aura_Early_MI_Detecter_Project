from flask import Blueprint, request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from app.utils.preprocessing import preprocess
from app.utils.active_polynomial import findActivePoly
from app.utils.predict import predictMI
import sys
import traceback
import subprocess

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

bp = Blueprint('routes', __name__)

ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

def convert_to_avi(input_path, output_path):
    """Convert video file to AVI format using FFmpeg"""
    try:
        print(f"[INFO] Converting {input_path} to AVI format...")

        # Check if FFmpeg is available
        result = subprocess.run(['ffmpeg', '-version'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise Exception("FFmpeg not found. Please install FFmpeg.")

        # Convert video to AVI with specific frame size and speed
        cmd = [
            'ffmpeg', '-y',
            '-i', input_path,
            '-vf', 'crop=in_h:in_h,scale=112:112',  # Center crop to square, then resize
            '-r', '50',                             # Set frame rate
            '-c:v', 'libx264',                      # Video codec
            '-preset', 'slow',                      # Better compression (slower but better quality)
            '-crf', '18',                           # Constant Rate Factor (lower is better quality)
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]


        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            print(f"[ERROR] FFmpeg conversion failed: {result.stderr}")
            raise Exception(f"Video conversion failed: {result.stderr}")

        print(f"[INFO] Successfully converted to: {output_path}")
        return True

    except subprocess.TimeoutExpired:
        raise Exception("Video conversion timed out (>5 minutes)")
    except FileNotFoundError:
        raise Exception("FFmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html")
    except Exception as e:
        raise Exception(f"Video conversion failed: {str(e)}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/upload', methods=['POST'])
def upload_video():
    print(f"[DEBUG] Request method: {request.method}")
    print(f"[DEBUG] Request files keys: {list(request.files.keys())}")
    print(f"[DEBUG] Base directory: {BASE_DIR}")
    
    # Check if video file is present in request
    if 'video' not in request.files:
        print("[ERROR] 'video' key not found in request")
        return jsonify({'error': 'No video file part'}), 400
    
    file = request.files['video']
    print(f"[DEBUG] Filename: '{file.filename}'")
    
    # Check if file is selected
    if file.filename == '':
        print("[ERROR] No file selected")
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        print(f"[ERROR] File type not allowed: {file.filename}")
        return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    print(f"[DEBUG] File allowed: {allowed_file(file.filename)}")
    
    # Process the file
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        temp_filepath = os.path.join(BASE_DIR, "temp", original_filename)
        final_filename = "0X1A0A263B22CCD966.avi"  # Fixed filename for processing
        final_filepath = os.path.join(BASE_DIR, "dynamic-master", "echonet", "data", "Videos", final_filename)
        preprocessed_filepath=os.path.join(BASE_DIR, "dynamic-master","output","segmentation","deeplabv3_resnet50_pretrained","videos", final_filename)
        
        # Create temp directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
        
        # Remove old files if they exist
        for filepath in [temp_filepath, final_filepath]:
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[INFO] Removed existing file: {filepath}")
        
        # Save the uploaded file temporarily
        try:
            file.save(temp_filepath)
            print(f"[INFO] Saved uploaded file to: {temp_filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save file: {e}")
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Convert to AVI if not already AVI
        try:
            if original_filename.lower().endswith('.avi'):
                # If already AVI, just move it
                os.rename(temp_filepath, final_filepath)
                print(f"[INFO] File is already AVI, moved to: {final_filepath}")
            else:
                # Convert to AVI
                convert_to_avi(temp_filepath, final_filepath)
                # Clean up temporary file
                os.remove(temp_filepath)
                print(f"[INFO] Cleaned up temporary file: {temp_filepath}")
                
        except Exception as e:
            print(f"[ERROR] Video conversion failed: {e}")
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            return jsonify({'error': f'Video conversion failed: {str(e)}'}), 500
        
        # Process the video through the ML pipeline
        try:
            print("[INFO] Starting preprocessing...")
            preprocess(final_filepath)
            print("[INFO] Preprocessing complete, starting active polynomial...")
            findActivePoly()
            print("[INFO] Active polynomial complete, starting prediction...")
            result = predictMI()
            print("[INFO] Prediction complete!")
            return jsonify(result), 200
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            traceback.print_exc()  # Print full stack trace for debugging
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
            
        finally:
            # Clean up the processed file
            if os.path.exists(preprocessed_filepath):
                os.remove(preprocessed_filepath)
                print(f"[INFO] Cleaned up processed file: {preprocessed_filepath}")
    
    # This shouldn't be reached due to earlier validation, but kept for safety
    return jsonify({'error': 'Invalid file type'}), 400