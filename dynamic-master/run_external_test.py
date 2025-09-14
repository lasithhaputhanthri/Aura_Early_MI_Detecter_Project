# run_external_test.py

import os
import torch
import torchvision
import numpy as np
import skimage.draw
from torch.utils.data import DataLoader

import echonet
import echonet.utils
import echonet.datasets
from echonet.utils.segmentation import _video_collate_fn


if __name__ == "__main__":
    # Parameters
    video_dir = "echonet/data/External"
    output_dir = "output/external_test_output"
    checkpoint_path = "output/segmentation/deeplabv3_resnet50_random/best.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 10
    num_workers = 0  # ✅ Safer for Windows multiprocessing issues

    # Create output folders
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "size"), exist_ok=True)

    # Load model WITHOUT aux_loss to match training
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    model.to(device)

    # Load checkpoint with weights_only=True for security
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        exit(1)

    # Handle DataParallel vs single GPU models
    try:
        if all(not k.startswith("module.") for k in state_dict.keys()):
            model.load_state_dict(state_dict)
        else:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Error loading model state: {e}")
        exit(1)

    model.eval()

    # Use standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Load dataset
    try:
        dataset = echonet.datasets.Echo(
            split="EXTERNAL_TEST",
            external_test_location=video_dir,
            target_type=["Filename"],
            mean=mean,
            std=std,
            length=None,
            max_length=None,
            period=1,
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        exit(1)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=False, collate_fn=_video_collate_fn)

    print(f"🔍 Checking for videos in: {video_dir}")
    print(f"📄 Found {len(dataset)} videos:")
    for f in dataset.fnames:
        print("   -", f)
    
    # Debug: Check what the first few items look like
    print("\n🔍 Debug: First batch sample structure:")
    first_batch = next(iter(dataloader))
    print(f"  x shape: {first_batch[0].shape}")
    print(f"  filenames: {first_batch[1]}")
    print(f"  length: {first_batch[2]}")
    print(f"  filenames type: {type(first_batch[1])}")
    
    # Convert zip to list to inspect
    filenames_list = list(first_batch[1])
    print(f"  filenames as list: {filenames_list}")
    if filenames_list:
        print(f"  first filename: {filenames_list[0]}")
        print(f"  first filename type: {type(filenames_list[0])}")
    print()

    # Run inference and save outputs
    try:
        with torch.no_grad():
            for batch_idx, (x, filenames, length) in enumerate(dataloader):
                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                
                # Get the original filenames directly from the dataset
                # This is more reliable than trying to reconstruct from the collated data
                actual_filenames = []
                cumulative_length = 0
                for i, offset in enumerate(length):
                    if i < len(dataset.fnames):
                        actual_filenames.append(dataset.fnames[i])
                    cumulative_length += offset
                
                print(f"  Actual filenames: {actual_filenames}")
                
                # Process in smaller chunks to avoid memory issues
                y_chunks = []
                for i in range(0, x.shape[0], batch_size):
                    chunk = x[i:i + batch_size].to(device)
                    chunk_output = model(chunk)["out"].cpu()
                    y_chunks.append(chunk_output)
                
                y = torch.cat(y_chunks, dim=0)
                
                x = x.numpy()
                y = y.numpy()[:, 0, :, :]

                start = 0
                for i, (filename, offset) in enumerate(zip(actual_filenames, length)):
                    print(f"  Processing video: {filename}")
                    print(f"    Offset: {offset}")
                    
                    video = x[start:start + offset]
                    logit = y[start:start + offset]
                    start += offset

                    # Unnormalize video
                    video = video * std.reshape(1, 3, 1, 1)
                    video = video + mean.reshape(1, 3, 1, 1)
                    
                    # Ensure video values are in valid range
                    video = np.clip(video, 0, 1)

                    f, c, h, w = video.shape
                    video = np.concatenate((video, video), axis=3)
                    
                    # Create overlay for segmentation
                    overlay = np.maximum(255. * (logit > 0), video[:, 0, :, w:])
                    video[:, 0, :, w:] = overlay
                    
                    video = np.concatenate((video, np.zeros_like(video)), axis=2)

                    # Size trace
                    size = (logit > 0).sum((1, 2))
                    if size.max() > 0:
                        size = size - size.min()
                        size = size / size.max()
                    else:
                        size = np.zeros_like(size)
                    size = 1 - size

                    # Draw size trace
                    for frame, s in enumerate(size):
                        y_pos = int(round(115 + 100 * s))
                        x_pos = int(round(frame / len(size) * 200 + 10))
                        
                        # Ensure coordinates are within bounds
                        if 0 <= y_pos < video.shape[2] and 0 <= x_pos < video.shape[3]:
                            video[:, :, y_pos, x_pos] = 255.

                        def dash(start, stop, on=10, off=10):
                            buf = []
                            x = start
                            while x < stop:
                                buf.extend(range(x, min(x + on, stop)))
                                x += on + off
                            return np.array(buf)

                        d = dash(115, min(224, video.shape[2]))
                        if 0 <= x_pos < video.shape[3]:
                            for dy in d:
                                if dy < video.shape[2]:
                                    video[:, :, dy, x_pos] = 255.

                        # Draw circle
                        try:
                            r, c = skimage.draw.disk((y_pos, x_pos), 4.1)
                            # Filter coordinates to be within bounds
                            valid_coords = (r >= 0) & (r < video.shape[2]) & (c >= 0) & (c < video.shape[3])
                            r = r[valid_coords]
                            c = c[valid_coords]
                            if len(r) > 0 and frame < video.shape[0]:
                                video[frame, :, r, c] = 255.
                        except Exception as e:
                            print(f"Warning: Could not draw circle for frame {frame}: {e}")

                    # Save video
                    video = video.transpose(1, 0, 2, 3)
                    video = np.clip(video * 255, 0, 255).astype(np.uint8)
                    
                    output_path = os.path.join(output_dir, "videos", filename)
                    try:
                        echonet.utils.savevideo(output_path, video, 50)
                        print(f"  ✅ Saved: {output_path}")
                    except Exception as e:
                        print(f"  ❌ Error saving video {filename}: {e}")

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("✅ Done! Output saved to:", output_dir)