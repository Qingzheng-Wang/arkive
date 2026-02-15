#!/usr/bin/env python3

import av
import io
import numpy as np

from arkive.definitions import AudioRead, VideoRead
from arkive.utils import check_file_type_func

def generic_video_read(video_bytes: bytes, file_type: str, start_time: int, end_time: int) -> VideoRead:
    video_file_like = io.BytesIO(video_bytes)

    container = av.open(video_file_like)

    sample_rate = container.streams.audio[0].rate
    frame_rate = float(container.streams.video[0].average_rate)

    # Extract video frames
    video = []
    for frame in container.decode(video=0):
        video.append(frame.to_ndarray(format='rgb24'))
    video = np.stack(video)

    # Extract audio
    audio = []
    container.seek(0)  # Reset to beginning
    for frame in container.decode(audio=0):
        audio.append(frame.to_ndarray())
    audio = np.concatenate(audio, axis=1).T
    
    container.close()

    if start_time is None:
        start_sample = 0
        start_frame = 0
    else:
        start_sample = start_time * sample_rate
        start_frame  = start_time * frame_rate

    if end_time is None:
        end_sample = audio.shape[0]
        end_frame = video.shape[0]
    else:
        end_sample = end_time * sample_rate
        end_frame = end_time * frame_rate

    audio_data = AudioRead(
        file_type=file_type,
        modality="audio",
        sample_rate=sample_rate,
        array=audio[start_sample : end_sample, :]
    )

    return VideoRead(
        file_type=file_type,
        modality="video",
        audio=audio_data,
        fps=frame_rate,
        array=video[start_frame : end_frame]
    )

def video_read_remote(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
)  -> VideoRead:
    http_headers = {'Range': f'bytes={start_offset}-{start_offset+file_size+1}'}
    response = requests.get(archive_path, headers=http_headers, stream=True)

    # Check if server supports range requests
    if response.status_code not in [200, 206]:
        print("Warning: Server doesn't support range requests.")
        raise Exception(f"Server returned status code {response.status_code}")

    file_type, modality = check_file_type_func(response.content)
    assert modality == "video", f"Expected video modality, got {modality} from {archive_path} at byte {start_offset}"

def video_read_local(
    archive_path: str, 
    start_offset: int, 
    file_size: int, 
    start_time: int = None, 
    end_time: int = None
) -> VideoRead:

