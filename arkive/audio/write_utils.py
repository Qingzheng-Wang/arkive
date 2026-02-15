import io
import numpy as np
import os
from pathlib import Path
import soundfile as sf
import subprocess
import tempfile
from typing import Optional, Tuple
from arkive.utils import _get_bit_depth_from_subtype

def _convert_audio_data(
    audio_data: np.ndarray,
    sample_rate: int,
    channels: int,
    target_format: str,
    target_bit_depth: int
) -> Tuple[bytes, int]:
    """
    Convert audio data to target format and bit depth.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate in Hz
        channels: Number of channels
        target_format: Target format ('flac', 'wav', etc.)
        target_bit_depth: Target bit depth (16, 32, or 64)
        
    Returns:
        Tuple of (file_data_bytes, samples_count)
    """
    # Reshape if mono
    if channels == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    # Determine subtype
    if target_bit_depth == 16:
        subtype = 'PCM_16'
    elif target_bit_depth == 32:
        subtype = 'PCM_32'
    elif target_bit_depth == 64:
        subtype = 'DOUBLE'
    else:
        raise ValueError(f"Unsupported target_bit_depth: {target_bit_depth}")
    
    # Convert to target format
    if target_format in ['flac', 'wav']:
        out_buf = io.BytesIO()
        out_buf.name = f'temp.{target_format}'
        sf.write(out_buf, audio_data, sample_rate, 
                subtype=subtype, format=target_format.upper())
        out_buf.seek(0)
        file_data = out_buf.read()
    elif target_format in ['mp3', 'opus']:
        raise NotImplementedError("Not yet supported")
    
    return file_data, len(audio_data)

# Static function for multiprocessing (must be at module level for pickle)
def _process_single_audio_file_static(
    audio_file: str, target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Process a single audio file to convert it to the target format and bit depth.
    This function must be at module level to be picklable.
    
    Returns:
        Tuple of (audio_file, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
    """
    try:
        # Read the file - handle ark format
        if ':' in audio_file and 'ark' in audio_file.lower():
            try:
                import kaldiio
            except ImportError:
                raise ImportError("kaldiio is not installed. Please `pip install kaldiio`")
            sample_rate, audio_data = kaldiio.load_mat(audio_file)
            channels = 1
            orig_bit_depth = 16
            orig_format = 'wav' # write into wav instead of flac, since flac cost more time
            
            buffer = io.BytesIO()
            buffer.name = 'temp.wav'
            sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            orig_file_data = buffer.read()
        else:
            with open(audio_file, 'rb') as f:
                orig_file_data = f.read()
            
            orig_format = Path(audio_file).suffix.lower().lstrip('.')
            
            try:
                audio_data, sample_rate = sf.read(audio_file, dtype='float32')
                info = sf.info(audio_file)
                orig_bit_depth = _get_bit_depth_from_subtype(info.subtype)
            except Exception as e:
                print(f"DEBUG: sf.read failed for {audio_file}: {type(e).__name__}: {e}")
                # Fallback to ffmpeg for formats not supported by soundfile (MP3, OPUS, etc.)
                if orig_format in ['mp3', 'opus', 'm4a', 'aac']:
                    try:
                        audio_data, sample_rate = _read_with_ffmpeg_static(audio_file)
                        orig_bit_depth = 16  # MP3/OPUS converted to 16-bit PCM
                    except Exception as e:
                        print(f"Warning: Could not read {audio_file} with soundfile or ffmpeg, skipping... ({e})")
                        return None
                else:
                    print(f"Warning: Could not read {audio_file} with soundfile, skipping...")
                    return None
            
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
        
        orig_samples = len(audio_data)
        
        # Determine if conversion is needed
        needs_conversion = False
        if target_format is not None:
            if orig_format != target_format or orig_bit_depth != target_bit_depth:
                needs_conversion = True
        
        if needs_conversion:
            # Convert using helper function
            file_data, samples = _convert_audio_data(
                audio_data, sample_rate, channels, target_format, target_bit_depth
            )
            file_format = target_format
            bit_depth = target_bit_depth
        else:
            file_data = orig_file_data
            file_format = orig_format
            bit_depth = orig_bit_depth
            samples = orig_samples
        
        return (audio_file, file_data, sample_rate, channels, samples, file_format, bit_depth)
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return None

def _process_audio_from_bytes_static(
    audio_item: dict, target_format: Optional[str], target_bit_depth: int
) -> Optional[Tuple]:
    """
    Process audio from bytes (no file I/O needed for input).
    This function must be at module level to be picklable.
    
    Args:
        audio_item: Dict with keys:
            - 'bytes': audio file bytes
            - 'key': unique identifier for this audio
            - 'format': original format (e.g., 'mp3', 'flac', 'wav')
        target_format: Target format for conversion
        target_bit_depth: Target bit depth
        
    Returns:
        Tuple of (key, file_data, sample_rate, channels, samples, format, bit_depth) or None if failed
    """
    try:
        audio_bytes = audio_item['bytes']
        audio_key = audio_item['key']
        orig_format = audio_item['format']
        
        # Create BytesIO buffer
        orig_buffer = io.BytesIO(audio_bytes)
        orig_buffer.name = f'temp.{orig_format}'  # soundfile needs this to infer format
        
        # Read audio from memory
        try:
            audio_data, sample_rate = sf.read(orig_buffer, dtype='float32')
            orig_buffer.seek(0)
            info = sf.info(orig_buffer)
            orig_bit_depth = _get_bit_depth_from_subtype(info.subtype)
        except Exception as e:
            print(f"Warning: Could not read audio {audio_key}: {e}")
            return None
        
        if audio_data.ndim == 1:
            channels = 1
        else:
            channels = audio_data.shape[1]
        
        orig_samples = len(audio_data)
        
        # Determine if conversion is needed
        needs_conversion = False
        if target_format is not None:
            if orig_format != target_format or orig_bit_depth != target_bit_depth:
                needs_conversion = True
        
        if needs_conversion:
            # Convert using helper function
            file_data, samples = _convert_audio_data(
                audio_data, sample_rate, channels, target_format, target_bit_depth
            )
            file_format = target_format
            bit_depth = target_bit_depth
        else:
            # No conversion needed, use original bytes
            file_data = audio_bytes
            file_format = orig_format
            bit_depth = orig_bit_depth
            samples = orig_samples
        
        return (audio_key, file_data, sample_rate, channels, samples, file_format, bit_depth)
    
    except Exception as e:
        print(f"Error processing audio {audio_item.get('key', 'unknown')}: {e}")
        return None


def _read_with_ffmpeg_static(audio_file: str) -> Tuple[np.ndarray, int]:
    """
    Read audio file using ffmpeg for formats not supported by soundfile (MP3, OPUS, etc.).
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        audio_file: Path to input audio file
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Get audio info first
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'stream=sample_rate,channels',
        '-of', 'default=noprint_wrappers=1',
        audio_file
    ]
    
    try:
        probe_output = subprocess.check_output(probe_cmd, stderr=subprocess.STDOUT).decode()
        sample_rate = int([line.split('=')[1] for line in probe_output.split('\n') if 'sample_rate' in line][0])
    except:
        sample_rate = 44100  # Default fallback
    
    # Convert to WAV PCM using ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        cmd = [
            'ffmpeg', '-i', audio_file,
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-y', tmp_path,
            '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Read the converted WAV file
        audio_data, actual_sr = sf.read(tmp_path, dtype='float32')
        sample_rate = actual_sr
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    return audio_data, sample_rate