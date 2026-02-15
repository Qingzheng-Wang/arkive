from typing import Optional, Tuple, Union
import zstandard as zstd

def _compress_text_data(
    text_data: Union[str, bytes],
    compression_level: int = 3
) -> Tuple[bytes, int]:
    """
    Compress text data using zstandard.
    
    Args:
        text_data: Text data as string or bytes
        compression_level: Zstandard compression level (1-22, default 3)
        
    Returns:
        Tuple of (compressed_bytes, original_size)
    """
    if isinstance(text_data, str):
        text_bytes = text_data.encode('utf-8')
    else:
        text_bytes = text_data
    
    original_size = len(text_bytes)
    
    # Compress with zstandard
    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_data = cctx.compress(text_bytes)
    
    return compressed_data, original_size


def _decompress_text_data(compressed_data: bytes) -> str:
    """
    Decompress zstandard compressed text data.
    
    Args:
        compressed_data: Zstandard compressed bytes
        
    Returns:
        Decompressed text as string
    """
    dctx = zstd.ZstdDecompressor()
    decompressed_bytes = dctx.decompress(compressed_data)
    return decompressed_bytes.decode('utf-8')

def _process_single_text_string_static(
    text_data: str, compression_level: int
) -> Optional[Tuple]:
    """
    Process a single text string to compress it with zstandard.
    This function must be at module level to be picklable.
    
    Returns:
        Tuple of (text_file, compressed_data, original_size, compressed_size) or None if failed
    """
    try:
        
        compressed_data, original_size = _compress_text_data(text_data, compression_level)
        compressed_size = len(compressed_data)
        
        return ("", compressed_data, original_size, compressed_size)
    
    except Exception as e:
        print(f"Error processing {text_file}: {e}")
        return None

def _process_single_text_file_static(
    text_file: str, compression_level: int
) -> Optional[Tuple]:
    """
    Process a single text file to compress it with zstandard.
    This function must be at module level to be picklable.
    
    Returns:
        Tuple of (text_file, compressed_data, original_size, compressed_size) or None if failed
    """
    try:
        with open(text_file, 'r', encoding='utf-8') as f:
            text_data = f.read()
        
        compressed_data, original_size = _compress_text_data(text_data, compression_level)
        compressed_size = len(compressed_data)
        
        return (text_file, compressed_data, original_size, compressed_size)
    
    except Exception as e:
        print(f"Error processing {text_file}: {e}")
        return None

def _process_text_from_bytes_static(
    text_item: dict, compression_level: int
) -> Optional[Tuple]:
    """
    Process text from bytes/string (no file I/O needed for input).
    This function must be at module level to be picklable.
    
    Args:
        text_item: Dict with keys:
            - 'text': text content (str or bytes)
            - 'key': unique identifier for this text
        compression_level: Zstandard compression level
        
    Returns:
        Tuple of (key, compressed_data, original_size, compressed_size) or None if failed
    """
    try:
        text_data = text_item['text']
        text_key = text_item['key']
        
        compressed_data, original_size = _compress_text_data(text_data, compression_level)
        compressed_size = len(compressed_data)
        
        return (text_key, compressed_data, original_size, compressed_size)
    
    except Exception as e:
        print(f"Error processing text {text_item.get('key', 'unknown')}: {e}")
        return None