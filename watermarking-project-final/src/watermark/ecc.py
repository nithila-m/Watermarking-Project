"""Reed-Solomon layer. Member C owns the final version; this one keeps the same
interface so A can test end to end before C's is merged.

Provides forward error correction (FEC) utilities using an (8, 4) Reed-Solomon
code over GF(2^8) with 4 parity bytes:
  - Corrects up to 2 corrupted byte symbols (or up to 4 known erasure positions).
  - Handles bit-to-byte packing and unpacking.
  - Converts UTF-8 string payloads to and from 32-bit binary representations.
"""
import numpy as np
from reedsolo import ReedSolomonError, RSCodec

from .config import ECC_PARITY_BYTES, MESSAGE_BITS

# Global Reed-Solomon codec instance initialized with 4 parity bytes
_rs = RSCodec(ECC_PARITY_BYTES)


def bits_to_bytes(bits):
    """Packs a 1-D binary bit array into bytes in MSB-first order.

    Args:
        bits (array-like): Sequence of binary bits (0 or 1).

    Returns:
        bytes: Packed binary byte stream.
    """
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def bytes_to_bits(data):
    """Unpacks a byte buffer into a 1-D numpy array of binary bits (MSB-first).

    Args:
        data (bytes or bytearray): Input byte buffer.

    Returns:
        np.ndarray: 1-D int64 array containing unpacked binary bits.
    """
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8)).astype(np.int64)


def encode(message_bits):
    """MESSAGE_BITS message bits -> PAYLOAD_BITS coded bits.

    Applies systematic Reed-Solomon encoding to expand the 32-bit message
    into a 64-bit codeword containing 4 parity bytes.

    Args:
        message_bits (array-like): Input binary message sequence of length MESSAGE_BITS (32).

    Returns:
        np.ndarray: 64-bit coded codeword sequence (int64).

    Raises:
        AssertionError: If message_bits length does not equal MESSAGE_BITS.
    """
    assert len(message_bits) == MESSAGE_BITS
    # Pack to 4 bytes, compute 4 parity bytes via RSCodec, and unpack back to 64 bits
    return bytes_to_bits(_rs.encode(bits_to_bytes(message_bits)))


def decode(coded_bits):
    """Return (message_bits, ok). ok=False if there were too many errors to correct.

    Attempts algebraic Reed-Solomon decoding. If byte errors are within the
    2-symbol correction bound, the corrected message bits are returned with ok=True.
    On uncorrectable corruption, the raw prefix bits are returned with ok=False.

    Args:
        coded_bits (array-like): Candidate 64-bit extracted codeword bitstream.

    Returns:
        tuple[np.ndarray, bool]: (decoded_message_bits, success_flag).
    """
    try:
        # Decode byte stream and strip parity bytes
        msg, _, _ = _rs.decode(bits_to_bytes(coded_bits))
        return bytes_to_bits(msg), True
    except ReedSolomonError:
        # Fall back to uncorrected payload prefix when error count exceeds symbol capacity
        return np.asarray(coded_bits[:MESSAGE_BITS], dtype=np.int64), False


def text_to_bits(text):
    """Up to 4 characters -> 32 bits (for the demo app).

    Encodes text into UTF-8, truncates to 4 bytes, right-pads with spaces
    to exactly 4 bytes if shorter, and returns the 32-bit binary array.

    Args:
        text (str): Watermark text payload string.

    Returns:
        np.ndarray: 32-bit unpacked binary representation (int64).
    """
    # Enforce exact 4-byte UTF-8 capacity with space-padding
    raw = text.encode("utf-8")[: MESSAGE_BITS // 8].ljust(MESSAGE_BITS // 8, b" ")
    return bytes_to_bits(raw)


def bits_to_text(bits):
    """Decodes a 32-bit binary message back into a UTF-8 text string.

    Args:
        bits (array-like): 32-bit binary array.

    Returns:
        str: Decoded UTF-8 string with replacement characters for invalid sequences.
    """
    return bits_to_bytes(bits).decode("utf-8", errors="replace")
