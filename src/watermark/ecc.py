"""Reed-Solomon layer. Member C owns the final version; this one keeps the same
interface so A can test end to end before C's is merged."""
import numpy as np
from reedsolo import ReedSolomonError, RSCodec

from .config import ECC_PARITY_BYTES, MESSAGE_BITS

_rs = RSCodec(ECC_PARITY_BYTES)


def bits_to_bytes(bits):
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def bytes_to_bits(data):
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8)).astype(np.int64)


def encode(message_bits):
    """MESSAGE_BITS message bits -> PAYLOAD_BITS coded bits."""
    assert len(message_bits) == MESSAGE_BITS
    return bytes_to_bits(_rs.encode(bits_to_bytes(message_bits)))


def decode(coded_bits):
    """Return (message_bits, ok). ok=False if there were too many errors to correct."""
    try:
        msg, _, _ = _rs.decode(bits_to_bytes(coded_bits))
        return bytes_to_bits(msg), True
    except ReedSolomonError:
        return np.asarray(coded_bits[:MESSAGE_BITS], dtype=np.int64), False


def text_to_bits(text):
    """Up to 4 characters -> 32 bits (for the demo app)."""
    raw = text.encode("utf-8")[: MESSAGE_BITS // 8].ljust(MESSAGE_BITS // 8, b" ")
    return bytes_to_bits(raw)


def bits_to_text(bits):
    return bits_to_bytes(bits).decode("utf-8", errors="replace")
