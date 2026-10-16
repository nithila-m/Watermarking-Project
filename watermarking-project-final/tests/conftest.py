import cv2
import numpy as np
import pytest
import skimage.data


@pytest.fixture(scope="session")
def image():
    """512x512 BGR test image (scikit-image astronaut), no dataset needed.

    Serves as a deterministic, standalone host image fixture for testing
    embedding, extraction, transform domains, and fidelity checks without
    requiring pre-downloaded external datasets.

    Returns:
        np.ndarray: 512x512 BGR uint8 image array.
    """
    img = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)


@pytest.fixture
def rng():
    """Provides a deterministic NumPy random number generator for test cases.

    Returns:
        np.random.Generator: Seeded pseudo-random generator (seed=0).
    """
    return np.random.default_rng(0)
