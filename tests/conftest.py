import cv2
import numpy as np
import pytest
import skimage.data


@pytest.fixture(scope="session")
def image():
    """512x512 BGR test image (scikit-image astronaut), no dataset needed."""
    img = cv2.cvtColor(skimage.data.astronaut(), cv2.COLOR_RGB2BGR)
    return cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
