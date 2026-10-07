import numpy as np

from sender import split_frames


def test_split_frames_pads_last():
    x = np.arange(10, dtype=np.int16).reshape(10, 1)
    blocks = list(split_frames(x, 4))
    # Every packet must be the same size (the receiver's playout block size),
    # so the leftover 2 frames are padded with silence.
    assert [b.shape for b in blocks] == [(4, 1)] * 3
    assert blocks[2][:, 0].tolist() == [8, 9, 0, 0]
