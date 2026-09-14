import numpy as np

from tools.scan_cctv import Track, _iou


def test_iou_overlap():
    a = (10, 10, 100, 100)
    assert _iou(a, a) == 1.0
    assert 0.0 < _iou(a, (30, 30, 100, 100)) < 1.0
    assert _iou(a, (200, 200, 40, 40)) == 0.0


def test_track_rejects_below_threshold():
    t = Track((0, 0, 80, 80))
    t.update((0, 0, 80, 80), np.zeros((112, 112, 3), np.uint8),
             "ALEX", "0000000000000001", 0.30, np.zeros((240, 320, 3), np.uint8),
             5, 1234, 0.40)
    assert t.result(0.40) is None


def test_track_requires_minimum_frames():
    t = Track((0, 0, 80, 80))
    t.update((0, 0, 80, 80), np.zeros((112, 112, 3), np.uint8),
             "ALEX", "0000000000000001", 0.80, np.zeros((240, 320, 3), np.uint8),
             0, 50, 0.40)
    assert t.result(0.40) is None  # only 1 frame so far


def test_track_keeps_best_match():
    t = Track((0, 0, 80, 80))
    low = np.zeros((112, 112, 3), np.uint8)
    t.update((0, 0, 80, 80), low, "ALEX", "0000000000000001", 0.50,
             np.zeros((240, 320, 3), np.uint8), 1, 100, 0.40)
    t.update((0, 0, 80, 80), low, "ALEX", "0000000000000001", 0.86,
             np.ones((240, 320, 3), np.uint8) * 255, 2, 200, 0.40)
    res = t.result(0.40)
    assert res is not None
    assert res["pct"] == 0.86
    assert res["pts"] == 200
    assert res["frame"] == 2