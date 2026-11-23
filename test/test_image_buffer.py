from types import SimpleNamespace
from f2g_pose_ros2.image_buffer import ImageBuffer


def test_byte_eviction_keeps_pinned_acquisition_alive():
    history = ImageBuffer(maxlen=4, max_bytes=10)
    pinned = SimpleNamespace(data=b'123456')
    history.append(pinned)
    history.append(SimpleNamespace(data=b'abcdef'))
    assert list(history) != [pinned]
    assert pinned.data == b'123456'
    assert history.bytes_used == 6
    history.append(SimpleNamespace(data=b'x' * 11))
    assert history.bytes_used == 6
    history.clear()
    assert not history and history.bytes_used == 0


def test_count_and_bytes_remain_consistent():
    history = ImageBuffer(maxlen=2, max_bytes=10)
    for _ in range(100):
        history.append(SimpleNamespace(data=b'123'))
        assert history.bytes_used == sum(len(item.data) for item in history) <= 10
        assert len(history) <= 2
    history.popleft()
    assert history.bytes_used == 3
