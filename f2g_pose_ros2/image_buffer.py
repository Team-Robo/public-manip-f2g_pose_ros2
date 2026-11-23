"""Count- and byte-bounded image history; acquisition pins own separate references."""
from collections import deque


class ImageBuffer(deque):
    def __init__(self, maxlen=32, max_bytes=33554432):
        if maxlen < 1 or max_bytes < 1:
            raise ValueError('Image history limits must be positive')
        super().__init__(maxlen=maxlen)
        self.max_bytes = max_bytes
        self.bytes_used = 0

    def popleft(self):
        item = super().popleft()
        self.bytes_used -= len(item.data)
        return item

    def clear(self):
        super().clear()
        self.bytes_used = 0

    def append(self, item):
        size = len(item.data)
        if size > self.max_bytes:
            return
        while self and (len(self) >= self.maxlen or self.bytes_used + size > self.max_bytes):
            self.popleft()
        super().append(item)
        self.bytes_used += size
