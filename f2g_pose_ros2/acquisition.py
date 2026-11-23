"""Validate an atomic selected-instance acquisition without depending on ROS."""
import math


def stamp_ns(stamp):
    return stamp.sec * 10**9 + stamp.nanosec


def validate_acquisition(acquisition, target, now_ns, max_age=180., max_bytes=64*1024*1024):
    packet = acquisition.target
    context = packet.target
    color, depth, mask, info = (acquisition.color, acquisition.depth,
                                acquisition.selected_mask, acquisition.camera_info)
    if packet.invalidated:
        raise ValueError('Selection was invalidated')
    if (not context.instance_validated or not context.instance_id or not context.perception_epoch
            or not context.acquisition_id or not context.target_generation
            or context.class_label.casefold() != target.casefold()):
        raise ValueError('Unverified or mismatched selected-instance identity')
    if (not color.header.frame_id or not packet.points.header.frame_id or context.acquisition_header != color.header
            or mask.header != color.header or context.geometry_header != packet.points.header
            or packet.points.header.stamp != color.header.stamp):
        raise ValueError('Cloud, selected mask and RGB acquisition do not match')
    age = (now_ns - stamp_ns(color.header.stamp)) / 1e9
    if stamp_ns(color.header.stamp) <= 0 or not 0 <= age <= max_age:
        raise ValueError('Acquisition is expired or in the future')
    shape = (color.width, color.height)
    if (min(shape) <= 0 or any((m.width, m.height) != shape for m in (depth, mask, info))
            or depth.header.frame_id != color.header.frame_id
            or abs(stamp_ns(depth.header.stamp) - stamp_ns(color.header.stamp)) > 50_000_000
            or mask.encoding != 'mono8' or depth.encoding not in ('16UC1', '32FC1')):
        raise ValueError('Acquisition images are not a registered RGB-D/selected-mask pair')
    if (info.header.frame_id != color.header.frame_id
            or stamp_ns(info.header.stamp) not in (0, stamp_ns(color.header.stamp))
            or len(info.k) != 9 or not all(math.isfinite(v) for v in info.k)
            or min(info.k[0], info.k[4]) <= 0):
        raise ValueError('Acquisition calibration does not match RGB')
    images = (color, depth, mask)
    if any(m.step <= 0 or len(m.data) < m.step * m.height for m in images):
        raise ValueError('Acquisition image data is incomplete')
    if sum(len(m.data) for m in (*images, packet.points)) > max_bytes:
        raise ValueError('Acquisition exceeds configured memory bound')
    return context


class SelectionGuard:
    """Monotonic selection state; duplicates and retired epochs cannot run again."""
    def __init__(self):
        self.epoch = None
        self.version = (-1, -1, -1)
        self.retired = set()
        self.invalidated = False
        self.processed = False

    def observe(self, context, invalidated=False, processing=False):
        epoch = context.perception_epoch
        if not epoch or epoch in self.retired:
            return False
        version = (context.perception_generation, context.target_generation, context.acquisition_id)
        if epoch != self.epoch:
            if len(self.retired) >= 64:
                raise ValueError('Selection epoch history exhausted; restart perception')
            if self.epoch is not None:
                self.retired.add(self.epoch)
            self.epoch, self.version = epoch, (-1, -1, -1)
        if version < self.version:
            return False
        if version > self.version:
            self.version, self.invalidated, self.processed = version, False, False
        if invalidated:
            self.invalidated = True
            return True
        if self.invalidated or (processing and self.processed):
            return False
        if processing:
            self.processed = True
        return True

    def can_process(self, context):
        if not context.perception_epoch or context.perception_epoch in self.retired:
            return False
        if context.perception_epoch == self.epoch:
            version = (context.perception_generation, context.target_generation, context.acquisition_id)
            if version < self.version or (version == self.version and (self.invalidated or self.processed)):
                return False
        return True

    def reset(self):
        if self.epoch is not None:
            self.retired.add(self.epoch)
        self.epoch, self.version = None, (-1, -1, -1)
        self.invalidated, self.processed = False, False
