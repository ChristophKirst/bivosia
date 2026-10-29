"""Inclusive playback ranges, loop bounds and frame stepping."""

__project__ = 'BiVOSIA'
__author__ = 'Kara Quine, Shuyu Wang, Christoph Kirst'
__contact__ = 'christoph.kirst@ucsf.edu'


LOOP_PADDING_FRAMES = 50
MINIMUM_LINK_LOOP_FRAMES = 100


def padded_loop(start, end, frame_count, padding=LOOP_PADDING_FRAMES):
    """Return an inclusive interval with the shared loop-border padding."""
    return max(0, start - padding), min(frame_count - 1, end + padding)


def zone_loop(zone, frame_count, padding=LOOP_PADDING_FRAMES):
    return padded_loop(zone.start, zone.end, frame_count, padding)


def link_loops(boundary, zones, frame_count):
    """Loop outward from each boundary using the shared padding and minimum span."""
    minimum_offset = MINIMUM_LINK_LOOP_FRAMES - 1
    result = {}
    for side, frame in (("left", boundary.left.end), ("right", boundary.right.start)):
        zone = next((z for z in zones if z.start <= frame <= z.end), None)
        if side == "left":
            start = min(
                frame - minimum_offset, zone.start - LOOP_PADDING_FRAMES
            ) if zone is not None else frame - minimum_offset
            result[side] = max(0, start), frame
        else:
            end = max(
                frame + minimum_offset, zone.end + LOOP_PADDING_FRAMES
            ) if zone is not None else frame + minimum_offset
            result[side] = frame, min(frame_count - 1, end)
    return result


def next_frame(frame, start, end, loop):
    if frame < start or frame > end:
        return start
    if frame < end:
        return frame + 1
    return start if loop else None


def upcoming_frames(frame, start, end, loop, count):
    for _ in range(min(count, end - start + 1)):
        frame = next_frame(frame, start, end, loop)
        if frame is None:
            break
        yield frame
