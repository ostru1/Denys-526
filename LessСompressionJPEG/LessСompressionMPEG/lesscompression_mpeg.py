from __future__ import annotations

import math
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "Results"
DEFAULT_VIDEO_NAME = "sample4.avi"
VIDEO_NAME_CANDIDATES = ("sample4.avi", "sample4 (1).avi")
HISTOGRAM_FILENAME = "Гістограма кількості біт на піксель для різних варіантів кодування.png"
FALLBACK_VIDEO_PATHS = [
    SCRIPT_DIR / DEFAULT_VIDEO_NAME,
    SCRIPT_DIR / "sample4 (1).avi",
    SCRIPT_DIR.parent / DEFAULT_VIDEO_NAME,
    SCRIPT_DIR.parent / "sample4 (1).avi",
    Path.cwd() / DEFAULT_VIDEO_NAME,
    Path.cwd() / "sample4 (1).avi",
    Path(r"D:\Zagruzki") / DEFAULT_VIDEO_NAME,
    Path(r"D:\Zagruzki") / "sample4 (1).avi",
]

FALLBACK_VIDEO_DIRS = tuple(dict.fromkeys(path.parent for path in FALLBACK_VIDEO_PATHS))


def segmentImage(anchor, blockSize=16):
    h, w = anchor.shape
    hSegments = int(h / blockSize)
    wSegments = int(w / blockSize)
    return hSegments, wSegments


def getCenter(x, y, blockSize):
    return int(x + blockSize / 2), int(y + blockSize / 2)


def getAnchorSearchArea(x, y, anchor, blockSize, searchArea):
    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)
    sx = max(0, cx - int(blockSize / 2) - searchArea)
    sy = max(0, cy - int(blockSize / 2) - searchArea)
    anchorSearch = anchor[
        sy : min(sy + searchArea * 2 + blockSize, h),
        sx : min(sx + searchArea * 2 + blockSize, w),
    ]
    return anchorSearch


def getBlockZone(p, aSearch, tBlock, blockSize):
    px, py = p
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)
    px, py = max(0, px), max(0, py)
    aBlock = aSearch[py : py + blockSize, px : px + blockSize]
    if aBlock.shape != tBlock.shape:
        return None
    return aBlock


def getMAD(tBlock, aBlock):
    diff = np.subtract(tBlock.astype(np.float32), aBlock.astype(np.float32))
    return float(np.sum(np.abs(diff)) / (tBlock.shape[0] * tBlock.shape[1]))


def getBestMatch(tBlock, aSearch, blockSize):
    step = max(1, 2 ** int(math.floor(math.log2(7))))
    ah, aw = aSearch.shape
    acy, acx = int(ah / 2), int(aw / 2)
    minP = (acx, acy)

    while step >= 1:
        minMAD = float("+inf")
        p1 = (acx, acy)
        p2 = (acx + step, acy)
        p3 = (acx, acy + step)
        p4 = (acx + step, acy + step)
        p5 = (acx - step, acy)
        p6 = (acx, acy - step)
        p7 = (acx - step, acy - step)
        p8 = (acx + step, acy - step)
        p9 = (acx - step, acy + step)
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

        for point in pointList:
            aBlock = getBlockZone(point, aSearch, tBlock, blockSize)
            if aBlock is None:
                continue
            MAD = getMAD(tBlock, aBlock)
            if MAD < minMAD:
                minMAD = MAD
                minP = point

        acx, acy = minP
        step = int(step / 2)

    px, py = minP
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)
    px, py = max(0, px), max(0, py)
    matchBlock = aSearch[py : py + blockSize, px : px + blockSize]
    if matchBlock.shape != tBlock.shape:
        return tBlock.copy()
    return matchBlock


def blockSearchBody(anchor, target, blockSize, searchArea=7):
    h, w = anchor.shape
    hSegments, wSegments = segmentImage(anchor, blockSize)
    predicted = anchor.copy()
    bcount = 0

    for y in range(0, int(hSegments * blockSize), blockSize):
        for x in range(0, int(wSegments * blockSize), blockSize):
            bcount += 1
            targetBlock = target[y : y + blockSize, x : x + blockSize]
            anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea)
            anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize)
            predicted[y : y + blockSize, x : x + blockSize] = anchorBlock

    assert bcount == int(hSegments * wSegments)
    return predicted


def getResidual(target, predicted):
    return np.subtract(target.astype(np.int16), predicted.astype(np.int16))


def getReconstructTarget(residual, predicted):
    restored = np.add(residual.astype(np.int16), predicted.astype(np.int16))
    return np.clip(restored, 0, 255).astype(np.uint8)


def getBitsPerPixel(im):
    pixels = np.asarray(im, dtype=np.float64)
    return float(np.log2(np.abs(pixels) + 1.0).mean())


def resolve_video_path(filename):
    candidate = Path(filename)
    if candidate.exists():
        return candidate

    requested_name = candidate.name
    for directory in FALLBACK_VIDEO_DIRS:
        requested_path = directory / requested_name
        if requested_path.exists():
            return requested_path

    for video_name in VIDEO_NAME_CANDIDATES:
        if candidate.name == video_name:
            continue
        alternative = candidate.with_name(video_name)
        if alternative.exists():
            return alternative

    for directory in FALLBACK_VIDEO_DIRS:
        for video_name in VIDEO_NAME_CANDIDATES:
            alternative = directory / video_name
            if alternative.exists():
                return alternative

    for path in FALLBACK_VIDEO_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate '{filename}'. Checked: "
        + ", ".join(str(path) for path in [candidate, *FALLBACK_VIDEO_PATHS])
    )


def get_frame_count(filename):
    video_path = resolve_video_path(filename)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def getFrames(filename, first_frame, second_frame):
    video_path = resolve_video_path(filename)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, first_frame - 1))
    res1, fr1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, second_frame - 1))
    res2, fr2 = cap.read()
    cap.release()

    if not res1 or fr1 is None:
        raise RuntimeError(f"Could not read frame {first_frame} from {video_path}")
    if not res2 or fr2 is None:
        raise RuntimeError(f"Could not read frame {second_frame} from {video_path}")

    return fr1, fr2


def ensure_results_dir(outfile):
    os.makedirs(outfile, exist_ok=True)


def to_uint8_image(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def residual_to_viewable_image(residual):
    viewable = residual.astype(np.int16) + 128
    return np.clip(viewable, 0, 255).astype(np.uint8)


def save_image(path, image):
    path = Path(path)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for saving: {path}")
    path.write_bytes(encoded.tobytes())


def save_chart(bitsAnchor, bitsDiff, bitsPredicted, outfile):
    barWidth = 0.25
    plt.figure(figsize=(12, 8))

    P1 = [sum(bitsAnchor), bitsAnchor[0], bitsAnchor[1], bitsAnchor[2]]
    Diff = [sum(bitsDiff), bitsDiff[0], bitsDiff[1], bitsDiff[2]]
    Mpeg = [sum(bitsPredicted), bitsPredicted[0], bitsPredicted[1], bitsPredicted[2]]

    br1 = np.arange(len(P1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, P1, color="r", width=barWidth, edgecolor="grey", label="Original frame")
    plt.bar(br2, Diff, color="g", width=barWidth, edgecolor="grey", label="Frame difference")
    plt.bar(br3, Mpeg, color="b", width=barWidth, edgecolor="grey", label="Motion compensated")
    plt.title(
        f"Compression ratio = {round(sum(bitsAnchor) / max(sum(bitsPredicted), 1e-9), 2)}",
        fontweight="bold",
        fontsize=15,
    )
    plt.ylabel("Bits per pixel", fontweight="bold", fontsize=15)
    plt.xticks(
        [r + barWidth for r in range(len(P1))],
        ["RGB", "R", "G", "B"],
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(outfile) / HISTOGRAM_FILENAME, dpi=600)
    plt.close()


def main(anchorFrame, targetFrame, blockSize=16, searchArea=7, saveOutput=True, outfile=RESULTS_DIR):
    bitsAnchor = []
    bitsDiff = []
    bitsPredicted = []

    h, w, ch = anchorFrame.shape
    diffFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)
    predictedFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)
    residualFrameRGB = np.zeros((h, w, ch), dtype=np.int16)
    restoreFrameRGB = np.zeros((h, w, ch), dtype=np.uint8)

    for i in range(0, 3):
        anchorFrame_c = anchorFrame[:, :, i]
        targetFrame_c = targetFrame[:, :, i]

        diffFrame = cv2.absdiff(anchorFrame_c, targetFrame_c)
        predictedFrame = blockSearchBody(anchorFrame_c, targetFrame_c, blockSize, searchArea)
        residualFrame = getResidual(targetFrame_c, predictedFrame)
        reconstructTargetFrame = getReconstructTarget(residualFrame, predictedFrame)

        bitsAnchor.append(getBitsPerPixel(anchorFrame_c))
        bitsDiff.append(getBitsPerPixel(diffFrame))
        bitsPredicted.append(getBitsPerPixel(residualFrame))

        diffFrameRGB[:, :, i] = diffFrame
        predictedFrameRGB[:, :, i] = predictedFrame
        residualFrameRGB[:, :, i] = residualFrame
        restoreFrameRGB[:, :, i] = reconstructTargetFrame

    reconstruction_ok = np.array_equal(restoreFrameRGB, targetFrame)

    if saveOutput:
        ensure_results_dir(outfile)
        save_image(Path(outfile) / "First frame.png", anchorFrame)
        save_image(Path(outfile) / "Second frame.png", targetFrame)
        save_image(Path(outfile) / "Difference between frame.png", diffFrameRGB)
        save_image(Path(outfile) / "Prediction frame.png", predictedFrameRGB)
        save_image(Path(outfile) / "Residual frame.png", residual_to_viewable_image(residualFrameRGB))
        save_image(Path(outfile) / "Restore frame.png", restoreFrameRGB)
        save_chart(bitsAnchor, bitsDiff, bitsPredicted, outfile)

    return {
        "bitsAnchor": bitsAnchor,
        "bitsDiff": bitsDiff,
        "bitsPredicted": bitsPredicted,
        "compression_ratio": sum(bitsAnchor) / max(sum(bitsPredicted), 1e-9),
        "reconstruction_ok": reconstruction_ok,
    }


if __name__ == "__main__":
    frame_count = get_frame_count(DEFAULT_VIDEO_NAME)
    max_first_frame = min(frame_count - 1, 3000)
    fr = random.randint(1, max_first_frame)
    frame1, frame2 = getFrames(DEFAULT_VIDEO_NAME, fr, fr + 1)
    metrics = main(frame1, frame2, saveOutput=True)

    print(f"Selected frames: {fr} and {fr + 1}")
    print(f"Reconstruction matches target: {metrics['reconstruction_ok']}")
    print(
        "Bits per pixel (anchor / diff / residual RGB sum): "
        f"{sum(metrics['bitsAnchor']):.4f} / "
        f"{sum(metrics['bitsDiff']):.4f} / "
        f"{sum(metrics['bitsPredicted']):.4f}"
    )
    print(f"Compression ratio: {metrics['compression_ratio']:.4f}")
