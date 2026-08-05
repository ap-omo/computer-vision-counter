# -*- coding: utf-8 -*-
# %%

# Setup
import cv2 as cv
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

net = cv.dnn.readNetFromONNX('yoloe-26n-seg.onnx')
if net is None:
    print("Could not load model")
    exit()

output_names = net.getUnconnectedOutLayersNames()

# %%

# Constants
input_size = 640
scale = 1/255.0     # Workshop 3's normalisation values changed as ONNX model expects input in [0,1] range
mean = (0, 0, 0)
conf_threshold = 0.03       # minimum model confidence for detection
mask_threshold = 0.5        # raise to tighten mask to cat, lower to allow more background
black_frac_threshold = 0.6  # threshold for cat to be labelled black
iou_match_threshold = 0.2   # (runs across frames) overlap needed to determine a detection as duplicate, raise for stricter matching
nms_threshold = 0.4         # (runs in single frames) overlap needed to determine a detection as duplicate, raise for stricter matching
containment_threshold = 0.6 # overlap of smaller box within a bigger one needed to be discarded
redetect_every = 1      # frames per detection ran (raise for faster speeds, lower for more accurate results)
confirm_hits = 3        # frames a detection must be seen in before counted
max_misses = 5          # frames a detection is not present before discarded
fast_forward_frames = 100   # amount of frames per fast forward keypress
rewind_frames = 100         # amount of frames per rewind keypress
# %%

# Letterbox function to prevent aspect ratio distortion
def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv.resize(img, (nw, nh))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[0:nh, 0:nw] = resized
    return padded, scale, nw, nh

# %%

# Non-black cat colour classification (white/ginger/tabby) with confidence
def classify_colour(hsv, mask):
    pixels = hsv[mask]
    if len(pixels) == 0:
        return 'unknown', 0.0

    h, s, v = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    total = len(pixels)

    # Fraction of pixels that match colour criteria
    white_frac = np.logical_and(s < 40, v > 150).sum() / total
    ginger_frac = np.logical_and(np.logical_and(h >= 5, h <= 25), s > 80).sum() / total
    tabby_frac = max(0.0, 1 - white_frac - ginger_frac)

    scores = {'white': white_frac, 'ginger': ginger_frac, 'tabby': tabby_frac}
    colour = max(scores, key=scores.get)
    return colour, scores[colour]

# %%

# Function wrapping detection, mask, colour code to enable running per frame
def detect_black_cats(frame):
    ih, iw = frame.shape[:2]
    padded_img, lb_scale, nw, nh = letterbox(frame, input_size)

    blob = cv.dnn.blobFromImage(padded_img, scalefactor=scale, size=(input_size, input_size),
                                 mean=mean, swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_names)
    output0 = outputs[0][0]
    proto_flat = outputs[1][0].reshape(32, -1)  # shape (32, 160*160)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dark_pixels = hsv[:, :, 2] < 60

    results = []
    for det in output0:
        confidence = det[4]
        if confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = det[0:4]
        x1 = int(np.clip(x1 / lb_scale, 0, iw))
        y1 = int(np.clip(y1 / lb_scale, 0, ih))
        x2 = int(np.clip(x2 / lb_scale, 0, iw))
        y2 = int(np.clip(y2 / lb_scale, 0, ih))
        if x2 <= x1 or y2 <= y1:
            continue

        weights = det[6:38]     # 32 mask weights for detection
        mask_small = (weights @ proto_flat).reshape(160, 160)   # M(x,y) = sum(w_i * P_i)

        # Resize mask up to the padded 640x640 space then undo letterbox padding
        mask_full = cv.resize(mask_small, (input_size, input_size))
        mask_cropped = mask_full[0:nh, 0:nw]
        mask_orig = cv.resize(mask_cropped, (iw, ih))
        cat_mask = mask_orig > mask_threshold

        # Only look at mask within detection's box
        box_mask = np.zeros_like(cat_mask)
        box_mask[y1:y2, x1:x2] = cat_mask[y1:y2, x1:x2]

        cat_pixel_count = box_mask.sum()
        if cat_pixel_count == 0:
            continue
        black_fraction = np.logical_and(box_mask, dark_pixels).sum() / cat_pixel_count
        is_black = black_fraction > black_frac_threshold
        if is_black:
            colour, colour_confidence = 'black', black_fraction
        else:
            colour, colour_confidence = classify_colour(hsv, box_mask)
        results.append((x1, y1, x2, y2, float(confidence), float(black_fraction),
                         is_black, colour, colour_confidence, box_mask))
    return results

# %%

# IoU (Intersection over Union) checks if new detection overlaps current detection
def iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

# NMS (Non Maximum Suppression) cleans up duplicate boxes within single frames
def apply_nms(detections, nms_threshold=nms_threshold):
    if not detections:
        return []
    boxes = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections]
    confidences = [d[4] for d in detections]
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return [detections[i] for i in indices]

# Remove smaller boxes overlapping within bigger ones
def remove_contained_boxes(detections, containment_threshold=containment_threshold):
    keep = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det[0:4]
        area = (x2 - x1) * (y2 - y1)

        is_contained = False
        for j, other in enumerate(detections):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = other[0:4]
            other_area = (ox2 - ox1) * (oy2 - oy1)
            if other_area <= area:
                continue  # only check against bigger boxes

            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            if area > 0 and inter / area > containment_threshold:
                is_contained = True
                break

        if not is_contained:
            keep.append(det)
    return keep

# Discard detections after some time, existing detections get more patience to avoid recounts
def miss_limit(cat):
    return max_misses * 5 if cat['counted'] else max_misses

# %%

# File picker (video/image)
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select a video or image file",
    filetypes=[
        ("Video or image files", "*.mp4 *.avi *.mov *.jpg *.jpeg *.png *.bmp"),
        ("Video files", "*.mp4 *.avi *.mov"),
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
    ]
)
if not file_path:
    print("No file selected")
 
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
is_image = os.path.splitext(file_path)[1].lower() in image_extensions
 
cap = None
known_cats = []   # {'box','confidence','black_fraction','is_black','colour','mask','hits','misses','counted'}
black_cat_count = 0
frame_idx = 0
count_log = []   # frame_idx at which each black cat was confirmed, so rewinding can undo counts

if is_image:
    frame = cv.imread(file_path)
    if frame is None:
        print("Could not load image")
        exit()
 
    detections = detect_black_cats(frame)
    detections = apply_nms(detections)
    detections = remove_contained_boxes(detections)
 
    for (x1, y1, x2, y2, confidence, black_fraction, is_black,
         colour, colour_confidence, mask) in detections:
        known_cats.append({
            'box': (x1, y1, x2, y2), 'confidence': confidence,
            'black_fraction': black_fraction, 'is_black': is_black,
            'colour': colour, 'colour_confidence': colour_confidence, 'mask': mask,
            'hits': 1, 'misses': 0, 'counted': is_black
        })
        if is_black:
            black_cat_count += 1
else:
    cap = cv.VideoCapture(file_path)
    if not cap.isOpened():
        print("Cannot open video")
 
    ret, frame = cap.read()
    if not ret:
        print("Could not read video")

# %%

# Keybinds
show_boxes = True
show_mask = False
show_help = False
paused = False

help_lines = [
    "SPACE: pause/play",
    "f: fast forward",
    "r: rewind",
    "b: toggle boxes",
    "m: toggle mask overlay",
    "q: quit",
]

if is_image:
    help_lines = [line for line in help_lines
                  if not (line.startswith('SPACE') or line.startswith('f:') or line.startswith('r:'))]
 
def reset_tracking():
    known_cats.clear()  # clears all tracked cats when fast forward / rewinding

# %%

# Video loop
while True:
    if not is_image and not paused:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % redetect_every == 0:
            detections = detect_black_cats(frame)
            detections = apply_nms(detections)
            detections = remove_contained_boxes(detections)
    
            for cat in known_cats:
                cat['misses'] += 1  # assume unmatched unless match found below
    
            for (x1, y1, x2, y2, confidence, black_fraction, is_black, 
                 colour, colour_confidence, mask) in detections:
                new_box = (x1, y1, x2, y2)
    
                best_match = None
                for cat in known_cats:
                    if iou(new_box, cat['box']) > iou_match_threshold:
                        best_match = cat
                        break
    
                if best_match:
                    best_match['box'] = new_box
                    best_match['confidence'] = confidence
                    best_match['black_fraction'] = black_fraction
                    best_match['is_black'] = is_black
                    best_match['colour'] = colour
                    best_match['colour_confidence'] = colour_confidence
                    best_match['mask'] = mask
                    best_match['hits'] += 1
                    best_match['misses'] = 0
    
                    if best_match['hits'] == confirm_hits and is_black and not best_match['counted']:
                        black_cat_count += 1
                        best_match['counted'] = True
                        count_log.append(frame_idx)

                else:
                    known_cats.append({
                        'box': new_box, 'confidence': confidence,
                        'black_fraction': black_fraction, 'is_black': is_black,
                        'colour': colour, 'colour_confidence': colour_confidence, 'mask': mask,
                        'hits': 1, 'misses': 0, 'counted': False
                    })
    
            known_cats = [cat for cat in known_cats if cat['misses'] <= miss_limit(cat)]
            
        frame_idx += 1
        
    display = frame.copy()

    if show_mask:
        overlay = display.copy()
        for cat in known_cats:
            overlay[cat['mask']] = (255, 0, 255)
        display = cv.addWeighted(overlay, 0.4, display, 0.6, 0)

    if show_boxes:
        for cat in known_cats:
            x1, y1, x2, y2 = cat['box']
            label = f"{cat['colour']} ({cat['colour_confidence']:.2f})"
            color = (0, 0, 255) if cat['is_black'] else (0, 255, 0)
            cv.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv.putText(display, f"cat: {cat['confidence']:.2f} | {label}", (x1, max(y1 - 10, 0)),
                       cv.FONT_HERSHEY_DUPLEX, 0.6, color, 1)

    cv.putText(display, f'Black cats counted: {black_cat_count}', (20, 40),
               cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 2)
    cv.putText(display, 'h - open keybind menu', (20, display.shape[0] - 15),
               cv.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

    if show_help:
        for i, line in enumerate(help_lines):
            cv.putText(display, line, (20, 70 + i * 22),
                       cv.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    cv.imshow('Black Cat Counter', display)

    wait_time = 0 if paused else 1
    key = cv.waitKey(wait_time) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
    elif key == ord('b'):
        show_boxes = not show_boxes
    elif key == ord('m'):
        show_mask = not show_mask
    elif key == ord('h'):
        show_help = not show_help
    elif key == ord('f') and not is_image:
        current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
        target_frame = current_frame + fast_forward_frames
        cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)
        frame_idx = int(target_frame)
        reset_tracking()
    elif key == ord('r') and not is_image:
        current_frame = cap.get(cv.CAP_PROP_POS_FRAMES)
        target_frame = max(0, current_frame - rewind_frames)
        cap.set(cv.CAP_PROP_POS_FRAMES, target_frame)
        frame_idx = int(target_frame)

        # Undo counts after the point rewinded to
        removed = [f for f in count_log if f > target_frame]
        black_cat_count -= len(removed)
        count_log = [f for f in count_log if f <= target_frame]
        reset_tracking()

if cap is not None:
    cap.release()
    
cv.destroyAllWindows()
print(f'Total black cats counted: {black_cat_count}')