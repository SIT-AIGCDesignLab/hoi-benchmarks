#!/usr/bin/env python3
"""
Crop HICO image into 3 separate images: person, object, and union regions.
"""

import json
import os
from PIL import Image


def compute_union_box(box1, box2):
    """
    Compute union bounding box of two boxes.
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        [x1, y1, x2, y2] union box
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return [x1, y1, x2, y2]


def crop_images(image_path, person_bbox, object_bbox, union_bbox, output_base_path, action="riding horse"):
    """
    Crop image into 3 separate images: person, object, and union regions.
    
    Args:
        image_path: Path to input image
        person_bbox: [x1, y1, x2, y2] person bounding box
        object_bbox: [x1, y1, x2, y2] object bounding box
        union_bbox: [x1, y1, x2, y2] union bounding box
        output_base_path: Base path for output images (will add suffixes)
        action: Action description for filename
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Crop person region
    person_crop = img.crop((person_bbox[0], person_bbox[1], person_bbox[2], person_bbox[3]))
    person_path = output_base_path.replace('.jpg', '_person.jpg')
    person_crop.save(person_path, quality=95)
    print(f"✓ Saved person crop to: {person_path} (size: {person_crop.size})")
    
    # Crop object region
    object_crop = img.crop((object_bbox[0], object_bbox[1], object_bbox[2], object_bbox[3]))
    object_path = output_base_path.replace('.jpg', '_object.jpg')
    object_crop.save(object_path, quality=95)
    print(f"✓ Saved object crop to: {object_path} (size: {object_crop.size})")
    
    # Crop union region
    union_crop = img.crop((union_bbox[0], union_bbox[1], union_bbox[2], union_bbox[3]))
    union_path = output_base_path.replace('.jpg', '_union.jpg')
    union_crop.save(union_path, quality=95)
    print(f"✓ Saved union crop to: {union_path} (size: {union_crop.size})")
    
    return person_path, object_path, union_path


def main():
    # Image path
    image_path = "data/hico_20160224_det/images/test2015/HICO_test2015_00000002.jpg"
    annotation_file = "data/benchmarks_simplified/hico_ground_test_simplified.json"
    output_path = "HICO_test2015_00000002_riding_horse.jpg"
    
    # Load annotations
    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Find the sample for this image with "riding horse" action
    sample = None
    for s in annotations:
        if (s['file_name'] == 'HICO_test2015_00000002.jpg' and 
            'riding' in s['action'].lower() and 
            'horse' in s['object_category'].lower()):
            sample = s
            break
    
    if sample is None:
        print("Error: Could not find annotation for HICO_test2015_00000002.jpg with 'riding horse' action")
        return 1
    
    print(f"Found annotation:")
    print(f"  Action: {sample['action']} {sample['object_category']}")
    print(f"  Image size: {sample['width']}x{sample['height']}")
    print(f"  Number of pairs: {sample['num_pairs']}")
    
    # Extract bounding boxes
    boxes = sample['boxes']
    gt_box_inds = sample['gt_box_inds']
    
    # Get person and object boxes (first pair)
    person_idx = gt_box_inds[0]
    object_idx = gt_box_inds[1]
    
    person_bbox = boxes[person_idx]
    object_bbox = boxes[object_idx]
    
    print(f"\nBounding boxes:")
    print(f"  Person: {person_bbox}")
    print(f"  Object (horse): {object_bbox}")
    
    # Compute union box
    union_bbox = compute_union_box(person_bbox, object_bbox)
    print(f"  Union: {union_bbox}")
    
    # Verify image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return 1
    
    # Create cropped images
    print(f"\nCreating cropped images...")
    action_desc = f"{sample['action']} {sample['object_category']}"
    person_path, object_path, union_path = crop_images(
        image_path, person_bbox, object_bbox, union_bbox, output_path, action=action_desc
    )
    
    print(f"\n✓ Cropping complete! Created 3 images:")
    print(f"  1. Person: {person_path}")
    print(f"  2. Object: {object_path}")
    print(f"  3. Union: {union_path}")
    return 0


if __name__ == '__main__':
    exit(main())
