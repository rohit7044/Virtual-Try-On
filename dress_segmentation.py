import cv2
import torch
import numpy as np
import os
import time
import argparse
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


def setup_environment():
    torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_sam2_predictor(checkpoint_path, config_path):
    sam2_model = build_sam2(config_path, checkpoint_path, device="cuda")
    return SAM2ImagePredictor(sam2_model)


def build_grounding_dino_model(config_path, checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return load_model(model_config_path=config_path, model_checkpoint_path=checkpoint_path, device=device)


def get_image_and_boxes(image_path, text_prompt, grounding_model):
    image_source, image = load_image(image_path)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=0.35,
        text_threshold=0.25
    )
    return image_source, image, boxes, labels


def convert_boxes(boxes, image_shape):
    h, w, _ = image_shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    return box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


def predict_masks(sam2_predictor, image_source, input_boxes):
    sam2_predictor.set_image(image_source)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    return masks.squeeze(1) if masks.ndim == 4 else masks


def categorize_and_save_masks(masks, labels, output_dir):
    upper_garments = ['shirt', 'jacket', 'blouse']
    lower_garments = ['pant', 'skirt', 'shorts', 'trousers', 'undergarments']
    footwear = ['shoe', 'boot', 'sandal']

    binary_upper_mask = np.zeros_like(masks[0], dtype=np.uint8)
    binary_lower_mask = np.zeros_like(masks[0], dtype=np.uint8)
    binary_footwear_mask = np.zeros_like(masks[0], dtype=np.uint8)

    for idx, class_name in enumerate(labels):
        if any(item in class_name for item in upper_garments):
            binary_upper_mask = np.maximum(binary_upper_mask, (masks[idx] * 255).astype(np.uint8))
        elif any(item in class_name for item in lower_garments):
            binary_lower_mask = np.maximum(binary_lower_mask, (masks[idx] * 255).astype(np.uint8))
        elif any(item in class_name for item in footwear):
            binary_footwear_mask = np.maximum(binary_footwear_mask, (masks[idx] * 255).astype(np.uint8))

    cv2.imwrite(os.path.join(output_dir, "upper_garments_mask.png"), binary_upper_mask)
    cv2.imwrite(os.path.join(output_dir, "lower_garments_mask.png"), binary_lower_mask)
    cv2.imwrite(os.path.join(output_dir, "footwear_mask.png"), binary_footwear_mask)


def main(image_path):
    start = time.time()

    # Environment settings
    setup_environment()

    # Hardcoded model paths and keywords
    model_paths = {
        'sam2_checkpoint': r"checkpoints/sam2_hiera_tiny.pt",
        'sam2_config': "sam2_hiera_t.yaml",
        'grounding_dino_config': r"grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        'grounding_dino_checkpoint': r"gdino_checkpoints/groundingdino_swint_ogc.pth"
    }
    keywords = "shirt, pant, shoe"

    # Build models
    sam2_predictor = build_sam2_predictor(
        checkpoint_path=model_paths['sam2_checkpoint'],
        config_path=model_paths['sam2_config']
    )
    grounding_model = build_grounding_dino_model(
        config_path=model_paths['grounding_dino_config'],
        checkpoint_path=model_paths['grounding_dino_checkpoint']
    )

    # Setup output directory
    output_dir = os.path.join(os.path.dirname(image_path), 'Results')

    # Get image and predict boxes
    image_source, image, boxes, labels = get_image_and_boxes(image_path, keywords, grounding_model)

    # Process the box prompt for SAM2
    input_boxes = convert_boxes(boxes, image_source.shape)

    # Predict masks
    masks = predict_masks(sam2_predictor, image_source, input_boxes)

    # Categorize and save masks
    categorize_and_save_masks(masks, labels, output_dir)

    print(f"Execution time: {time.time() - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dress Segmentation with SAM2 and GroundingDINO")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')

    args = parser.parse_args()

    main(image_path=args.image)