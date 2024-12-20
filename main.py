import argparse
import json
import os
import re
import time
import numpy as np
import cv2
import openai
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

def parse_gpt4_answer(answer_str, num_vectors):
    try:
        fixed_answer_str = re.sub(r'(?<!")(\b\d+\b)(?=:)', r'"\1"', answer_str.strip())
        confidence_dict = json.loads(fixed_answer_str)
    except Exception:
        confidence_dict = {}
    confidences = {i: 0.0 for i in range(1, num_vectors+1)}
    for key, value in confidence_dict.items():
        try:
            key_int = int(key)
            if key_int in confidences:
                confidences[key_int] = float(value)
        except:
            pass
    return confidences

def compute_weighted_average_vector(vectors, confidences):
    sorted_conf = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    top2 = sorted_conf[:2]

    weighted_sum = np.zeros(2)
    total_conf = 0.0
    for idx, conf in top2:
        weighted_sum += vectors[idx-1] * conf
        total_conf += conf

    if total_conf == 0:
        return vectors[0] / np.linalg.norm(vectors[0])
    weighted_average = weighted_sum / total_conf
    return weighted_average / np.linalg.norm(weighted_average)

def sample_around_vector(base_vector, num_samples=5, angle_variation=np.radians(30)):
    base_angle = np.arctan2(base_vector[1], base_vector[0])
    sampled_vectors = []
    for i in range(num_samples):
        angle = base_angle + (np.random.uniform(-1,1)*angle_variation)
        vec = np.array([np.cos(angle), np.sin(angle)])
        sampled_vectors.append(vec)
    return sampled_vectors

def get_user_click_position(image):
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Click a point on the image, then close the window.")
    clicked_point = []
    def onclick(event):
        if event.button == MouseButton.LEFT:
            clicked_point.append((event.xdata, event.ydata))
            plt.close(fig)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    if len(clicked_point) == 0:
        print("No point selected. Exiting.")
        exit(0)
    return clicked_point[0]

def clamp_point(p, width, height):
    x = min(max(int(round(p[0])), 0), width-1)
    y = min(max(int(round(p[1])), 0), height-1)
    return (x, y)

def ensure_label_in_image(label_x, label_y, text_width, text_height, w, h):
    # Adjust so that label fully fits inside the image
    if label_x < 0:
        label_x = 0
    if label_y - text_height < 0:
        label_y = text_height
    if label_x + text_width >= w:
        label_x = w - text_width
    if label_y >= h:
        label_y = h - 1
    return label_x, label_y

def adjust_circle_position(centroid_pixel, target_pixel, placed_circles, min_distance=60, max_shift=50, step=5, w=640, h=480, circle_radius=25):
    """
    Attempt to adjust circle position to prevent overlap with already placed circles and ensure it stays within image bounds.
    Returns the adjusted position or None if not found.
    """
    new_p_pixel = list(target_pixel)
    angle = np.arctan2(target_pixel[1] - centroid_pixel[1], target_pixel[0] - centroid_pixel[0])
    shifts = 0

    def fits_in_image(cx, cy):
        # Check if the circle fits fully inside the image
        return (cx - circle_radius >= 0 and cx + circle_radius < w and cy - circle_radius >= 0 and cy + circle_radius < h)

    while shifts < max_shift:
        if fits_in_image(new_p_pixel[0], new_p_pixel[1]):
            overlap = False
            for placed in placed_circles:
                dist = np.hypot(new_p_pixel[0] - placed[0], new_p_pixel[1] - placed[1])
                if dist < min_distance:
                    overlap = True
                    break
            if not overlap:
                return tuple(new_p_pixel)
        
        # Shift perpendicular to the vector from centroid to target
        shift_direction = (-1) ** (shifts // step)
        shift_angle = angle + (np.pi / 2) * shift_direction
        shift_x = step * np.cos(shift_angle)
        shift_y = step * np.sin(shift_angle)
        new_p_pixel[0] += int(round(shift_x))
        new_p_pixel[1] += int(round(shift_y))
        # Clamp after shift
        new_p_pixel[0] = min(max(new_p_pixel[0], circle_radius), w - circle_radius - 1)
        new_p_pixel[1] = min(max(new_p_pixel[1], circle_radius), h - circle_radius - 1)
        shifts += step
    
    return None  # Could not find a non-overlapping in-bounds position

def annotate_points_2d(image, centroid_2d, points_2d,
                       line_thickness=3,
                       circle_radius=25,
                       circle_color=(255, 255, 255),  
                       circle_opacity=0.5,
                       arrow_opacity=0.7,
                       font=cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale=0.6,
                       font_color=(0, 0, 0),
                       font_thickness=1):
    """
    Annotate a 2D image with semi-transparent white arrows and white semi-transparent circles.
    Strictly ensure no element goes outside the image. If cannot place a circle/label inside, discard that arrow.
    """
    h, w = image.shape[:2]
    annotated_image = image.copy()

    centroid_pixel = clamp_point(centroid_2d, w, h)
    overlay_arrows = np.zeros_like(image, dtype=np.uint8)
    overlay_circles = np.zeros_like(image, dtype=np.uint8)

    # Red centroid
    cv2.circle(annotated_image, centroid_pixel, 5, (0, 0, 255), -1)

    placed_circles = []
    circle_positions = []

    for i, p in enumerate(points_2d):
        p_pixel = clamp_point(p, w, h)
        dx = p_pixel[0] - centroid_pixel[0]
        dy = p_pixel[1] - centroid_pixel[1]
        dist = np.hypot(dx, dy)

        # If dist is zero, skip (point at centroid)
        if dist == 0:
            continue

        # Compute arrow end: shorten it before the circle
        if dist < circle_radius+5:
            ratio = 0.5
        else:
            ratio = (dist - circle_radius - 5) / dist
            if ratio < 0: ratio = 0.5  # fallback

        end_x = centroid_pixel[0] + dx * ratio
        end_y = centroid_pixel[1] + dy * ratio
        arrow_end = clamp_point((end_x, end_y), w, h)

        # Try to place the circle within bounds and no overlap
        adjusted_circle_pos = adjust_circle_position(centroid_pixel, p_pixel, placed_circles, 
                                                     min_distance=circle_radius*2+10, 
                                                     max_shift=50, step=5, w=w, h=h, 
                                                     circle_radius=circle_radius)
        if adjusted_circle_pos is None:
            # Could not place this arrow inside bounds without overlap
            continue

        # Double-check circle fully inside
        cx, cy = adjusted_circle_pos
        if (cx - circle_radius < 0 or cx + circle_radius >= w or cy - circle_radius < 0 or cy + circle_radius >= h):
            # Even after adjustment, doesn't fit
            continue

        # Draw arrow
        cv2.arrowedLine(overlay_arrows, centroid_pixel, arrow_end, (255,255,255), line_thickness, tipLength=0.1)

        # Draw circle
        cv2.circle(overlay_circles, adjusted_circle_pos, circle_radius, circle_color, -1)

        # Label
        label = str(i+1)
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = adjusted_circle_pos[0] - text_width // 2
        text_y = adjusted_circle_pos[1] + text_height // 2
        text_x, text_y = ensure_label_in_image(text_x, text_y, text_width, text_height, w, h)

        # Ensure label is inside circle and image bounds
        # If text placement is drastically changed, it's still better to have text inside image
        # This is acceptable as long as it's inside the circle's area. Usually small adjustments won't break that assumption.
        circle_positions.append((label, text_x, text_y, font, font_scale, font_color, font_thickness))
        placed_circles.append(adjusted_circle_pos)

    # Blend arrows and circles first
    annotated_image = cv2.addWeighted(overlay_arrows, arrow_opacity, annotated_image, 1, 0)
    annotated_image = cv2.addWeighted(overlay_circles, circle_opacity, annotated_image, 1, 0)

    # Now draw text fully opaque on the annotated image
    for label, lx, ly, fnt, fscale, fcolor, fthick in circle_positions:
        cv2.putText(annotated_image, label, (lx, ly), fnt, fscale, fcolor, fthick, cv2.LINE_AA)

    return annotated_image

def process_prompt(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_initial_vectors(num_vectors=16, length=300.0):
    angles = np.linspace(0, 2*np.pi, num_vectors, endpoint=False)
    vectors = []
    for a in angles:
        vec = np.array([np.cos(a), np.sin(a)])
        vec *= length
        vectors.append(vec)
    return vectors

def process_and_refine_vectors(
    image,
    base_point,
    object_name="cabinet",
    num_refinement_stages=1,
    confidence_iteration=2,
    show_plots=True
):
    length = 300.0
    initial_directions = generate_initial_vectors(num_vectors=16, length=length)
    vectors = [base_point + v for v in initial_directions]

    confidences_history = []
    annotated_images_history = []

    for stage in range(num_refinement_stages + 1):
        print(f"=== Refinement Stage {stage} ===")
        annotated = annotate_points_2d(image, base_point, vectors)
        annotated_images_history.append(annotated.copy())

        prompt = (
            f"I have a {object_name} represented on a 2D image. I am holding the {object_name} at a red dot in the image. "
            f"There are {len(vectors)} large, clearly visible arrows drawn from that red dot, numbered 1 to {len(vectors)}, each representing a direction. "
            f"Please provide a dictionary where each key is an arrow number and each value is a confidence (0 to 1) that moving the {object_name} along that arrow will open it. "
            f"Only output a dictionary, for example: {{1:0.8,2:0.9}}"
        )

        confidences = {}
        for _ in range(confidence_iteration):
            gpt4_answer = process_prompt(prompt)
            print("GPT-4 Answer:", gpt4_answer)
            current_confidence = parse_gpt4_answer(gpt4_answer, len(vectors))
            for key, value in current_confidence.items():
                confidences[key] = confidences.get(key, 0) + value

        for key in confidences.keys():
            confidences[key] /= confidence_iteration

        print("Parsed Confidences:", confidences)
        confidences_history.append(confidences.copy())

        if stage == num_refinement_stages:
            break

        direction_vectors = [p - base_point for p in vectors]
        weighted_avg = compute_weighted_average_vector(direction_vectors, confidences)

        sampled = sample_around_vector(weighted_avg, num_samples=10, angle_variation=np.radians(40))
        vectors = [base_point + v*length for v in [weighted_avg]+sampled]

    final_confidences = confidences_history[-1]
    if final_confidences:
        arr_conf = [final_confidences.get(i+1,0) for i in range(len(vectors))]
        arr_conf = np.array(arr_conf)
        sum_c = np.sum(arr_conf)
        if sum_c > 0:
            norm_conf = arr_conf / sum_c
            direction_vectors = [p - base_point for p in vectors]
            direction_norms = np.array([v/np.linalg.norm(v) for v in direction_vectors])
            weighted_sum = np.sum(direction_norms*norm_conf[:,None], axis=0)
            final_direction = weighted_sum/np.linalg.norm(weighted_sum)
            final_vector = base_point + final_direction*length
        else:
            final_vector = vectors[0]
    else:
        final_vector = vectors[0]

    final_annotated = annotate_points_2d(image, base_point, [final_vector])
    annotated_images_history.append(final_annotated.copy())

    if show_plots:
        num_stages = len(annotated_images_history)
        fig, axes = plt.subplots(1, num_stages, figsize=(5*num_stages, 5))
        if num_stages == 1:
            axes = [axes]

        for stage_idx, img in enumerate(annotated_images_history):
            ax = axes[stage_idx]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if stage_idx < num_refinement_stages + 1:
                ax.set_title(f"Stage {stage_idx}")
            else:
                ax.set_title("Final Stage")
            ax.axis('off')

        plt.tight_layout()
        plt.savefig("pivot.jpg", format="jpg", dpi=300, bbox_inches='tight')
        plt.show()

    cv2.imwrite("output_final.jpg", final_annotated)
    print("Final Vector (in pixels):", final_vector - base_point)
    print("Confidences History:", confidences_history)
    print("Output saved to output_final.jpg")

def main():
    parser = argparse.ArgumentParser(description="2D Vector Refinement with GPT-4, no overlap, all inside image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--object_name", type=str, default="cabinet", help="Object name.")
    parser.add_argument("--action", type=str, default="open", help="Action to perform.")
    args = parser.parse_args()

    if not os.path.exists("key.txt"):
        print("No key.txt found. Please provide your API key in key.txt.")
        return
    with open("key.txt","r") as f:
        api_key = f.read().strip()

    openai.api_key = api_key

    orig_image = cv2.imread(args.image)
    if orig_image is None:
        print("Could not load image:", args.image)
        return

    image = cv2.resize(orig_image, (640, 480))

    base_point = get_user_click_position(image)
    base_point = np.array([base_point[0], base_point[1]])

    selected_img = image.copy()
    cv2.circle(selected_img, (int(base_point[0]), int(base_point[1])), 10, (0,255,0), -1)
    plt.imshow(cv2.cvtColor(selected_img, cv2.COLOR_BGR2RGB))
    plt.title("Selected Starting Position")
    plt.axis('off')
    plt.show()

    process_and_refine_vectors(
        image=image,
        base_point=base_point,
        object_name=args.object_name,
        num_refinement_stages=1,
        confidence_iteration=2,
        show_plots=True
    )

if __name__ == "__main__":
    main()
