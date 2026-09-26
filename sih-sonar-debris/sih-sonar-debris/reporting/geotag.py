import os
import cv2
import json

processed_dir = "data/processed"
raw_dir = "data/raw"
os.makedirs(processed_dir, exist_ok=True)

print("=== DEBUGGING GEOTAG & BOUNDING BOX SCRIPT ===")
print(f"Looking for processed images in: {processed_dir}")
print(f"Looking for raw files (.jpg and .txt) in: {raw_dir}")

# Step 1: Index all .txt files in data/raw
label_map = {}
if os.path.exists(raw_dir):
    for file in os.listdir(raw_dir):
        if file.lower().endswith(".txt"):
            base_name = os.path.splitext(file)[0]
            label_map[base_name] = os.path.join(raw_dir, file)

print(f"Successfully indexed {len(label_map)} label (.txt) files.")
if len(label_map) > 0:
    print("Sample label key example:", list(label_map.keys())[0])

reports_json = []
count = 0

# Step 2: Loop through processed images and match them
if os.path.exists(processed_dir):
    processed_files = os.listdir(processed_dir)
    print(f"Total files in processed folder: {len(processed_files)}")
    
    for filename in processed_files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")) and not filename.startswith("annotated_"):
            img_path = os.path.join(processed_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read image: {filename}")
                continue
                
            h, w, _ = img.shape
            annotated_img = img.copy()
            
            # Match the processed filename back to the raw base name
            raw_base = filename.replace("processed_", "").rsplit(".", 1)[0]
            txt_path = label_map.get(raw_base)
            
            print(f"Image: {filename} | Looking for base key: {raw_base} | Found Match? {txt_path is not None}")

            anomaly_count = 0
            if txt_path and os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    print(f"  -> Reading {len(lines)} line(s) from text file.")
                    for line in lines:
                        vals = line.strip().split()
                        if len(vals) >= 5:
                            class_id, x_c, y_c, box_w, box_h = map(float, vals[:5])
                            
                            # Convert YOLO normalized coordinates to absolute pixels
                            xmin = int((x_c - box_w / 2) * w)
                            ymin = int((y_c - box_h / 2) * h)
                            xmax = int((x_c + box_w / 2) * w)
                            ymax = int((y_c + box_h / 2) * h)
                            
                            anomaly_count += 1
                            
                            # Draw green bounding boxes
                            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.putText(annotated_img, f"Debris #{anomaly_count}", 
                                        (xmin, max(ymin - 5, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            reports_json.append({
                                "image": filename,
                                "anomaly_id": anomaly_count,
                                "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
                            })
                print(f"  -> Successfully drew {anomaly_count} bounding box(es).")
            else:
                print(f"  -> SKIPPED: No matching .txt file found for '{raw_base}'.")

            # Save the annotated image
            annotated_path = os.path.join(processed_dir, f"annotated_{filename}")
            cv2.imwrite(annotated_path, annotated_img)
            count += 1

# Save report
with open(os.path.join(processed_dir, "debris_report.json"), "w") as jf:
    json.dump(reports_json, jf, indent=4)

print(f"Done! Processed {count} images total.")