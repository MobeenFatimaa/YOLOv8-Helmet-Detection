import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

IMAGES_DIR = "raw_data/images"
ANNOTATIONS_DIR = "raw_data/annotations"
OUTPUT_DIR = "dataset"

for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# We will dynamically track unique classes found
found_classes = set()

def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def parse_xml_to_yolo(xml_path, txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(txt_path, 'w') as out_file:
        for obj in root.findall('object'):
            cls = obj.find('name').text.strip()
            found_classes.add(cls.lower())
            
            # Map variations safely to index 0 ('Without helmet') or 1 ('With helmet')
            cls_lower = cls.lower()
            if "without" in cls_lower:
                cls_id = 0
            elif "with" in cls_lower:
                cls_id = 1
            else:
                continue

            xmlbox = obj.find('bndbox')
            b = (
                float(xmlbox.find('xmin').text),
                float(xmlbox.find('xmax').text),
                float(xmlbox.find('ymin').text),
                float(xmlbox.find('ymax').text)
            )
            bb = convert_box((w, h), b)
            out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

def process_split(img_list, split_name):
    count = 0
    for img_name in img_list:
        base_name, _ = os.path.splitext(img_name)
        xml_name = base_name + ".xml"
        
        src_img = os.path.join(IMAGES_DIR, img_name)
        src_xml = os.path.join(ANNOTATIONS_DIR, xml_name)
        
        if not os.path.exists(src_xml):
            continue
            
        dst_img = os.path.join(OUTPUT_DIR, split_name, "images", img_name)
        dst_txt = os.path.join(OUTPUT_DIR, split_name, "labels", base_name + ".txt")
        
        shutil.copy(src_img, dst_img)
        parse_xml_to_yolo(src_xml, dst_txt)
        count += 1
    print(f"Processed {count} images for {split_name}")

process_split(train_imgs, "train")
process_split(val_imgs, "val")
print(f"Conversion complete! Unique XML classes detected: {found_classes}")
