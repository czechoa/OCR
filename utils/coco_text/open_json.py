import json
annotation_filename = "../../data/COCO_Text.json"

with open(annotation_filename) as json_file:
    coco_text = json.load(json_file)

print(coco_text["cats"])
print(coco_text.keys())
