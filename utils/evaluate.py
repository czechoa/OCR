import json
from json import JSONDecodeError

import numpy as np

from utils.coco_text import coco_evaluation


def save_result(generated_text, bboxes, img, append=False):
    print('save result')
    json_outputs = []

    if len(generated_text) == 0:
        generated_text.append("")

    if len(bboxes) == 0:
        bboxes.append(np.array([0, 0, 0, 0]))

    for text, bbox in zip(generated_text, bboxes):
        json_out = {}
        json_out["utf8_string"] = text
        json_out["image_id"] = img["id"]
        json_out["bbox"] = list(bbox.astype(float))
        json_outputs.append(json_out)

    # file_mode = 'a' if append else 'w'

    try:
        with open("our_results_test.json") as fp:
            listObj = json.load(fp)
    except JSONDecodeError:
        listObj = []

    listObj.extend(json_outputs)

    with open("our_results_test.json", 'w') as json_file:
        json.dump(listObj, json_file, indent=2,
                  )

    return json_outputs


def evaluation_and_end_to_end_results(ct, output_file='our_results_test.json'):
    our_results = ct.loadRes(output_file)

    our_detections = coco_evaluation.getDetections(ct, our_results, imgIds=list(our_results.imgs.keys()),
                                                   detection_threshold=0.5)
    # our_detections
    print('True positives have a ground truth id and an evaluation id: ', our_detections['true_positives'][0])
    print('False positives only have an evaluation id: ',
          our_detections['false_positives'][0] if len(our_detections['false_positives']) > 1 else [])
    print('True negatives only have a ground truth id: ',
          our_detections['false_negatives'][0] if len(our_detections['false_negatives']) > 1 else [])

    our_endToEnd_results = coco_evaluation.evaluateEndToEnd(ct, our_results, imgIds=list(our_results.imgs.keys()),
                                                            detection_threshold=0.5)
    coco_evaluation.printDetailedResults(ct, our_detections, our_endToEnd_results, 'our approach')
