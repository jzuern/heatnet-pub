from __future__ import print_function
import json
import  datasets.c_relabeller.mapping as mapping

iros_20_labels = ['road/parking', 'ground/sidewalk', 'building', 'curb', 'fence', 'pole/sign', 'vegetation',
                  'terrain', 'sky', 'person', 'car/bus/train/truck', 'bicycle/motorcycle', 'background', 'ignore']

with open('config_vistas.json') as config_file:
    config = json.load(config_file)
labels = config['labels']

for i, l in enumerate(labels):
    print('ID: %i: %s' % (i, l['name']))

print('################ Mapping ############################')

for i in range(15):
    mapped_names = ""
    for key, value in mapping.relabel_dict.items():
        if value == i:
            mapped_names += " / " + labels[key]['name']
    print('Map: %s - > %s' % (iros_20_labels[i] if i < len(iros_20_labels) else 'None', mapped_names))

