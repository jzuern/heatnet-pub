# import glob, os
# base_path = "/home/vertensj/Documents/robocar_bags/data_collection/31.10.17/"
# os.chdir(base_path + "back_front/")
# back_front = []
# for file in glob.glob("*.bag"):
#     back_front.append(file)
# back_front.sort()
# print back_front

import fnmatch
import os
import gmplot

gmap = gmplot.GoogleMapPlotter(48.013551, 7.833116, 16)

matches = []
core_dir = "/mnt/hpc.vertensj/percepcar_data_3/heatmaps"
for root, dirnames, filenames in os.walk(core_dir):
    for filename in fnmatch.filter(filenames, 'heat_stats_*'):
        matches.append(os.path.join(root, filename))
print(matches)

heat_lat = []
heat_long = []

for f in matches:
    with open(f) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]

    for line in content:
        lat, longi = line.split(" ")
        heat_lat.append(float(lat))
        heat_long.append(float(longi))

if len(heat_lat) > 0:
    print("Draw google map with %d entries" % len(heat_lat))
    gmap.heatmap(heat_lat, heat_long)
    gmap.draw("/home/vertensj/heatmaps2.html")