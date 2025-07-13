from ultralytics.data.split import autosplit
autosplit("coco128/",(0.85, 0.15, 0))