"""Average dimension class"""

from typing import List
import numpy as np
import os
import json

class DimensionAverages:
    """
    Class to calculate the average dimensions of the objects in the dataset.
    """
    def __init__(
        self, 
        categories: List[str] = ['car', 'pedestrian', 'cyclist'],
        save_file: str = 'dimension_averages.txt'
    ):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/' + save_file
        self.categories = categories

        if len(self.categories) == 0:
            self.load_items_from_file()

        for det in self.categories:
            cat_ = det.lower()
            if cat_ in self.dimension_map.keys():
                continue
            self.dimension_map[cat_] = {}
            self.dimension_map[cat_]['count'] = 0
            self.dimension_map[cat_]['total'] = np.zeros(3, dtype=np.float32)

    def add_items(self, items_path):
        for path in items_path:
            with open(path, "r") as f:
                for line in f:
                    line = line.split(" ")
                    if line[0].lower() in self.categories:
                        self.add_item(
                            line[0], 
                            np.array([float(line[8]), float(line[9]), float(line[10])])
                        )

    def add_item(self, cat, dim):
        cat = cat.lower()
        self.dimension_map[cat]['count'] += 1
        self.dimension_map[cat]['total'] += dim

    def get_item(self, cat):
        cat = cat.lower()
        return self.dimension_map[cat]['total'] / self.dimension_map[cat]['count']

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for cat in dimension_map:
            dimension_map[cat]['total'] = np.asarray(dimension_map[cat]['total'])

        self.dimension_map = dimension_map

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def recognized_class(self, cat):
        return cat.lower() in self.dimension_map

class ClassAverages:
    def __init__(self, classes=[]):
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        class_ = class_.lower()
        return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

    def dump_to_file(self):
        f = open(self.filename, "w")
        f.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        f.close()

    def load_items_from_file(self):
        f = open(self.filename, 'r')
        dimension_map = json.load(f)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        return class_.lower() in self.dimension_map

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self,obj)