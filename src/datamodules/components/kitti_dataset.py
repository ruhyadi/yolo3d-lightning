"""
Dataset modules for load kitti dataset and convert to yolo3d format
"""

import copy
import csv
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from src.utils import Calib as calib
from src.utils.ClassAverages import ClassAverages
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class KITTIDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "./data/KITTI",
        dataset_sets: str = "./data/KITTI/train.txt",  # [train.txt, val.txt]
        bins: int = 2,
        overlap: float = 0.1,
    ):
        super().__init__()

        # dataset path
        dataset_path = Path(dataset_path)
        self.image_path = dataset_path / "images"  # image_2
        self.label_path = dataset_path / "label_2"
        self.calib_path = dataset_path / "calib"
        self.global_calib = dataset_path / "calib_kitti.txt"
        self.dataset_sets = Path(dataset_sets)

        # set projection matrix
        self.proj_matrix = calib.get_P(self.global_calib)

        # index from images_path
        self.sets = open(self.dataset_sets, "r")
        self.ids = [id.split("\n")[0] for id in self.sets.readlines()]
        # self.ids = [x.split(".")[0] for x in sorted(os.listdir(self.image_path))]

        self.num_images = len(self.ids)

        # set ANGLE BINS
        self.bins = bins
        self.angle_bins = self.generate_bins(self.bins)
        self.interval = 2 * np.pi / self.bins
        self.overlap = overlap

        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(
                (
                    (i * self.interval - overlap) % (2 * np.pi),
                    (i * self.interval + self.interval + overlap) % (2 * np.pi),
                )
            )

        # AVERANGE num classes dataset
        # class_list same as in detector
        self.class_list = ["Car", "Pedestrian", "Cyclist", "Truck"]
        self.averages = ClassAverages(self.class_list)

        # list of object [id (000001), line_num]
        self.object_list = self.get_objects(self.ids)

        # label: contain image label params {bbox, dimension, etc}
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id
            self.labels[id][str(line_num)] = label

        # current id and image
        self.curr_id = ""
        self.curr_img = None

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            # read image (.png)
            self.curr_img = cv2.imread(str(self.image_path / f"{id}.png"))

        label = self.labels[id][str(line_num)]

        obj = DetectedObject(
            self.curr_img, label["Class"], label["Box_2D"], self.proj_matrix, label=label
        )

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    # def generate_sets(self, sets_file):
    #     with open(self.dataset_sets) as file:
    #         for line_num, line in enumerate(file):
    #             ids = line

    def generate_bins(self, bins):
        angle_bins = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            angle_bins[i] = i * interval
        angle_bins += interval / 2  # center of bins

        return angle_bins

    def get_objects(self, ids):
        """Get objects parameter from labels, like dimension and class name."""
        objects = []
        for id in ids:
            with open(self.label_path / f"{id}.txt") as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(" ")
                    obj_class = line[0]
                    if obj_class not in self.class_list:
                        continue

                    dimension = np.array(
                        [float(line[8]), float(line[9]), float(line[10])], dtype=np.double
                    )
                    self.averages.add_item(obj_class, dimension)

                    objects.append((id, line_num))

        self.averages.dump_to_file()
        return objects

    def get_label(self, id, line_num):
        lines = open(self.label_path / f"{id}.txt").read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2 * np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        line = line[:-1].split(" ")

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        # Alpha is orientation will be regressing
        # Alpha = [-pi, pi]
        Alpha = line[3]
        Ry = line[14]
        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        # Dimension: height, width, length
        Dimension = np.array([line[8], line[9], line[10]], dtype=np.double)
        # modify the average
        Dimension -= self.averages.get_item(Class)

        # Locattion: x, y, z
        Location = [line[11], line[12], line[13]]
        # bring the KITTI center up to the middle of the object
        Location[1] -= Dimension[0] / 2

        Orientation = np.zeros((self.bins, 2))
        Confidence = np.zeros(self.bins)

        # angle on range [0, 2pi]
        angle = Alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for bin_idx in bin_idxs:
            angle_diff = angle - self.angle_bins[bin_idx]

            Orientation[bin_idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            Confidence[bin_idx] = 1

        label = {
            "Class": Class,
            "Box_2D": Box_2D,
            "Dimensions": Dimension,
            "Alpha": Alpha,
            "Orientation": Orientation,
            "Confidence": Confidence,
        }

        return label

    def get_averages(self):
        dims_avg = {key: np.array([0, 0, 0]) for key in self.class_list}
        dims_count = {key: 0 for key in self.class_list}

        for i in range(len(os.listdir(self.image_path))):
            current_data = self.image_path[i]


class DetectedObject:
    """Processing image for NN input."""

    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        # check if proj_matrix is path
        if isinstance(proj_matrix, str):
            proj_matrix = calib.get_P(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        """Calculate global angle of object, see paper."""
        width = img.shape[1]
        # Angle of View: fovx (rad) => 3.14
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):
        # transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        process = transforms.Compose([transforms.ToTensor(), normalize])

        # crop image
        pt1, pt2 = box_2d[0], box_2d[1]
        crop = img[pt1[1] : pt2[1] + 1, pt1[0] : pt2[0] + 1]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_CUBIC)

        # apply transform for batch
        batch = process(crop)

        return batch


def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of bins

    return angle_bins


"""
KITTI DataLoader
Source: https://github.com/lzccccc/3d-bounding-box-estimation-for-autonomous-driving
"""


class KITTILoader(Dataset):
    def __init__(
        self,
        base_dir: str,
        KITTI_cat: List[str],
        bin: int = 2,
        overlap: float = 0.1,
        jit: int = 3,
        norm_w: int = 224,
        norm_h: int = 224,
        batch_size: int = 1,
    ):
        super(KITTILoader, self).__init__()

        self.base_dir = base_dir
        self.KITTI_cat = KITTI_cat
        self.bin = bin
        self.overlap = overlap

        label_dir = os.path.join(self.base_dir, "label_2")
        image_dir = os.path.join(self.base_dir, "images")  # images_2

        self.image_data = []
        self.images = []

        for i, fn in enumerate(os.listdir(label_dir)):
            label_full_path = os.path.join(label_dir, fn)
            image_full_path = os.path.join(image_dir, fn.replace(".txt", ".png"))

            self.images.append(image_full_path)
            fieldnames = ["type", "truncated", "occluded", "alpha", "xmin", "ymin", 
                        "xmax", "ymax", "dh", "dw", "dl", "lx", "ly", "lz", "ry"]
            with open(label_full_path, "r") as csv_file:
                reader = csv.DictReader(csv_file, delimiter=" ", fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    if row["type"] in self.KITTI_cat:
                        # if subset == 'training':
                        new_alpha = self.get_new_alpha(row["alpha"])
                        dimensions = np.array(
                            [float(row["dh"]), float(row["dw"]), float(row["dl"])]
                        )
                        annotation = {
                            "name": row["type"],
                            "image": image_full_path,
                            "xmin": int(float(row["xmin"])),
                            "ymin": int(float(row["ymin"])),
                            "xmax": int(float(row["xmax"])),
                            "ymax": int(float(row["ymax"])),
                            "dims": dimensions,
                            "new_alpha": new_alpha,
                        }

                        # elif subset == 'eval':
                        #     dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                        #     translations = np.array([float(row['lx']), float(row['ly']), float(row['lz'])])
                        #     annotation = {'name': row['type'], 'image': image_full_path,
                        #                   'alpha': float(row['alpha']),
                        #                   'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                        #                   'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                        #                   'dims': dimensions, 'trans': translations, 'rot_y': float(row['ry'])}

                        self.image_data.append(annotation)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, index):
        """Iterate items"""
        data = self.orientation_confidence_flip(self.image_data[index])

        return []

    def get_average_dimension(self):
        dims_avg = {key: np.array([0, 0, 0]) for key in self.KITTI_cat}
        dims_cnt = {key: 0 for key in self.KITTI_cat}

        for i in range(len(self.image_data)):
            current_data = self.image_data[i]
            if current_data["name"] in self.KITTI_cat:
                dims_avg[current_data["name"]] = (
                    dims_cnt[current_data["name"]] * dims_avg[current_data["name"]]
                    + current_data["dims"]
                )
                dims_cnt[current_data["name"]] += 1
                dims_avg[current_data["name"]] /= dims_cnt[current_data["name"]]
        return dims_avg, dims_cnt

    def get_new_alpha(self, alpha):
        """
        change the range of orientation from [-pi, pi] to [0, 2pi]
        :param alpha: original orientation in KITTI
        :return: new alpha
        """
        new_alpha = float(alpha) + np.pi / 2.0
        if new_alpha < 0:
            new_alpha = new_alpha + 2.0 * np.pi
            # make sure angle lies in [0, 2pi]
        new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

        return new_alpha

    def compute_anchors(self, angle):
        """
        compute angle offset and which bin the angle lies in
        input: fixed local orientation [0, 2pi]
        output: [bin number, angle offset]

        For two bins:

        if angle < pi, l = 0, r = 1
            if    angle < 1.65, return [0, angle]
            elif  pi - angle < 1.65, return [1, angle - pi]

        if angle > pi, l = 1, r = 2
            if    angle - pi < 1.65, return [1, angle - pi]
        elif     2pi - angle < 1.65, return [0, angle - 2pi]
        """
        anchors = []

        wedge = 2.0 * np.pi / self.bin  # 2pi / bin = pi
        l_index = int(angle / wedge)  # angle/pi
        r_index = l_index + 1

        # (angle - l_index*pi) < pi/2 * 1.05 = 1.65
        if (angle - l_index * wedge) < wedge / 2 * (1 + self.overlap / 2):
            anchors.append([l_index, angle - l_index * wedge])

        # (r*pi + pi - angle) < pi/2 * 1.05 = 1.65
        if (r_index * wedge - angle) < wedge / 2 * (1 + self.overlap / 2):
            anchors.append([r_index % self.bin, angle - r_index * wedge])

        return anchors

    def orientation_confidence_flip(self, image_data, dims_avg):
        for data in image_data:

            # minus the average dimensions
            data["dims"] = data["dims"] - dims_avg[data["name"]]

            # fix orientation and confidence for no flip
            orientation = np.zeros((self.bin, 2))
            confidence = np.zeros(self.bin)

            anchors = self.compute_anchors(data["new_alpha"])

            for anchor in anchors:
                # each angle is represented in sin and cos
                orientation[anchor[0]] = np.array(
                    [np.cos(anchor[1]), np.sin(anchor[1])]
                )
                confidence[anchor[0]] = 1

            confidence = confidence / np.sum(confidence)

            data["orient"] = orientation
            data["conf"] = confidence

            # Fix orientation and confidence for random flip
            orientation = np.zeros((self.bin, 2))
            confidence = np.zeros(self.bin)

            anchors = self.compute_anchors(
                2.0 * np.pi - data["new_alpha"]
            )  # compute orientation and bin
            # for flipped images

            for anchor in anchors:
                orientation[anchor[0]] = np.array(
                    [np.cos(anchor[1]), np.sin(anchor[1])]
                )
                confidence[anchor[0]] = 1

            confidence = confidence / np.sum(confidence)

            data["orient_flipped"] = orientation
            data["conf_flipped"] = confidence

        return image_data

    def prepare_input_and_output(self, train_inst, image_dir):
        """
        prepare image patch for training
        input:  train_inst -- input image for training
        output: img -- cropped bbox
                train_inst['dims'] -- object dimensions
                train_inst['orient'] -- object orientation (or flipped orientation)
                train_inst['conf_flipped'] -- orientation confidence
        """
        xmin = train_inst["xmin"] + np.random.randint(-self.jit, self.jit + 1)
        ymin = train_inst["ymin"] + np.random.randint(-self.jit, self.jit + 1)
        xmax = train_inst["xmax"] + np.random.randint(-self.jit, self.jit + 1)
        ymax = train_inst["ymax"] + np.random.randint(-self.jit, self.jit + 1)

        img = cv2.imread(image_dir)

        if self.jit != 0:
            xmin = max(xmin, 0)
            ymin = max(ymin, 0)
            xmax = min(xmax, img.shape[1] - 1)
            ymax = min(ymax, img.shape[0] - 1)

        img = copy.deepcopy(img[ymin : ymax + 1, xmin : xmax + 1]).astype(np.float32)

        # flip the image
        # 50% percent choose 1, 50% percent choose 0
        flip = np.random.binomial(1, 0.5)
        if flip > 0.5:
            img = cv2.flip(img, 1)

        # resize the image to standard size
        img = cv2.resize(img, (self.norm_h, self.norm_w))
        # minus the mean value in each channel
        img = img - np.array([[[103.939, 116.779, 123.68]]])

        ### Fix orientation and confidence
        if flip > 0.5:
            return (
                img,
                train_inst["dims"],
                train_inst["orient_flipped"],
                train_inst["conf_flipped"],
            )
        else:
            return img, train_inst["dims"], train_inst["orient"], train_inst["conf"]

    def data_gen(self, all_objs):
        """
        generate data for training
        input: all_objs -- all objects used for training
            batch_size -- number of images used for training at once
        yield: x_batch -- (batch_size, 224, 224, 3),  input images to training process at each batch
            d_batch -- (batch_size, 3),  object dimensions
            o_batch -- (batch_size, 2, 2), object orientation
            c_batch -- (batch_size, 2), angle confidence
        """
        num_obj = len(all_objs)

        keys = list(range(num_obj))
        np.random.shuffle(keys)

        l_bound = 0
        r_bound = self.batch_size if self.batch_size < num_obj else num_obj

        while True:
            if l_bound == r_bound:
                l_bound = 0
                r_bound = self.batch_size if self.batch_size < num_obj else num_obj
                np.random.shuffle(keys)

            currt_inst = 0
            x_batch = np.zeros((r_bound - l_bound, 224, 224, 3))
            d_batch = np.zeros((r_bound - l_bound, 3))
            o_batch = np.zeros((r_bound - l_bound, self.bin, 2))
            c_batch = np.zeros((r_bound - l_bound, self.bin))

            for key in keys[l_bound:r_bound]:
                # augment input image and fix object's orientation and confidence
                (
                    image,
                    dimension,
                    orientation,
                    confidence,
                ) = self.prepare_input_and_output(
                    all_objs[key],
                    all_objs[key]["image"],
                )

                x_batch[currt_inst, :] = image
                d_batch[currt_inst, :] = dimension
                o_batch[currt_inst, :] = orientation
                c_batch[currt_inst, :] = confidence

                currt_inst += 1

            yield x_batch, [d_batch, o_batch, c_batch]

            l_bound = r_bound
            r_bound = r_bound + self.batch_size

            if r_bound > num_obj:
                r_bound = num_obj


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    # KITTI DATASET LOADER
    # dataset = KITTIDataset()
    # train_loader = DataLoader(dataset, 1)
    # for img, label in train_loader:
    #     print(img.shape)
    #     break

    dataset = KITTILoader(
        base_dir="./data/KITTI",
        KITTI_cat=["Car", "Van", "Truck", "Pedestrian", "Cyclist"],
    )

    print(dataset.get_average_dimension())
