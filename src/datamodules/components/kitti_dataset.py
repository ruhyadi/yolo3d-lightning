"""
Dataset modules for load kitti dataset and convert to yolo3d format
"""

import csv
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np
from src.utils import Calib as calib
from src.utils.averages import ClassAverages, DimensionAverages
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class KITTIDataset3(Dataset):
    """KITTI dataset loader"""
    def __init__(
        self,
        dataset_dir: str = "./data/KITTI",
        dataset_sets: str = "./data/KITTI/train.txt", # or val.txt
        bins: int = 2,
        overlap: float = 0.1,
        image_size: int = 224,
        categories: List[str] = ["car", "pedestrian", "cyclist"],
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        with open(dataset_sets, "r") as f:
            self.ids = [id.split("\n")[0] for id in f.readlines()]
        self.bins = bins
        self.overlap = overlap
        self.image_size = image_size
        self.categories = categories

        self.images_dir = self.dataset_dir / "images" # image_2
        self.labels_dir = self.dataset_dir / "label_2"
        self.P2 = calib.get_P(self.dataset_dir / "calib_kitti.txt") # calibration matrix P2

        # get images and labels paths
        self.images_path = [self.images_dir / (id + ".png") for id in self.ids]
        self.labels_path = [self.labels_dir / (id + ".txt") for id in self.ids]

        # get dimension average for every object in categories
        self.dimensions_averages = DimensionAverages(self.categories)
        self.dimensions_averages.add_items(self.labels_path)

        # KITTI fieldnames
        self.fieldnames = [
            "type", "truncated", "occluded", "alpha",
            "xmin", "ymin", "xmax", "ymax", "dh", "dw", "dl",
            "lx", "ly", "lz", "ry"
            ]

        # get images data
        self.images_data = self.preprocess_labels(self.labels_path)

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        """Data loader looper"""
        image_data = self.images_data[idx]

        image = self.preprocess_image(image_data)
        orientation = image_data["orientation"]
        confidence = image_data["confidence"]
        dimensions = image_data["dimensions"]
        
        return image, {"orientation": orientation, "confidence": confidence, "dimensions": dimensions}

    def preprocess_labels(self, labels_path: str):
        """
        > Preprocessing labels for yolo3d format.
        The function takes in a list of paths to the labels, 
        and returns a list of dictionaries, where
        each dictionary contains the information for one image
        Args:
          labels_path (str): The path to the labels file.
        Returns:
          A list of dictionaries, each dictionary contains the information of one object in one image.
        """
        IMAGES_DATA = []
        # generate angle bins, center of each bin [pi/2, 3pi/2] for 2 bin
        center_bins = self.generate_bins(self.bins)
        # initialize orientation and confidence

        for path in labels_path:
            with open(path, "r") as f:
                reader = csv.DictReader(f, delimiter=" ", fieldnames=self.fieldnames)
                for line, row in enumerate(reader):
                    if row["type"].lower() in self.categories:
                        orientation = np.zeros((self.bins, 2))
                        confidence = np.zeros(self.bins)
                        # convert from [-pi, pi] to [0, 2pi]
                        angle = float(row["alpha"]) + np.pi # or new_alpha
                        bin_idxs = self.get_bin_idxs(angle)
                        # update orientation and confidence
                        for idx in bin_idxs:
                            angle_diff = angle - center_bins[idx]
                            orientation[idx, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
                            confidence[idx] = 1
                        # averaging dimensions
                        dimensions = np.array([float(row["dh"]), float(row["dw"]), float(row["dl"])])
                        dimensions -= self.dimensions_averages.get_item(row["type"])

                        image_data = {
                            "name": row["type"],
                            "image_path": self.images_dir / (path.name.split(".")[0] + ".png"),
                            "xmin": int(float(row["xmin"])),
                            "ymin": int(float(row["ymin"])),
                            "xmax": int(float(row["xmax"])),
                            "ymax": int(float(row["ymax"])),
                            "alpha": float(row["alpha"]),
                            "orientation": orientation,
                            "confidence": confidence,
                            "dimensions": dimensions
                        }

                        IMAGES_DATA.append(image_data)
        
        return IMAGES_DATA

    def preprocess_image(self, image_data: dict):
        """
        It takes an image and a bounding box, crops the image to the bounding box, 
        resizes the cropped image
        to the size of the input to the model, and then normalizes the image
        Args:
          image_data (dict): a dictionary containing the following keys:
        Returns:
          A tensor of the image
        """
        image = cv2.imread(str(image_data["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        crop = image[image_data["ymin"]:image_data["ymax"]+1, image_data["xmin"]:image_data["xmax"]+1]
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transform(crop)

    def generate_bins(self, bins):
        """
        It takes the number of bins you want to use and returns 
        an array of angles that are the centers of those bins
        Args:
          bins: number of bins to use for the histogram
        Returns:
          the angle_bins array.
        """
        angle_bins = np.zeros(bins)
        interval = 2 * np.pi / bins
        for i in range(1, bins):
            angle_bins[i] = i * interval
        angle_bins += interval / 2  # center of bins

        return angle_bins

    def get_bin_idxs(self, angle):
        """
        It takes an angle and returns the indices of the bins that the angle falls into
        Args:
          angle: the angle of the line
        Returns:
          The bin_idxs are being returned.
        """
        interval = 2 * np.pi / self.bins

        # range of bins from [0, 2pi]
        bin_ranges = []
        for i in range(0, self.bins):
            bin_ranges.append((
                (i * (interval - self.overlap)) % (2 * np.pi),
                ((i * interval) + interval + self.overlap) % (2 * np.pi)
            ))

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2 * np.pi
            angle = (angle - min) if (angle - min) > 0 else (angle - min) + 2 * np.pi
            return angle < max

        bin_idxs = []
        for bin_idx, bin_range in enumerate(bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def get_average_dimension(self, labels_path: str):
        """
        > For each line in the labels file, 
        if the object type is in the categories list, add the dimensions
        to the dimensions_averages object
        Args:
          labels_path (str): The path to the labels file.
        """
        for path in labels_path:
            with open(path, "r") as f:
                reader = csv.DictReader(f, delimiter=" ", fieldnames=self.fieldnames)
                for line, row in enumerate(reader):
                    if row["type"] in self.categories:
                        self.dimensions_averages.add_item(
                            row["type"], 
                            row["dh"], 
                            row["dw"], 
                            row["dl"]
                        )


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

class KITTIDataset2(Dataset):
    """KITTI Data Loader"""

    def __init__(
        self,
        dataset_dir: str = "./data//KITTI",
        dataset_sets_path: str = "./data/KITTI/train.txt",
        bin: int = 2,
        overlap: float = 0.5,
        image_size: int = 224,
        categories: list = ["Car", "Pedestrian", "Cyclist", "Van", "Truck"],
    ) -> None:
        """Initialize dataset"""
        super().__init__()
        # arguments
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"  # image_2
        self.labels_dir = self.dataset_dir / "label_2"  # label_2
        self.P2 = calib.get_P(self.dataset_dir / "calib_kitti.txt")  # calibration matrix
        with open(dataset_sets_path, "r") as f:
            self.dataset_sets = [id.split("\n")[0] for id in f.readlines()]  # sets training/validation/test sets
        self.bin = bin  # binning factor
        self.overlap = overlap  # overlap factor
        self.image_size = image_size  # image size
        self.categories = categories  # object categories

        # get image and label paths
        self.images_path = self.get_paths(self.images_dir)
        self.labels_path = self.get_paths(self.labels_dir)
        # get label annotations data
        self.images_data = self.get_label_annos(self.labels_path)
        # get dimension average
        self.dims_avg, self.dims_cnt = self.get_average_dimension(self.images_data)
        # get orientation, confidence, and augmented annotations
        self.images_data = self.orientation_confidence_flip(self.images_data, self.dims_avg)

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        """Get item"""
        # preprocessing and augmenting data
        image, label = self.get_augmentation(self.images_data[idx])

        return image, label

    def get_paths(self, dir: Path) -> List[Path]:
        """Get image and label paths"""
        return [
            path
            for path in dir.iterdir()
            if path.name.split(".")[0] in self.dataset_sets
        ]

    def get_label_annos(self, labels_path) -> list:
        """Get label annotations"""
        IMAGES_DATA = []
        fieldnames = [
            "type", "truncated", "occluded", "alpha",
            "xmin", "ymin", "xmax", "ymax", "dh", "dw", "dl",
            "lx", "ly", "lz", "ry"
            ]
        for path in labels_path:
            with open(path, "r") as f:
                reader = csv.DictReader(f, delimiter=" ", fieldnames=fieldnames)
                for line, row in enumerate(reader):
                    if row["type"] in self.categories:
                        new_alpha = self.get_new_alpha(row["alpha"])
                        dimensions = np.array([float(row["dh"]), float(row["dw"]), float(row["dl"])])
                        image_data = {
                            "name": row["type"],
                            "image": self.images_dir / (path.name.split(".")[0] + ".png"),
                            "xmin": int(float(row["xmin"])),
                            "ymin": int(float(row["ymin"])),
                            "xmax": int(float(row["xmax"])),
                            "ymax": int(float(row["ymax"])),
                            "dims": dimensions,
                            "new_alpha": new_alpha,
                        }
                        IMAGES_DATA.append(image_data)

        return IMAGES_DATA

    def get_new_alpha(self, alpha: float):
        """
        Change the range of orientation from [-pi, pi] to [0, 2pi]
        :param alpha: original orientation in KITTI
        :return: new alpha
        """
        new_alpha = float(alpha) + np.pi / 2.0
        if new_alpha < 0:
            new_alpha = new_alpha + 2.0 * np.pi
            # make sure angle lies in [0, 2pi]
        new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

        return new_alpha

    def get_average_dimension(self, images_data: list = None) -> tuple:
        """Get average dimension for every object in categories"""
        dims_avg = {key: np.array([0.0, 0.0, 0.0]) for key in self.categories}
        dims_cnt = {key: 0 for key in self.categories}

        for i in range(len(images_data)):
            current = images_data[i]
            if current["name"] in self.categories:
                dims_avg[current["name"]] = (
                    dims_cnt[current["name"]] * dims_avg[current["name"]] 
                    + current["dims"]
                )
                dims_cnt[current["name"]] += 1
                dims_avg[current["name"]] /= dims_cnt[current["name"]]

        return [dims_avg, dims_cnt]

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

    def orientation_confidence_flip(self, images_data, dims_avg):
        """Generate orientation, confidence and augment with flip"""
        for data in images_data:
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

        return images_data

    def get_augmentation(self, data):
        """
        Preprocess image and augmentation
        input: image_data
        output: image, bounding box
        """
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([transforms.ToTensor(), normalizer])
        xmin = data["xmin"]
        ymin = data["ymin"]
        xmax = data["xmax"]
        ymax = data["ymax"]

        # read and crop image
        img = cv2.imread(str(data["image"]))
        crop_img = img[ymin : ymax + 1, xmin : xmax + 1]
        crop_img = cv2.resize(crop_img, (self.image_size, self.image_size))

        # NOTE: Disable Augmentation
        # augmented image with flip
        # flip = np.random.random(1, 0.5)
        # if flip > 0.5:
        #     crop_img = cv2.flip(crop_img, 1)
        # transforms image
        crop_img = preprocess(crop_img)

        # if flip > 0.5:
        #     return (
        #         crop_img,
        #         {"orientation": data["orient_flipped"],
        #          "confidence": data["conf_flipped"],
        #          "dimensions": data["dims"],
        #         },
        #     )
        # else:
        return (
            crop_img,
            {"orientation": data["orient"],
                "confidence": data["conf"],
                "dimensions": data["dims"],
            },
        )


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    # KITTI DATASET LOADER
    # dataset = KITTIDataset()
    # train_loader = DataLoader(dataset, 1)
    # for img, label in train_loader:
    #     print(img.shape)
    #     print(label)
    #     break

    # output
    # torch.Size([1, 3, 224, 224])
    # {'Class': ['Pedestrian'], 'Box_2D': [[tensor([712]), tensor([143])], [tensor([811]), tensor([308])]], 'Dimensions': tensor([[ 0.1223, -0.1478,  0.3820]], dtype=torch.float64), 'Alpha': tensor([-0.2000], dtype=torch.float64), 'Orientation': tensor([[[0.1987, 0.9801],
    #         [0.0000, 0.0000]]], dtype=torch.float64), 'Confidence': tensor([[1., 0.]], dtype=torch.float64)}

    # dataset = KITTIDataset2(
    #     dataset_dir="./data/KITTI",
    #     dataset_sets_path="./data/KITTI/train_80.txt",
    #     bin=2,
    # )

    # dataloader = DataLoader(
    #     dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    # )
    # print(len(dataset))
    # print(len(dataloader))

    # for i, (x, y) in enumerate(dataloader):
    #     print(x.shape)
    #     print("Orientation: ", y["orientation"])
    #     print("Confidence: ", y["confidence"])
    #     print("Dimensions: ", y["dimensions"])
    #     break

    # output
    # torch.Size([1, 3, 224, 224])
    # Orientation:  tensor([[[ 0.0000,  0.0000],
    #         [ 0.9174, -0.3979]]], dtype=torch.float64)
    # Confidence:  tensor([[0., 1.]], dtype=torch.float64)
    # Dimensions:  tensor([[-7.2352, -2.6087, -7.1447]], dtype=torch.float64)

    # kitti dataset3
    dataset = KITTIDataset3(
        dataset_dir="./data/KITTI",
        dataset_sets="./data/KITTI/val_95.txt",
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    for x, y in dataloader:
        print(x.shape)
        print("Orientation: ", y["orientation"])
        print("Confidence: ", y["confidence"])
        print("Dimensions: ", y["dimensions"])
        break