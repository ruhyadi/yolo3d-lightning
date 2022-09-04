"""KITTI data loader"""

import copy
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src.utils.Calib import get_P

class KITTIDataLoader(Dataset):
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
        self.P2 = get_P(self.dataset_dir / "calib_kitti.txt")  # calibration matrix
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
        return len(self.images_path)

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
                dims_avg[current["name"]] += (
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

        # augmented image with flip
        flip = np.random.binomial(1, 0.5)
        if flip > 0.5:
            crop_img = cv2.flip(crop_img, 1)
        # transforms image
        crop_img = preprocess(crop_img)

        if flip > 0.5:
            return crop_img, [data["orient_flipped"], data["conf_flipped"], data["dims"]]
        else:
            return crop_img, [data["orient"], data["conf"], data["dims"]]

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
    """Testing KITTI Loader"""

    from torch.utils.data import DataLoader

    dataset = KITTIDataLoader(
        dataset_dir="./data/KITTI",
        dataset_sets_path="./data/KITTI/val_95.txt",
        bin=2,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    for i, (x, y) in enumerate(dataloader):
        print(x.shape)
        print("Orientation: ", y[0])
        print("Confidence: ", y[1])
        print("Dimension: ", y[2])
        break