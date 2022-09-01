import os
from matplotlib.path import Path
import cv2
from PIL import Image
import numpy as np
from enum import Enum
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from src.utils.utils import detectionInfo, save_file

from src.utils.Calib import *
from src.utils.Math import *

# from .Calib import *
# from .Math import *
from src.utils import Calib


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)


def constraint_to_color(constraint_idx):
    return {
        0: cv_colors.PURPLE.value,  # left
        1: cv_colors.ORANGE.value,  # top
        2: cv_colors.MINT.value,  # right
        3: cv_colors.YELLOW.value,  # bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2] / point[2]
    point = point.astype(np.int16)

    return point


# take in 3d points and plot them on image as red circles
def plot_3d_pts(
    img,
    pts,
    center,
    calib_file=None,
    cam_to_img=None,
    relative=False,
    constraint_idx=None,
):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j, i in enumerate(pt)]  # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)


def plot_3d_box(img, cam_to_img, ry, dimension, center):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)

    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    # LINE
    cv2.line(
        img,
        (box_3d[0][0], box_3d[0][1]),
        (box_3d[2][0], box_3d[2][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[4][0], box_3d[4][1]),
        (box_3d[6][0], box_3d[6][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[0][0], box_3d[0][1]),
        (box_3d[4][0], box_3d[4][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[2][0], box_3d[2][1]),
        (box_3d[6][0], box_3d[6][1]),
        cv_colors.GREEN.value,
        2,
    )

    cv2.line(
        img,
        (box_3d[1][0], box_3d[1][1]),
        (box_3d[3][0], box_3d[3][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[1][0], box_3d[1][1]),
        (box_3d[5][0], box_3d[5][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[7][0], box_3d[7][1]),
        (box_3d[3][0], box_3d[3][1]),
        cv_colors.GREEN.value,
        2,
    )
    cv2.line(
        img,
        (box_3d[7][0], box_3d[7][1]),
        (box_3d[5][0], box_3d[5][1]),
        cv_colors.GREEN.value,
        2,
    )

    for i in range(0, 7, 2):
        cv2.line(
            img,
            (box_3d[i][0], box_3d[i][1]),
            (box_3d[i + 1][0], box_3d[i + 1][1]),
            cv_colors.GREEN.value,
            2,
        )

    # frame to drawing polygon
    frame = np.zeros_like(img, np.uint8)

    # front side
    cv2.fillPoly(
        frame,
        np.array(
            [[[box_3d[0]], [box_3d[1]], [box_3d[3]], [box_3d[2]]]], dtype=np.int32
        ),
        cv_colors.BLUE.value,
    )

    alpha = 0.5
    mask = frame.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]


def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)


def calc_theta_ray(img_width, box_2d, proj_matrix):
    """Calculate global angle of object, see paper."""

    # check if proj_matrix is path
    if isinstance(proj_matrix, str):
        proj_matrix = Calib.get_P(proj_matrix)

    # Angle of View: fovx (rad) => 3.14
    fovx = 2 * np.arctan(img_width / (2 * proj_matrix[0][0]))
    # center_x = (box_2d[1][0] + box_2d[0][0]) / 2
    center_x = ((box_2d[2] - box_2d[0]) / 2) + box_2d[0]
    dx = center_x - (img_width / 2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan((2 * dx * np.tan(fovx / 2)) / img_width)
    angle = angle * mult

    return angle


def calc_alpha(orient, conf, bins=2):
    angle_bins = generate_bins(bins=bins)

    argmax = np.argmax(conf)
    orient = orient[argmax, :]
    cos = orient[0]
    sin = orient[1]
    alpha = np.arctan2(sin, cos)
    alpha += angle_bins[argmax]
    alpha -= np.pi

    return alpha


def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of bins

    return angle_bins


class Plot3DBox:
    """
    Plotting 3DBox
    source: https://github.com/lzccccc/3d-bounding-box-estimation-for-autonomous-driving
    """

    def __init__(
        self,
        image_path: str = None,
        pred_path: str = None,
        label_path: str = None,
        calib_path: str = None,
        vehicle_list: list = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"],
        mode: str = "training",
        save_path: str = None,
    ) -> None:

        self.image_path = image_path
        self.pred_path = pred_path
        self.label_path = label_path if label_path is not None else pred_path
        self.calib_path = calib_path
        self.vehicle_list = vehicle_list
        self.mode = mode
        self.save_path = save_path

        self.dataset = [name.split('.')[0] for name in sorted(os.listdir(self.image_path))]
        self.start_frame = 0
        self.end_frame = len(self.dataset)

    def compute_birdviewbox(self, line, shape, scale):
        npline = [np.float64(line[i]) for i in range(1, len(line))]
        h = npline[7] * scale
        w = npline[8] * scale
        l = npline[9] * scale
        x = npline[10] * scale
        y = npline[11] * scale
        z = npline[12] * scale
        rot_y = npline[13]

        R = np.array([[-np.cos(rot_y), np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
        t = np.array([x, z]).reshape(1, 2).T

        x_corners = [0, l, l, 0]  # -l/2
        z_corners = [w, w, 0, 0]  # -w/2

        x_corners += -w / 2
        z_corners += -l / 2

        # bounding box in object coordinate
        corners_2D = np.array([x_corners, z_corners])
        # rotate
        corners_2D = R.dot(corners_2D)
        # translation
        corners_2D = t - corners_2D
        # in camera coordinate
        corners_2D[0] += int(shape / 2)
        corners_2D = (corners_2D).astype(np.int16)
        corners_2D = corners_2D.T

        return np.vstack((corners_2D, corners_2D[0, :]))

    def draw_birdeyes(self, ax2, line_gt, line_p, shape):
        # shape = 900
        scale = 15

        pred_corners_2d = self.compute_birdviewbox(line_p, shape, scale)
        gt_corners_2d = self.compute_birdviewbox(line_gt, shape, scale)

        codes = [Path.LINETO] * gt_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(gt_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color="orange", label="ground truth")
        ax2.add_patch(p)

        codes = [Path.LINETO] * pred_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(pred_corners_2d, codes)
        p = patches.PathPatch(pth, fill=False, color="green", label="prediction")
        ax2.add_patch(p)

    def compute_3Dbox(self, P2, line):
        obj = detectionInfo(line)
        # Draw 2D Bounding Box
        xmin = int(obj.xmin)
        xmax = int(obj.xmax)
        ymin = int(obj.ymin)
        ymax = int(obj.ymax)
        # width = xmax - xmin
        # height = ymax - ymin
        # box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color='red', linewidth='3')
        # ax.add_patch(box_2d)

        # Draw 3D Bounding Box

        R = np.array(
            [
                [np.cos(obj.rot_global), 0, np.sin(obj.rot_global)],
                [0, 1, 0],
                [-np.sin(obj.rot_global), 0, np.cos(obj.rot_global)],
            ]
        )

        x_corners = [0, obj.l, obj.l, obj.l, obj.l, 0, 0, 0]  # -l/2
        y_corners = [0, 0, obj.h, obj.h, 0, 0, obj.h, obj.h]  # -h
        z_corners = [0, 0, 0, obj.w, obj.w, obj.w, obj.w, 0]  # -w/2

        x_corners = [i - obj.l / 2 for i in x_corners]
        y_corners = [i - obj.h for i in y_corners]
        z_corners = [i - obj.w / 2 for i in z_corners]

        corners_3D = np.array([x_corners, y_corners, z_corners])
        corners_3D = R.dot(corners_3D)
        corners_3D += np.array([obj.tx, obj.ty, obj.tz]).reshape((3, 1))

        corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
        corners_2D = P2.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]
        corners_2D = corners_2D[:2]

        return corners_2D

    def draw_3Dbox(self, ax, P2, line, color):

        corners_2D = self.compute_3Dbox(P2, line)

        # draw all lines through path
        # https://matplotlib.org/users/path_tutorial.html
        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
        verts = bb3d_on_2d_lines_verts.T
        codes = [Path.LINETO] * verts.shape[0]
        codes[0] = Path.MOVETO
        # codes[-1] = Path.CLOSEPOLYq
        pth = Path(verts, codes)
        p = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

        width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
        height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
        # put a mask on the front
        front_fill = patches.Rectangle(
            (corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4
        )
        ax.add_patch(p)
        ax.add_patch(front_fill)

    def visualization(self):

        for index in range(self.start_frame, self.end_frame):
            image_file = os.path.join(self.image_path, self.dataset[index] + ".png")
            label_file = os.path.join(self.label_path, self.dataset[index] + ".txt")
            prediction_file = os.path.join(self.pred_path, self.dataset[index] + ".txt")
            if self.calib_path.endswith(".txt"):
                calibration_file = self.calib_path
            else:
                calibration_file = os.path.join(self.calib_path, self.dataset[index] + ".txt")
            for line in open(calibration_file):
                if "P2" in line:
                    P2 = line.split(" ")
                    P2 = np.asarray([float(i) for i in P2[1:]])
                    P2 = np.reshape(P2, (3, 4))

            fig = plt.figure(figsize=(20.00, 5.12), dpi=100)

            # fig.tight_layout()
            gs = GridSpec(1, 4)
            gs.update(wspace=0)  # set the spacing between axes.

            ax = fig.add_subplot(gs[0, :3])
            ax2 = fig.add_subplot(gs[0, 3:])

            # with writer.saving(fig, "kitti_30_20fps.mp4", dpi=100):
            image = Image.open(image_file).convert("RGB")
            shape = 900
            birdimage = np.zeros((shape, shape, 3), np.uint8)

            with open(label_file) as f1, open(prediction_file) as f2:
                for line_gt, line_p in zip(f1, f2):
                    line_gt = line_gt.strip().split(" ")
                    line_p = line_p.strip().split(" ")

                    truncated = np.abs(float(line_p[1]))
                    occluded = np.abs(float(line_p[2]))
                    trunc_level = 1 if self.mode == "training" else 255

                    # truncated object in dataset is not observable
                    if line_p[0].lower() in self.vehicle_list and truncated < trunc_level:
                        color = "green"
                        if line_p[0] == "Cyclist":
                            color = "yellow"
                        elif line_p[0] == "Pedestrian":
                            color = "cyan"
                        self.draw_3Dbox(ax, P2, line_p, color)
                        self.draw_birdeyes(ax2, line_gt, line_p, shape)

            # visualize 3D bounding box
            ax.imshow(image)
            ax.set_xticks([])  # remove axis value
            ax.set_yticks([])

            # plot camera view range
            x1 = np.linspace(0, shape / 2)
            x2 = np.linspace(shape / 2, shape)
            ax2.plot(x1, shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5)
            ax2.plot(x2, x2 - shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5)
            ax2.plot(shape / 2, 0, marker="+", markersize=16, markeredgecolor="red")

            # visualize bird eye view
            ax2.imshow(birdimage, origin="lower")
            ax2.set_xticks([])
            ax2.set_yticks([])
            # add legend
            handles, labels = ax2.get_legend_handles_labels()
            legend = ax2.legend(
                [handles[0], handles[1]],
                [labels[0], labels[1]],
                loc="lower right",
                fontsize="x-small",
                framealpha=0.2,
            )
            for text in legend.get_texts():
                plt.setp(text, color="w")

            if self.save_path is None:
                plt.show()
            else:
                fig.savefig(
                    os.path.join(self.save_path, self.dataset[index]),
                    dpi=fig.dpi,
                    bbox_inches="tight",
                    pad_inches=0,
                )
            # video_writer.write(np.uint8(fig))

class Plot3DBoxBev:
    """Plot 3D bounding box and bird eye view"""
    def __init__(
        self,
        proj_matrix = None, # projection matrix P2
        object_list = ["car", "pedestrian", "truck", "cyclist", "motorcycle", "bus"],
        
    ) -> None:

        self.proj_matrix = proj_matrix
        self.object_list = object_list

        self.fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
        gs = GridSpec(1, 4)
        gs.update(wspace=0)
        self.ax = self.fig.add_subplot(gs[0, :3])
        self.ax2 = self.fig.add_subplot(gs[0, 3:])

        self.shape = 900
        self.scale = 15

        self.COLOR = {
            "car": "blue",
            "pedestrian": "green",
            "truck": "yellow",
            "cyclist": "red",
            "motorcycle": "cyan",
            "bus": "magenta",
        }

    def compute_bev(self, dim, loc, rot_y):
        """compute bev"""
        # convert dimension, location and rotation
        h = dim[0] * self.scale
        w = dim[1] * self.scale
        l = dim[2] * self.scale
        x = loc[0] * self.scale
        y = loc[1] * self.scale
        z = loc[2] * self.scale
        rot_y = np.float64(rot_y)

        R = np.array([[-np.cos(rot_y), np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
        t = np.array([x, z]).reshape(1, 2).T
        x_corners = [0, l, l, 0]  # -l/2
        z_corners = [w, w, 0, 0]  # -w/2
        x_corners += -w / 2
        z_corners += -l / 2
        # bounding box in object coordinate
        corners_2D = np.array([x_corners, z_corners])
        # rotate
        corners_2D = R.dot(corners_2D)
        # translation
        corners_2D = t - corners_2D
        # in camera coordinate
        corners_2D[0] += int(self.shape / 2)
        corners_2D = (corners_2D).astype(np.int16)
        corners_2D = corners_2D.T

        return np.vstack((corners_2D, corners_2D[0, :]))

    def draw_bev(self, dim, loc, rot_y):
        """draw bev"""

        # gt_corners_2d = self.compute_bev(self.gt_dim, self.gt_loc, self.gt_rot_y)
        pred_corners_2d = self.compute_bev(dim, loc, rot_y)

        codes = [Path.LINETO] * pred_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(pred_corners_2d, codes)
        patch = patches.PathPatch(pth, fill=False, color="green", label="prediction")
        self.ax2.add_patch(patch)

    def compute_3dbox(self, bbox, dim, loc, rot_y):
        """compute 3d box"""
        # 2d bounding box
        xmin, ymin = int(bbox[0]), int(bbox[1])
        xmax, ymax = int(bbox[2]), int(bbox[3])

        # convert dimension, location
        h, w, l = dim[0], dim[1], dim[2]
        x, y, z = loc[0], loc[1], loc[2]

        R = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
        x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
        y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
        z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

        x_corners += -l / 2
        y_corners += -h
        z_corners += -w / 2

        corners_3D = np.array([x_corners, y_corners, z_corners])
        corners_3D = R.dot(corners_3D)
        corners_3D += np.array([x, y, z]).reshape(3, 1)

        corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
        corners_2D = self.proj_matrix.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]
        corners_2D = corners_2D[:2]

        return corners_2D

    def draw_3dbox(self, class_object, bbox, dim, loc, rot_y):
        """draw 3d box"""
        color = self.COLOR[class_object]
        corners_2D = self.compute_3dbox(bbox, dim, loc, rot_y)

        # draw all lines through path
        # https://matplotlib.org/users/path_tutorial.html
        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
        verts = bb3d_on_2d_lines_verts.T
        codes = [Path.LINETO] * verts.shape[0]
        codes[0] = Path.MOVETO
        pth = Path(verts, codes)
        patch = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

        width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
        height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
        # put a mask on the front
        front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
        self.ax.add_patch(patch)
        self.ax.add_patch(front_fill)

    def plot(
        self,
        img = None,
        class_object: str = None,
        bbox = None, # bbox 2d [xmin, ymin, xmax, ymax]
        dim = None, # dimension of the box (l, w, h)
        loc = None, # location of the box (x, y, z)
        rot_y = None, # rotation of the box around y-axis
    ):
        """plot 3d bbox and bev"""
        # initialize bev image
        bev_img = np.zeros((self.shape, self.shape, 3), np.uint8)

        # loop through all detections
        if class_object in self.object_list:
            self.draw_3dbox(class_object, bbox, dim, loc, rot_y)
            self.draw_bev(dim, loc, rot_y)

        # visualize 3D bounding box
        self.ax.imshow(img)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, self.shape / 2)
        x2 = np.linspace(self.shape / 2, self.shape)
        self.ax2.plot(x1, self.shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5)
        self.ax2.plot(x2, x2 - self.shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5)
        self.ax2.plot(self.shape / 2, 0, marker="+", markersize=16, markeredgecolor="red")

        # visualize bird eye view (bev)
        self.ax2.imshow(bev_img, origin="lower")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        # add legend
        # handles, labels = ax2.get_legend_handles_labels()
        # legend = ax2.legend(
        #     [handles[0], handles[1]],
        #     [labels[0], labels[1]],
        #     loc="lower right",
        #     fontsize="x-small",
        #     framealpha=0.2,
        # )
        # for text in legend.get_texts():
        #     plt.setp(text, color="w")

    def save_plot(self, path, name):
        self.fig.savefig(
            os.path.join(path, f"{name}.png"),
            dpi=self.fig.dpi,
            bbox_inches="tight",
            pad_inches=0.0,
        )

if __name__ == "__main__":

    plot = Plot3DBox(
        image_path="./data/demo/videos/2011_09_26/image_02/data",
        label_path="./outputs/2022-09-01/22-12-09/inference",
        calib_path="./data/calib_kitti_images.txt",
        pred_path="./outputs/2022-09-01/22-12-09/inference",
        save_path="./data/results",
        mode="training",
    )

    plot.visualization()