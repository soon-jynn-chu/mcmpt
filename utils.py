import os
import time
import json
from pathlib import Path

import cv2
import lap
import torch
import torchreid
import numpy as np
import pandas as pd
import onnxruntime as ort
import torchvision.transforms as T
from PIL import Image
from torch import nn
from copy import deepcopy
from ultralytics.utils.plotting import Annotator
from ultralytics import YOLO

from deep_ocsort import DeepOCSort


class ReIDDetectMultiBackend(nn.Module):
    # https://github.com/mikel-brostrom/yolo_tracking.git
    def __init__(self, weights, providers):
        super().__init__()

        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Build transform functions
        input_shape = self.session.get_inputs()[0].shape
        image_size = (input_shape[2], input_shape[3])
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        self.preprocess = T.Compose(transforms)
        self.to_pil = T.ToPILImage()

        self.warmup(input_shape)

    def _preprocess(self, im_batch):
        images = []
        for element in im_batch:
            image = self.to_pil(element)
            image = self.preprocess(image)
            images.append(image)
        return torch.stack(images, dim=0).cpu().numpy()

    def forward(self, im_batch):
        im_batch = self._preprocess(im_batch)

        features = []
        for im in im_batch:
            feature = self.session.run([self.output_name], {self.input_name: [im]})[
                0
            ].reshape(-1)
            features.append(feature)

        return torch.Tensor(np.array(features))

    def warmup(self, imgsz):
        t = time.time()
        imgsz = (imgsz[2], imgsz[3], imgsz[1])
        im = [np.empty(imgsz, dtype=np.uint8)]
        for _ in range(2):
            self.forward(im)


class Utilities:
    def __init__(self, configs):
        self.sub_folders = configs["sub_folders"]
        self.header = configs["header"]

        self.emotion_color = configs["emotion_color"]

        (
            self.save_path,
            self.mapping_path,
            self.results_path,
            self.video_path,
        ) = self.generate_directory_name(Path(configs["output_location"]))

        self.anonymize = configs["anonymize"]

        self.global_embs = {}
        self.max_id = 0

    def global_id_assignment(self, local_embs, mapping, threshold=0.5):
        # Filter unmatched global & local IDs
        matched_local_ind = list(mapping.keys())
        matched_global_ind = list(mapping.values())

        unmatched_local_embs = {
            key: value
            for key, value in local_embs.items()
            if key not in matched_local_ind
        }
        unmatched_global_embs = {
            key: value
            for key, value in self.global_embs.items()
            if key not in matched_global_ind
        }

        # Nothing to match
        if unmatched_local_embs == {}:
            return mapping

        # Populate empty global dictionary
        if unmatched_global_embs == {}:
            for key in unmatched_local_embs.keys():
                self.max_id += 1
                self.global_embs[self.max_id] = unmatched_local_embs[key]
                mapping[key] = self.max_id
            return mapping

        # Compute cost matrix
        x = torch.Tensor(np.array(list(unmatched_global_embs.values())))
        y = torch.Tensor(np.array(list(unmatched_local_embs.values())))

        distances = torchreid.metrics.compute_distance_matrix(
            x, y, metric="cosine"
        ).numpy()
        matched_indices = self.linear_assignment(distances)

        # Map matched IDs
        global_inds = list(unmatched_global_embs.keys())
        local_inds = list(unmatched_local_embs.keys())

        matched_id = []

        for m in matched_indices:
            global_ind = global_inds[m[0]]
            local_ind = local_inds[m[1]]
            if distances[m[0], m[1]] < threshold:
                matched_id.append(local_ind)
                mapping[local_ind] = global_ind

        # Create new field for unmatched IDs
        for id in local_inds:
            if id not in matched_id:
                self.max_id += 1
                self.global_embs[self.max_id] = unmatched_local_embs[id]
                mapping[id] = self.max_id

        return mapping

    def linear_assignment(self, cost_matrix):
        # https://github.com/mikel-brostrom/yolo_tracking.git
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])

    def generate_directory_name(self, name, x=0):
        """
        Create a directory for the input name.
        If the directory exists, append a number in the increasing order.
        """
        name.parent.mkdir(parents=True, exist_ok=True)

        while True:
            dir_name = Path((str(name) + (str(x) if x != 0 else "")).strip())
            if not dir_name.exists():
                dir_name.mkdir()
                return [str(dir_name)] + self.generate_sub_directories(str(dir_name))
            else:
                x += 1

    def generate_sub_directories(self, main_folder):
        sub_names = []
        for sub in self.sub_folders:
            sub_name = os.path.join(main_folder, sub)
            if not os.path.exists(sub_name):
                os.mkdir(sub_name)
            sub_names.append(sub_name)
        return sub_names


class Models:
    def __init__(self, configs):
        print("Initializing models ...")
        t = time.time()

        # Specify hardware
        self.device = 0 if torch.cuda.is_available() else "cpu"

        # Initialize models
        ## Detection models
        self.person_det = YOLO(configs["person_det"])
        self.person_seg_configs = configs["person_seg_configs"]
        self.face_det = YOLO(configs["face_det"])
        self.face_det_configs = configs["face_det_configs"]

        ## Classification models
        self.posture_model = YOLO(configs["posture_model"])

        self.emotion_model = ort.InferenceSession(
            configs["emotion_model"], providers=configs["onnx_providers"]
        )
        self.emotion_input_name = self.emotion_model.get_inputs()[0].name
        self.emotion_output_name = self.emotion_model.get_outputs()[0].name

        self.gender_model = ort.InferenceSession(
            configs["gender_model"], providers=configs["onnx_providers"]
        )
        self.gender_input_name = self.gender_model.get_inputs()[0].name
        self.gender_output_name = self.gender_model.get_outputs()[0].name

        ## Other models
        self.reid_model = ReIDDetectMultiBackend(
            configs["reid_model"], providers=configs["onnx_providers"]
        )

        # Define labels
        self.gender_labels = configs["gender_labels"]
        self.posture_labels = configs["posture_labels"]
        self.emotion_labels = configs["emotion_labels"]

        print(f"Completed in {time.time() - t:.2f}")

    def _alignment_procedure(self, img, left_eye, right_eye):
        """
        https://github.com/serengil/deepface.git
        this function aligns given face in img based on left and right eye coordinates
        """
        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        # find rotation direction
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # find length of triangle edges
        a = self._findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
        b = self._findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
        c = self._findEuclideanDistance(np.array(right_eye), np.array(left_eye))

        # apply cosine rule
        if (
            b != 0 and c != 0
        ):  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / np.pi  # radian to degree

            # rotate base image
            if direction == -1:
                angle = 90 - angle

            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))

        return img

    def _findEuclideanDistance(self, source_representation, test_representation):
        # https://github.com/serengil/deepface.git
        if isinstance(source_representation, list):
            source_representation = np.array(source_representation)

        if isinstance(test_representation, list):
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def _preprocess(self, img):
        """
        Preprocessing for emotion and gender classification
        Converts image to (N, 64, 64, 1) and normalize
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img.astype(np.float32) / 255
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)
        return img

    def _is_point_inside_quadrilateral(self, point, quadrilateral_pts):
        """Check if the person shoes are inside the quadrilateral"""
        return cv2.pointPolygonTest(quadrilateral_pts, point, False) >= 0

    def detect_person(self, image):
        results = self.person_det(
            image,
            device=self.device,
            **self.person_seg_configs,
        )
        h, w = results[0].orig_shape
        dets = results[0].boxes.data.cpu().numpy()
        masks = (
            results[0].masks.data.cpu().numpy().astype(bool)
            if len(dets) > 0
            else np.empty((0, 5, h, w))
        )
        return dets, masks

    def classify_emotion_gender_posture(self, image):
        """
        Args: image (cropped person)
        """
        posture = self.posture_model(image, verbose=None, device=self.device)
        posture = self.posture_labels[np.argmax(posture[0].probs.data.cpu().numpy())]

        faces = self.face_det(image, device=self.device, **self.face_det_configs)
        dets = faces[0].boxes.data.cpu().numpy().astype(int)

        if len(dets) == 0:
            return "empty", "empty", posture

        keypoints = faces[0].keypoints.data.cpu().numpy().astype(int)

        # Only get the face with the highest confidence
        x1, y1, x2, y2, _, _ = dets[0]
        left_eye, right_eye, _, _, _ = keypoints[0]

        # Preprocessing
        face = self._alignment_procedure(
            image[y1:y2, x1:x2], left_eye[:2], right_eye[:2]
        )
        face = self._preprocess(face)

        # Emotion & gender inference
        emotion = self.emotion_model.run(
            [self.emotion_output_name], {self.emotion_input_name: face}
        )[0]
        emotion = self.emotion_labels[np.argmax(emotion)]

        gender = self.gender_model.run(
            [self.gender_output_name], {self.gender_input_name: face}
        )[0]
        gender = self.gender_labels[np.argmax(gender)]

        return emotion, gender, posture


class SingleCamera:
    def __init__(self, input, models, utilities, configs):
        self.camera_id = Path(input).stem

        self.models = models
        self.utilities = utilities

        # Initialize tracker
        self.tracker = DeepOCSort(
            models=self.models,
            **configs["tracker_configs"],
        )

        self.cap = cv2.VideoCapture(input)

        # Get video information
        self.length, fps, width, height = [
            int(self.cap.get(prop))
            for prop in [
                cv2.CAP_PROP_FRAME_COUNT,
                cv2.CAP_PROP_FPS,
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
            ]
        ]
        print(
            f"{input}: frames={self.length}, width={width}, height={height}, fps={fps}"
        )

        self.out = cv2.VideoWriter(
            self.utilities.video_path + f"/{self.camera_id}.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        self.frame_count = 0
        self.mapping = {}  # {local_id: global_id}

    def update_single_frame(self):
        self.frame_count += 1
        ret, frame = self.cap.read()
        draw_frame = deepcopy(frame)

        if not ret:
            self.close()
            return True

        dets, masks = self.models.detect_person(frame)
        tracker_outputs, behaviors, embs = self.tracker.update(dets, frame, masks)

        # Anonymize people
        if self.utilities.anonymize:
            for i in masks:
                draw_frame[masks[i]] = 0

        annotator = Annotator(draw_frame)
        annotator.text([5, 25], f"Count: {len(tracker_outputs)}", txt_color=(0, 255, 0))

        if len(tracker_outputs) > 0:
            data = []
            local_embeds = {}

            for j, [x1, y1, x2, y2, local_id, conf, _] in enumerate(tracker_outputs):
                x1, y1, x2, y2, local_id = np.array(
                    [x1, y1, x2, y2, local_id], dtype=int
                )
                emotion, gender, posture = behaviors[j]

                local_embeds[local_id] = embs[j]

                data.append(
                    [self.camera_id, self.frame_count]
                    + [local_id, x1, y1, x2, y2, conf]
                    + [emotion, gender, posture]
                )

            # Update local to global ID mapping
            self.mapping = self.utilities.global_id_assignment(
                local_embeds, self.mapping
            )
            df = pd.DataFrame(data, columns=self.utilities.header[:-1])
            df["global_id"] = df["local_id"].astype(int).map(self.mapping)

            # Draw box for each person in image
            for _, row in df.iterrows():
                annotator.box_label(
                    [row["x1"], row["y1"], row["x2"], row["y2"]],
                    f"{row['global_id']} {row['posture']} {row['conf']:.2f}",
                    color=self.utilities.emotion_color[row["emotion"]],
                )

            # Save results
            output_path = f"{self.utilities.results_path}/{self.camera_id}.txt"
            df.to_csv(
                output_path,
                header=not os.path.exists(output_path),
                index=None,
                mode="a",
            )

        self.out.write(draw_frame)

        return False

    def close(self):
        self.cap.release()
        self.out.release()

        mapping = {int(key): int(value) for key, value in self.mapping.items()}

        with open(
            f"{self.utilities.mapping_path}/maps_{self.camera_id}.json", "w"
        ) as f:
            json.dump(mapping, f, indent=4, sort_keys=True)

        print(f"{self.camera_id} closed")
