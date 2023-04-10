import os
import mediapipe as mp
import pickle

joint_names = ["Wrist", "Thumb CMC", "Thumb MCP", "Thumb IP", "Thumb Tip",
                            "Index MCP", "Index PIP", "Index DIP", "Index Tip",
                            "Middle MCP", "Middle PIP", "Middle DIP", "Middle Tip",
                            "Ring MCP", "Ring PIP", "Ring DIP", "Ring Tip",
                            "Little MCP", "Little PIP", "Little DIP", "Little Tip"]


node_list = [
            [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],  # Index finger
            [0, 9], [9, 10], [10, 11], [11, 12],  # Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16],  # Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20]  # Little finger
        ]


class HandGestureDataset:
    def __init__(self):
        model_path = 'commons/hand_landmarker.task'
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(base_options=base_options,
                                                        num_hands=1)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)


    def get_joint_points(self, image_path):
        image = mp.Image.create_from_file(image_path)

        detection_result = self.detector.detect(image)

        joint_points = []

        if len(detection_result.hand_world_landmarks) == 0:
            return None
        for hand_landmark in detection_result.hand_world_landmarks[0]:
            joint_points.append([hand_landmark.x, hand_landmark.y, hand_landmark.z])

        return joint_points

    def load_dataset_from_image(self, root_dir):
        data = []
        for label in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir) or label_dir.endswith('.DS_Store'):
                continue
            print(label_dir)
            for image_file in sorted(os.listdir(label_dir)):
                image_path = os.path.join(label_dir, image_file)
                if image_file.endswith('.jpeg'):
                    joint_points = self.get_joint_points(image_path)
                    if joint_points is not None:
                        data.append((joint_points, label))
        return data

    def create_dataset(self):
        dataset = HandGestureDataset()
        root_dir = "dataset/asl_dataset"
        asl_dataset = dataset.load_dataset_from_image(root_dir)

        with open('dataset/asl_dataset.pickle', 'wb') as f:
            pickle.dump(asl_dataset, f)

    def load_dataset(self):
        pickle_path = 'dataset/asl_dataset.pickle'
        if not os.path.isfile(pickle_path):
            self.create_dataset()

        with open(pickle_path, 'rb') as f:
            asl_dataset = pickle.load(f)
        return asl_dataset
