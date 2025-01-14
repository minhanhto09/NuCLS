import os
import yaml
from datasets import load_dataset, concatenate_datasets
from PIL import Image
import numpy as np
from ultralytics import YOLO


class YOLOPipeline:
    def __init__(self, base_path='./YOLOV8', model_path='yolov8n.pt', epochs=10):
        """
        Initialize the YOLO Pipeline.
        
        Args:
            base_path (str): Path to store processed data and model files.
            model_path (str): Path to the YOLO model weights.
            epochs (int): Number of training epochs.
        """
        self.base_path = base_path
        self.model_path = model_path
        self.epochs = epochs
        self.dataset = None
        self.model = None

    def resize_image(self, image, target_size=(640, 640)):
        """
        Resize an image while preserving aspect ratio. Adds padding to maintain the aspect ratio.
        """
        scale = min(target_size[0] / image.width, target_size[1] / image.height)
        new_size = (int(image.width * scale), int(image.height * scale))
        resized_image = image.resize(new_size, Image.LANCZOS)
        padding = (
            (target_size[0] - new_size[0]) // 2,
            (target_size[1] - new_size[1]) // 2,
        )
        padded_image = Image.new("RGB", target_size)
        padded_image.paste(resized_image, padding)
        return padded_image, scale, padding

    def adjust_annotation_coordinates(self, coords, scale, padding):
        """
        Adjust annotation coordinates based on scaling and padding.
        """
        adjusted_coords = {}
        for key in ['xmin', 'ymin', 'xmax', 'ymax']:
            scaled_coords = np.array(coords[key]) * scale
            if key in ['xmin', 'xmax']:
                adjusted_coords[key] = (scaled_coords + padding[0]).tolist()
            elif key in ['ymin', 'ymax']:
                adjusted_coords[key] = (scaled_coords + padding[1]).tolist()
        return adjusted_coords

    def create_dataset_files(self, split_name, target_size=(640, 640)):
        """
        Create YOLO-compatible dataset files for a specific split (train/test).
        """
        image_dir = os.path.join(self.base_path, 'images', split_name)
        label_dir = os.path.join(self.base_path, 'labels', split_name)

        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for sample in self.dataset[split_name]:
            image = sample['rgb_image']
            image_path = os.path.join(image_dir, f"{sample['file_name']}.png")
            label_path = os.path.join(label_dir, f"{sample['file_name']}.txt")

            resized_image, scale, padding = self.resize_image(image, target_size)
            resized_image.save(image_path)

            adjusted_coords = self.adjust_annotation_coordinates(sample['annotation_coordinates'], scale, padding)

            with open(label_path, 'w') as f:
                for i in range(len(adjusted_coords['xmin'])):
                    class_id = {
                        'AMBIGUOUS': 0,
                        'lymphocyte': 1,
                        'macrophage': 2,
                        'nonTILnonMQ_stromal': 3,
                        'plasma_cell': 4,
                        'tumor_mitotic': 5,
                        'tumor_nonMitotic': 6
                    }.get(sample['annotation_coordinates']['main_classification'][i], -1)
                    x_center = ((adjusted_coords['xmin'][i] + adjusted_coords['xmax'][i]) / 2) / target_size[0]
                    y_center = ((adjusted_coords['ymin'][i] + adjusted_coords['ymax'][i]) / 2) / target_size[1]
                    width = (adjusted_coords['xmax'][i] - adjusted_coords['xmin'][i]) / target_size[0]
                    height = (adjusted_coords['ymax'][i] - adjusted_coords['ymin'][i]) / target_size[1]
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def create_yaml_file(self):
        """
        Create a YAML file for YOLO training configuration.
        """
        data_dict = {
            'path': self.base_path,
            'train': os.path.join('images', 'train'),
            'val': os.path.join('images', 'test'),
            'nc': 7,
            'names': [
                'AMBIGUOUS', 'lymphocyte', 'macrophage',
                'nonTILnonMQ_stromal', 'plasma_cell',
                'tumor_mitotic', 'tumor_nonMitotic'
            ]
        }

        with open(os.path.join(self.base_path, 'data.yaml'), 'w') as file:
            yaml.dump(data_dict, file, sort_keys=False)

    def prepare_dataset(self):
        """
        Load and prepare the dataset by combining folds and generating YOLO-compatible files.
        """
        dataset = load_dataset("minhanhto09/NuCLS_dataset", name="default")
        train_splits = [dataset[f'train_fold_{i}'] for i in range(1, 6)]
        test_splits = [dataset[f'test_fold_{i}'] for i in range(1, 6)]

        combined_train = concatenate_datasets(train_splits)
        combined_test = concatenate_datasets(test_splits)

        self.dataset = {'train': combined_train, 'test': combined_test}

        print(f"Combined train size: {len(self.dataset['train'])}")
        print(f"Combined test size: {len(self.dataset['test'])}")

        os.makedirs(self.base_path, exist_ok=True)
        self.create_dataset_files('train')
        self.create_dataset_files('test')
        self.create_yaml_file()

    def train_model(self):
        """
        Train the YOLO model.
        """
        self.model = YOLO(self.model_path)
        self.model.train(data=os.path.join(self.base_path, 'data.yaml'), epochs=self.epochs)

    def test_model(self, test_image_path):
        """
        Test the trained YOLO model on a sample image.
        """
        results = self.model(test_image_path)
        for result in results:
            result.show()
            result.save(filename='result.jpg')


if __name__ == "__main__":
    pipeline = YOLOPipeline(base_path='./YOLOV8', model_path='yolov8n.pt', epochs=10)
    pipeline.prepare_dataset()
    pipeline.train_model()
    pipeline.test_model('./YOLOV8/images/test/sample_image.png')
