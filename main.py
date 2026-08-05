# ==============================
# Gun Detection Object Detection Model
# Setup and Data Preparation
# ==============================


# Connect Google Drive (only needed when running in Google Colab)
from google.colab import drive

drive.mount("/gdrive")


# Install required packages (run once in Colab)
!apt-get install -qq protobuf-compiler python-pil python-lxml python-tk
!pip install -qq Cython contextlib2 pillow lxml matplotlib pycocotools
!pip install tensorflow-object-detection-api


# Imports
import os
import glob
import io
import xml.etree.ElementTree as ET

import pandas as pd
import tensorflow.compat.v1 as tf

from collections import namedtuple
from PIL import Image


print("TensorFlow version:", tf.__version__)


# ==============================
# Convert XML annotations to CSV
# ==============================

def xml_to_csv(folder_path):
    """
    Reads XML annotation files and converts them into a dataframe.

    Each XML file contains:
    - image name
    - image size
    - object class
    - bounding box coordinates
    """

    annotations = []
    class_names = []

    xml_files = glob.glob(os.path.join(folder_path, "*.xml"))

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        filename = root.find("filename").text

        width = int(root.find("size")[0].text)
        height = int(root.find("size")[1].text)

        objects = root.findall("object")

        for obj in objects:
            class_name = obj.find("name").text
            class_names.append(class_name)

            box = obj.find("bndbox")

            xmin = int(box[0].text)
            ymin = int(box[1].text)
            xmax = int(box[2].text)
            ymax = int(box[3].text)

            annotations.append(
                [
                    filename,
                    width,
                    height,
                    class_name,
                    xmin,
                    ymin,
                    xmax,
                    ymax
                ]
            )


    columns = [
        "filename",
        "width",
        "height",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax"
    ]

    dataframe = pd.DataFrame(
        annotations,
        columns=columns
    )

    # Remove duplicate class names.
    class_names = sorted(list(set(class_names)))

    return dataframe, class_names



# Convert training and testing XML files into CSV files

label_folders = [
    "train_labels",
    "test_labels"
]


all_classes = []

for folder in label_folders:

    dataframe, classes = xml_to_csv(folder)

    all_classes.extend(classes)

    csv_name = folder + ".csv"

    dataframe.to_csv(
        csv_name,
        index=False
    )

    print("Created:", csv_name)



# ==============================
# Create TensorFlow label map
# ==============================


# Remove duplicate classes and sort them.
all_classes = sorted(list(set(all_classes)))


label_map_path = "label_map.pbtxt"


with open(label_map_path, "w") as file:

    for index, class_name in enumerate(all_classes):

        file.write(
            "item {\n"
            f"    id: {index + 1}\n"
            f"    name: '{class_name}'\n"
            "}\n\n"
        )


print("Created label map:", label_map_path)

# ==============================
# TensorFlow Setup and TFRecord Generation
# ==============================


# Download TensorFlow Object Detection API models
!git clone --q https://github.com/tensorflow/models.git


# Compile TensorFlow protobuf files
!protoc models/research/object_detection/protos/*.proto --python_out=models/research


# Add TensorFlow models folder to Python path
import sys

sys.path.append("models/research")
sys.path.append("models/research/slim")


# Test TensorFlow Object Detection API installation
!python models/research/object_detection/builders/model_builder_tf1_test.py



# ==============================
# Create TFRecords
# ==============================

from object_detection.utils import dataset_util



# Folder containing images and annotation files
data_folder = "/gdrive/My Drive/object_detection/data"

image_folder = os.path.join(
    data_folder,
    "images"
)



def class_text_to_int(class_name):
    """
    Converts class names into integer labels.

    TensorFlow models require numerical labels instead of text.
    """

    if class_name == "pistol":
        return 1

    return None



def split_data(dataframe, group_column):
    """
    Groups dataframe rows by image filename.

    Multiple objects can exist in one image,
    so all objects belonging to one image are grouped together.
    """

    grouped_data = dataframe.groupby(group_column)

    grouped_images = []

    ImageData = namedtuple(
        "ImageData",
        [
            "filename",
            "objects"
        ]
    )

    for filename, objects in grouped_data:

        grouped_images.append(
            ImageData(
                filename,
                objects
            )
        )

    return grouped_images



def create_tf_example(group, image_folder):

    image_path = os.path.join(
        image_folder,
        group.filename
    )


    # Read image bytes because TFRecords store raw image data.
    with tf.io.gfile.GFile(
        image_path,
        "rb"
    ) as image_file:

        encoded_image = image_file.read()


    image = Image.open(
        io.BytesIO(encoded_image)
    )

    width, height = image.size


    filename = group.filename.encode("utf8")


    image_format = b"jpg"


    xmin = []
    xmax = []
    ymin = []
    ymax = []

    class_text = []
    class_labels = []


    for _, row in group.objects.iterrows():

        # Bounding boxes must be converted from pixels
        # into values between 0 and 1.
        xmin.append(
            row["xmin"] / width
        )

        xmax.append(
            row["xmax"] / width
        )

        ymin.append(
            row["ymin"] / height
        )

        ymax.append(
            row["ymax"] / height
        )


        class_text.append(
            row["class"].encode("utf8")
        )


        class_labels.append(
            class_text_to_int(row["class"])
        )



    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={

                "image/height":
                    dataset_util.int64_feature(height),

                "image/width":
                    dataset_util.int64_feature(width),

                "image/filename":
                    dataset_util.bytes_feature(filename),

                "image/source_id":
                    dataset_util.bytes_feature(filename),

                "image/encoded":
                    dataset_util.bytes_feature(encoded_image),

                "image/format":
                    dataset_util.bytes_feature(image_format),


                "image/object/bbox/xmin":
                    dataset_util.float_list_feature(xmin),

                "image/object/bbox/xmax":
                    dataset_util.float_list_feature(xmax),

                "image/object/bbox/ymin":
                    dataset_util.float_list_feature(ymin),

                "image/object/bbox/ymax":
                    dataset_util.float_list_feature(ymax),


                "image/object/class/text":
                    dataset_util.bytes_list_feature(class_text),

                "image/object/class/label":
                    dataset_util.int64_list_feature(class_labels)
            }
        )
    )


    return tf_example




# Create TFRecords for training and testing data

csv_files = [
    "train_labels",
    "test_labels"
]


for csv_file in csv_files:


    csv_path = os.path.join(
        data_folder,
        csv_file + ".csv"
    )


    output_path = os.path.join(
        data_folder,
        csv_file + ".record"
    )


    dataframe = pd.read_csv(
        csv_path
    )


    grouped_images = split_data(
        dataframe,
        "filename"
    )


    writer = tf.io.TFRecordWriter(
        output_path
    )


    for image_group in grouped_images:

        example = create_tf_example(
            image_group,
            image_folder
        )

        writer.write(
            example.SerializeToString()
        )


    writer.close()


    print(
        "Created TFRecord:",
        output_path
    )

# ==============================
# Download Pretrained Model
# ==============================


import os
import urllib.request
import tarfile
import shutil



# Select the TensorFlow object detection model.
# SSD MobileNet is lightweight and works well for smaller datasets.
model_name = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"


model_download_url = (
    "http://download.tensorflow.org/models/"
    "object_detection/tf2/20200711/"
    f"{model_name}.tar.gz"
)



# Download location
model_file = model_name + ".tar.gz"



# Folder where the extracted model will be stored
pretrained_folder = os.path.join(
    data_folder,
    "pretrained_model"
)



# Download the model if it does not already exist

if not os.path.exists(model_file):

    print("Downloading pretrained model...")

    urllib.request.urlretrieve(
        model_download_url,
        model_file
    )

    print("Download complete.")

else:

    print("Model already downloaded.")



# Extract the model files

if os.path.exists(pretrained_folder):

    shutil.rmtree(
        pretrained_folder
    )


with tarfile.open(
    model_file
) as tar:

    tar.extractall()



# Rename extracted folder
shutil.move(
    model_name,
    pretrained_folder
)



# Delete compressed file after extraction

os.remove(
    model_file
)


print(
    "Pretrained model saved at:",
    pretrained_folder
)

# ==============================
# Training Pipeline Setup
# ==============================


from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2

from google.protobuf import text_format



# Location of the base pipeline configuration file

pipeline_file = os.path.join(
    pretrained_folder,
    "pipeline.config"
)



# Location where we will save our customized config

training_folder = os.path.join(
    data_folder,
    "training"
)


os.makedirs(
    training_folder,
    exist_ok=True
)


updated_pipeline_file = os.path.join(
    training_folder,
    "pipeline.config"
)



# Load pipeline configuration

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()


with tf.io.gfile.GFile(
    pipeline_file,
    "r"
) as file:

    text_format.Merge(
        file.read(),
        pipeline_config
    )



# ==============================
# Modify important settings
# ==============================


# Number of object classes.
# Our model only detects guns.
pipeline_config.model.ssd.num_classes = 1



# Training batch size.
# Lower this if GPU memory is limited.
pipeline_config.train_config.batch_size = 8



# Number of training steps.
pipeline_config.train_config.num_steps = 50000



# Location of pretrained weights.

pipeline_config.train_config.fine_tune_checkpoint = os.path.join(
    pretrained_folder,
    "checkpoint",
    "ckpt-0"
)



pipeline_config.train_config.fine_tune_checkpoint_type = (
    "detection"
)



# Training dataset TFRecord

pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(
        data_folder,
        "train_labels.record"
    )
]



# Label map location

pipeline_config.train_input_reader.label_map_path = (
    os.path.join(
        data_folder,
        "label_map.pbtxt"
    )
)



# Testing dataset TFRecord

pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(
        data_folder,
        "test_labels.record"
    )
]


pipeline_config.eval_input_reader[0].label_map_path = (
    os.path.join(
        data_folder,
        "label_map.pbtxt"
    )
)



# Save updated configuration

with tf.io.gfile.GFile(
    updated_pipeline_file,
    "w"
) as file:

    file.write(
        text_format.MessageToString(
            pipeline_config
        )
    )



print(
    "Training pipeline created:",
    updated_pipeline_file
)

# ==============================
# Train the Object Detection Model
# ==============================


import os



# Location of TensorFlow training script

train_script = os.path.join(
    "models",
    "research",
    "object_detection",
    "model_main_tf2.py"
)



# Folder where checkpoints will be saved

checkpoint_folder = os.path.join(
    data_folder,
    "training"
)



# Start training

!python {train_script} \
    --pipeline_config_path={updated_pipeline_file} \
    --model_dir={checkpoint_folder} \
    --alsologtostderr

# ==============================
# TensorBoard
# ==============================


%load_ext tensorboard


%tensorboard --logdir {checkpoint_folder}

# ==============================
# Export Trained Model
# ==============================


export_script = os.path.join(
    "models",
    "research",
    "object_detection",
    "exporter_main_v2.py"
)



export_folder = os.path.join(
    data_folder,
    "exported_model"
)



!python {export_script} \
    --input_type=image_tensor \
    --pipeline_config_path={updated_pipeline_file} \
    --trained_checkpoint_dir={checkpoint_folder} \
    --output_directory={export_folder}

# ==============================
# Load Saved Model
# ==============================


import tensorflow as tf



saved_model_path = os.path.join(
    export_folder,
    "saved_model"
)



model = tf.saved_model.load(
    saved_model_path
)



print("Model loaded successfully")

# ==============================
# Run Object Detection
# ==============================


import cv2
import numpy as np
import matplotlib.pyplot as plt



def load_image(image_path):
    """
    Loads an image and converts it into
    the format TensorFlow expects.
    """

    image = cv2.imread(
        image_path
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    return image



def detect_objects(image):

    # TensorFlow models expect batches,
    # so add an extra dimension.
    input_tensor = tf.convert_to_tensor(
        image
    )


    input_tensor = input_tensor[tf.newaxis, ...]



    detections = model(
        input_tensor
    )


    return detections



def draw_detections(image, detections, threshold=0.5):

    boxes = detections["detection_boxes"][0].numpy()

    scores = detections["detection_scores"][0].numpy()



    height, width, _ = image.shape



    for i in range(len(scores)):

        if scores[i] < threshold:
            continue


        ymin, xmin, ymax, xmax = boxes[i]


        start_point = (
            int(xmin * width),
            int(ymin * height)
        )


        end_point = (
            int(xmax * width),
            int(ymax * height)
        )


        cv2.rectangle(
            image,
            start_point,
            end_point,
            (255, 0, 0),
            2
        )


        cv2.putText(
            image,
            f"Gun {scores[i]:.2f}",
            start_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )


    return image

# ==============================
# Test Detection
# ==============================


test_image_path = os.path.join(
    image_folder,
    "example.jpg"
)



image = load_image(
    test_image_path
)


detections = detect_objects(
    image
)


result = draw_detections(
    image,
    detections
)



plt.figure(
    figsize=(10,8)
)

plt.imshow(
    result
)

plt.axis("off")

plt.show()
