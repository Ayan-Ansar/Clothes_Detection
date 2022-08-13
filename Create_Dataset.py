
from pathlib import Path
from PIL import Image
import os
import json 
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_annotation_dir():
    return r'/content/data/'
    
def images_dir():
    return r'/content/dataset/'

def get_classes(path):
    classes = []
    for image in os.listdir(path):
        if not image.startswith('.'):
            jsonfile = open(path + image, 'r')
            data = jsonfile.read()
            obj = json.loads(data)
            bboxes = obj['arr_boxes']
            for i in range(len(bboxes)):
                classname = bboxes[i].get('class')      
                if classname not in classes:
                    classes.append(classname)

    classes.sort()
    return classes
 

annotations = get_annotation_dir()
images_folder = images_dir()
classes_name = get_classes(annotations)

def get_annotations_list(annotations):
    annotations_list = []
    for i in os.listdir(annotations):
        if (not i.startswith('.')):
            annotations_list.append(i)
    
    return annotations_list

def Create_dataset(images_dir,annotations,annotations_list, dataset_type):
    images_path= Path(f"FashionDetect/images/{dataset_type}")
    images_path.mkdir(parents=True, exist_ok= True)

    labels_path = Path(f"FashionDetect/labels/{dataset_type}")
    labels_path.mkdir(parents=True, exist_ok= True)

    for img_id, image in enumerate(tqdm(annotations_list)):

            jsonfile = open(annotations + image, 'r')
            data = jsonfile.read()
            obj = json.loads(data)
            img = obj['file_name']
            image_name = f"{img_id}.jpeg"
            image_open = Image.open(images_dir + img)
            image_open = image_open.convert("RGB")
            image_open.save(str(images_path / image_name), "JPEG")
            image_width, image_height = image_open.size

            label_name = f"{img_id}.txt"
            bboxes = obj['arr_boxes']
            with (labels_path / label_name).open(mode = "w") as label_file:
                for i in range(len(bboxes)):

                # Normalized(Xmin) = (Xmin+w/2)/Image_Width
                # Normalized(Ymin) = (Ymin+h/2)/Image_Height
                # Normalized(w) = w/Image_Width
                # Normalized(h) = h/Image_Height
                    w = bboxes[i].get('width')
                    n_w = bboxes[i].get('width') / image_width
                    h = bboxes[i].get('height') 
                    n_h = bboxes[i].get('height') / image_height
                    x = ( bboxes[i].get('x') + w / 2 ) / image_width
                    y = ( bboxes[i].get('y') + h / 2 ) /  image_height
                
                    class_index = classes_name.index(bboxes[i].get('class'))

                    label_file.write(
                        f"{class_index} {x} {y} {n_w} {n_h}\n"
                    )



