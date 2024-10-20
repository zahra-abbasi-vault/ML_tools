import cv2
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model



# Function to preprocess an image
def preprocess_image(input_shape,image_path):
    image = cv2.imread(image_path)
    
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    image = image / 255
    return image


class GenrateBatches:
    def __init__(self , base_directory) -> None:
        self.base_directory =base_directory

    def get_image_paths(self):
            batch = []
            proprocces = []
            for root, dirs, files in os.walk(self.base_directory):
                for file in files:
                    if file != None:
                        if file.endswith((".jpg" , ".bmp")) :
                            img_path = os.path.join(root, file)
                            im = cv2.imread(img_path)
                            if im is not None:
                                batch.append(img_path)
                                proprocces.append(preprocess_image(img_path))

                    #if it reached the length of batch size it yeilds the images
                    if len(batch) == 32:
                        yield batch ,np.array(proprocces)
                        batch.clear()
                        proprocces = []
                        proprocces.clear()

                    #else if the images in the dataset ended and we didnt reach the goal
                    #TODO: add the remaining images to the batch


                    
                


model_path = r"path/to/the/classification/model"
root_of_datasets = r"path/to/root/of/datasets"
result_dir = r"path/to/result/dir"
outputs_of_model = ["negative" , "positive"]
input_shape = [100,100,3]

model = load_model(model_path) 
model.summary()

datasets = os.listdir(root_of_datasets)
for dataset_name in datasets:
    generator = GenrateBatches(os.path.join(root_of_datasets , dataset_name))

    for bch , prc in generator.get_image_paths():
        predictions = model.predict(prc)
        predicted_models  = predictions
        predicted_models_index = predicted_models.argmax(axis=1)

        for prediction, image_path , conf in zip(predicted_models_index, bch , predictions):
            predicted_output_class = outputs_of_model[prediction]

            folder_name = image_path.split("/")[-2]   
            im_name = image_path.split("/")[-1]  

            save_directory = os.path.join(result_dir,dataset_name , predicted_output_class)

            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            shutil.copy(
                src = image_path, 
                dst = os.path.join(save_directory ,folder_name + "_" + im_name ) #to be sure names are unique
                )




