from __future__ import print_function
import pandas as pd
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


class Generate_data():
    """
    Generate data class helps to generate the data from csv files in form of images and create train, validation and test datasets
    """
    def __init__(self, datapath):
        self.data_path = datapath

    def str_to_image(self, str_img=' '):
        """
        Helper function to convert pixels to images
            params:-
                str_img = pixel values in string format
        """
        imgarray_str = str_img.split(' ')
        imgarray = np.asarray(imgarray_str, dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(imgarray)

    def split_data(self, test_filename = 'finaltest', val_filename= 'val'):
        """
        Helper function to split the validation and test data from general test file as it contains (Public test, Private test)
            params:-
                data_path = path to the folder that contains the test data file
        """
        csv_path = self.data_path +"/"+ 'icml_face_data.csv'
        test = pd.read_csv(csv_path)
        test = test.rename(columns=lambda x: x.strip())
        training_data = pd.DataFrame(test[test['Usage'] == 'Training'])
        validation_data = pd.DataFrame(test[test['Usage'] == 'PrivateTest'])
        test_data = pd.DataFrame(test[test['Usage'] == 'PublicTest'])
        training_data.to_csv(self.data_path+"/"+"train.csv")
        test_data.to_csv(self.data_path+"/"+test_filename+".csv")
        validation_data.to_csv(self.data_path+"/"+val_filename+".csv")
        print("Done splitting the file into training, validation & final test file")

    def save_images(self, datatype='train'):
        """
        Helper function to read strings from csv file and save as images in the folder
            params:-
                datatype = type of data to be saved. Default value is train
        """
        foldername = self.data_path + "/" + datatype
        csvfile_path = self.data_path + "/" + datatype + '.csv'
        if not os.path.exists(foldername):
            os.mkdir(foldername)

        data = pd.read_csv(csvfile_path)
        data = data.rename(columns=lambda x: x.strip())
        images = data['pixels']  # dataframe to series pandas
        numberofimages = images.shape[0]
        for index in tqdm(range(numberofimages)):
            img = self.str_to_image(images[index])
            img.save(os.path.join(foldername, '{}{}.jpg'.format(datatype, index)), 'JPEG')
        print('Done saving {} data'.format((foldername)))