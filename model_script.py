# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# ST-01
# Import needed dependencies 
import tensorflow as tf

#For data visualization -> (plt.imshow())
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

#for one hot encoding 
import sklearn.preprocessing as skp
import sklearn.model_selection as skm

#For OS and DIR convenience 
import os 

#For image manipulatio 
import cv2
import imutils


# %%
#ST-02
# Data pre processing, loading and categorizing 
#Instantiate directory to work with - we use an r string to ignore any slashes or backslashes
DATA_DIR = r"./dataset"

#We instantiate a list with the categories 
CATEGORY = ["with_mask", "without_mask"]

#Data resizing - decalare height and width when loading 
height = 224
width = 224

#We create two lists to hold the data we load 
img_data = []
img_labels = []

#We loop through the category and then inside the dir
#Append images and labels to the empty lists created 
def data_preparation(img_data, img_labels):
    
    for category in CATEGORY:

        #Use os path to join them 
        new_path = os.path.join(DATA_DIR, category)

        #Now loop through the dataset(with and without mask)
        for img in os.listdir(new_path):

            #decalare the image path 
            img_temp_path = os.path.join(new_path, img)

            #Load image
            img_temp = tf.keras.preprocessing.image.load_img(img_temp_path, target_size=(height, width))

            #convert the image to an array 
            img_temp = tf.keras.preprocessing.image.img_to_array(img_temp)

            #For objective 1 we are testing with mobileNet for speed
            #So we process input using that - it will adequate it to the way the model requires
            img_temp = tf.keras.applications.mobilenet_v3.preprocess_input(img_temp)

            #Append the processed image to the list with the corresponding label
            img_data.append(img_temp)
            img_labels.append(category)
    
    return img_data, img_labels

#Call the function 
img_data, img_labels = data_preparation(img_data, img_labels)


# %%
#The image data is in numerical form but label is in alphabet 
#Convert labels to numerical using one-hot encoding 
label_binarizer = skp.LabelBinarizer()
img_labels = label_binarizer.fit_transform(img_labels)

#Convert the vector to matrix
img_labels = tf.keras.utils.to_categorical(img_labels)

#Convert data to numpy array 
img_data = np.array(img_data, dtype="float32")

#Split the data into training and testing 
(trainX, testX, trainY, testY) = skm.train_test_split(img_data, img_labels, test_size=0.20, stratify=img_labels, random_state=42)

#Final Step is to augment the image - apply transformations and rotation
augmentation_def = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)


# %%
#Define a callback function to manage training based on own metrics
class myCallback(tf.keras.callbacks.Callback):

    #We initialize it with a value accuracy level
    #We can specify the needed accuracy in the code
    def __init__(self, accuracy_level):
        self.accuracy_level = accuracy_level

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy') > self.accuracy_level):
            print("Reached Sufficient accuracy. Concluding Training")
            self.model.stop_training = True

#Defining an instance of the class
accuracy_level = 0.92 
mycallback = myCallback(accuracy_level)


# %%
#ST-03 Model Architecture 
#Instead of defining the layers manually we use MobileNet- then we pass the output to final layer

#First we initialize learning rate, epoch_no and batch size 
#(Can use 1e-4 instead)
LR = 0.0001 

#Number of images per batch 
BS = 32 

#Number of iterations / epochs
ES = 20 

#We will define the basemodel 
#The include top is if we want a fully connected layer at the top
base_model = tf.keras.applications.MobileNetV3Small(
    weights='imagenet', include_top=False, input_shape=(height, width, 3))

#Top model will be in front of base model. output of base is conected to top
top_model = base_model.output

#Downsample to 7x7 
top_model = tf.keras.layers.AveragePooling2D(pool_size=(7,7))(top_model)

#Flatten the model
top_model = tf.keras.layers.Flatten(name="flatten")(top_model)

#We add a dense layer to extract features 
top_model = tf.keras.layers.Dense(128, activation="relu")(top_model)

#Drop out layer randoly makes zero to prevent overfitting 
top_model = tf.keras.layers.Dropout(0.5)(top_model)

#Add a final dense layer - outpul layers - 2 because we are checking for with and without mask
top_model = tf.keras.layers.Dense(2, activation="softmax")(top_model)

#The final model declaration - takes 2 inputs 
final_model = tf.keras.models.Model(inputs =base_model.input, outputs = top_model)

#Freeze the base model as we wont be training that, only using that for inference
for layer in base_model.layers:
    layer.trainable = False

#Define the optimizer
user_def_optimizer = tf.keras.optimizers.Adam(learning_rate=LR, decay= LR/ES)

#Compile the model 
#Binary cross entropy is another loss calucation function that is independet of other variables
final_model.compile(loss = "binary_crossentropy", optimizer = user_def_optimizer, metrics = ["accuracy"])

#ST-04 Training Model 
#Train top_model of the network - store it in a rand var

#Print the model summary 
final_model.summary()

TR = final_model.fit(
    augmentation_def.flow(trainX, trainY, batch_size =BS), 
    steps_per_epoch = len(trainX) // BS, 
    validation_data = (testX, testY),
    validation_steps =len(testX) // BS,
    epochs = ES)



# %%




