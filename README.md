# Font-detection

Using deep learning , creating a font detection model , which is capable of detecting fonts from images and designs .

# Dataset

We are using  AdobeVFR Dataset since it  is huge in size and contains lot of font categories. https://www.dropbox.com/sh/o320sowg790cxpe/AAAw2l7aeTH1fJqkVXoaZmTga/Raw%20Image/VFR_real_test/TimesLTStd-Bold?dl=0&subfolder_nav_tracking=1

# Platform

We are using Google - colab / Anaconda - Navigator for training the dataset and building the model.

# Libraries

  * scikit-learn
  * pandas
  * numpy
  * matplotlib
  * tensorflow
  * Opencv
  * keras

# Training Dataset
   
The dataset should contain pictures of the available fonts and  each font style should be stored into seperate folders under proper labeling.  

# Testing Dataset

Collect real world images and store it into a folder , which contain font styles in the images .   

## Processing of datasets 

Import the datasets from the folder using 

      data_path = " **** " 

We have to conver the images into arrays(numpy array) and store it into a variable and labels of the images should be stored into another array(numpy array). We convert the images into array because it is easier way to represent the working model.

      data=[]
      labels=[]

We can resige the image using the function .    
  
    image.resize()     
     
The dimensions can be  (105,105) , (126, 126) ,(256, 256) etc. The dimensions can be customised.

Fonts are not like objects , to have to huge spatial information to classify their features . To identify minute features I have to use certain preprocessing techniques like:

* Noise
* Blur
* Perpective Rotation
* Gradient

With the help of augmentation we can get many instances of the images as it extract insight from that data(image) and present deeper than before , which helps the model to understand better. We have to give the dimension of the image in each preprocessing techniques.  

We are going to use ImageDataGenerator because it apply any random transformations on each training image as it is passed to the model. This will not only make your model robust but will also save up on the overhead memory.

# Model Architecture

We are using CNN network . First let's conver the images into grayscale , then we take a filter/kernel(3×3 matrix) and apply it to the input image to get the convolved feature.
When an image is put in a ConvNet, each layer generates several activation functions that are passed on to the next layer. First layer extracts the basics features such as horizontal and vertical edges. Then these features is passed onto next layer which detects more complex features such as corners or combinational edges and this goes on.

We have to create the model using 
      
      model=Sequential()

Then its time to create the layers using 2D convolution layer , BatchNormalization , MaxPooling2D .
       
       ///These are CNN layers
      
      model.add(MaxPooling2D(pool_size=(2, 2)))
      
      model.add(BatchNormalization())
      
      model.add(Conv2D(128, kernel_size=(24, 24), activation='relu'))
      
  We have to determine the pool size and the activation function. 
  
 ### Activation function ->> Sigmoid , relu , softmax , linear and many more
 
 The activation function is a non-linear transformation that we do over the input before sending it to the next layer of neurons or finalizing it as output.
 
   #### Number of Epochs the model have to run we have to determine that  and give the parameters.
 
    epochs = 50
 
 ### After that fit the model and wait for evalutaion . Evaluate the loss and accuracy and again prepare the model for training . 
 
 ### The accuracy depends on the type of the dataset we are using and the complexity of the model.
 
 # Atlast save the model .


