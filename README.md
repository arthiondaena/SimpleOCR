# OCR (OPTICAL CHARACTER RECOGNITION)

This is a simple OCR project which can recognize the character present in the image.
For now, it is only capable of recognizing digits and capital letter.

## Dataset

For digits, I have used MNIST dataset, and for alphabets I have used kaggle [A-Z Handwritten Alphabets](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)


## Model
For the model architecture I have used Resnet architecture which you can read about [here](https://www.geeksforgeeks.org/residual-networks-resnet-deep-learning/).

This is the image of the architecture I have used.

![Model](https://github.com/arthiondaena/SimpleOCR/blob/master/model.PNG?raw=true)


## Usage
First install the required modules from requirements.txt

### Testing the model
If you would like to test the model, you can run GUI_predict.py and try.
Here is the sample output you could expect.

![OUTPUT](https://i.imgur.com/o79sIGH.png)

### Training the model
If you would like to train the model, you need to download and extract the 'A-Z Handwritten Alphabets' to data/ and edit the ocr_train.py as per your needs. I have set it to 50 epochs, which would take around 
300sec/epoch in google colab
