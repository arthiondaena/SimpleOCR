from dataset import dataset
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers.legacy import SGD
from keras.callbacks import ModelCheckpoint
from model import ResNet
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data, labels = dataset()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
# class weights to address class imbalance
le = LabelBinarizer()
labels = le.fit_transform(labels)

counts = labels.sum(axis=0)

classTotals = labels.sum(axis=0)
classWeight = {}

for  i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

aug = ImageDataGenerator(
rotation_range = 10,
zoom_range = 0.05,
width_shift_range = 0.1,
height_shift_range = 0.1,
shear_range = 0.15,
horizontal_flip = False,
fill_mode = "nearest")

EPOCHS = 50
INIT_LR = 1e-1
BS = 128 #Batch Size
SAVE_PERIOD = 10
STEPS_PER_EPOCH = len(X_train) // BS
checkpoint_path = "weights.{epoch:02d}-{val_loss:.2f}.keras"

opt = SGD(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)

model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3),
                    (64, 64, 128, 256), reg=0.0005)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq = int(SAVE_PERIOD*STEPS_PER_EPOCH))

H = model.fit(
    aug.flow(X_train, y_train, batch_size=BS), 
    validation_data = (X_test, y_test),
    steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS,
    class_weight=classWeight,
    verbose=1
)

model.save('OCR_Resnet.keras')