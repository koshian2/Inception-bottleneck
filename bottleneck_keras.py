from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, Input, Concatenate
from keras.models import Model
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, Callback
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import time, pickle, os, glob

def create_single_block(input_tensor, output_channels, conv_kernel):
    x = Conv2D(output_channels, conv_kernel, padding="same", kernel_regularizer=l2(1e-3))(input_tensor)
    x = BatchNormalization()(x)
    return Activation("relu")(x)

def create_block_with_bottleneck(input_tensor, mode, output_channels, alpha, conv_kernel):
    if mode == 1:
        return create_single_block(input_tensor, output_channels, conv_kernel)
    elif mode == 2:
        x = create_single_block(input_tensor, output_channels//alpha, 1)
        x = create_single_block(x, output_channels, conv_kernel)
        return x
    elif mode == 3:
        x = create_single_block(input_tensor, output_channels//alpha, 1)
        x = create_single_block(x, output_channels//alpha, conv_kernel)
        x = create_single_block(x, output_channels, 1)
        return x

def create_inception_block(input_tensor, mode, output_channels, alpha=2):
    # mode 1 : no-bottleneck
    # mode 2 : bottleneck -> conv
    # mode 3 : bottleneck -> conv -> bottleneck
    # branches = 50%:3x3, 25%:1x1, 12.5%:5x5, 12.5%:3x3pool
    # 
    # alpha = bottleneck_ratio : conv_channel / alpha = bottleneck_channels
    assert output_channels % (8*alpha) == 0
    assert mode >= 1 and mode <= 3 and type(mode) is int
    # 1x1 conv
    conv1 = create_single_block(input_tensor, output_channels//4, 1)
    # 3x3, 5x5 conv
    conv3 = create_block_with_bottleneck(input_tensor, mode, output_channels//2, alpha, 3)
    conv5 = create_block_with_bottleneck(input_tensor, mode, output_channels//8, alpha, 5)
    # 3x3 pool
    pool = MaxPool2D(3, strides=1, padding="same")(input_tensor)
    pool = create_single_block(pool, output_channels//8, 1)
    # Concat
    return Concatenate()([conv1, conv3, conv5, pool])

def create_model(mode, alpha=2):
    input = Input(shape=(32,32,3))
    x = create_inception_block(input, mode, 96, alpha)
    # pool 32 -> 16
    x = AveragePooling2D(2)(x)
    x = create_inception_block(x, mode, 256, alpha)
    # pool 16 -> 8
    x = AveragePooling2D(2)(x)
    x = create_inception_block(x, mode, 384, alpha)
    x = create_inception_block(x, mode, 384, alpha)
    x = create_inception_block(x, mode, 256, alpha)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax", kernel_regularizer=l2(1e-3))(x)
    # model
    model = Model(input, x)
    model.compile(Adam(1e-3), loss="categorical_crossentropy", metrics=["acc"])
    return model


class TimerCallBack(Callback):
    def __init__(self):
        super().__init__()
        self.time_log = []

    def on_train_begin(self, logs):
        self.time_log = []
        self.lasttime = time.time()

    def on_epoch_end(self, epoch, logs):
        current = time.time()
        self.time_log.append(current-self.lasttime)
        self.lasttime = time.time()

def lr_milestones(epochs):
    if epochs <= nb_epochs * 0.5 : return 1e-3
    elif epochs <= nb_epochs * 0.85: return 1e-4
    else: return 1e-5

def train(trial):
    # load-cifar
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    # generator
    train_gen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2)
    val_gen = ImageDataGenerator(rescale=1.0/255)
    train_gen.fit(X_train)
    # create-model
    model = create_model(mode, alpha=2)
    # callbacks
    lr_schduler = LearningRateScheduler(lr_milestones)
    timer_log = TimerCallBack()
    # train
    history = model.fit_generator(train_gen.flow(X_train, y_train, batch_size=128), steps_per_epoch=X_train.shape[0]/128,
                                  epochs=nb_epochs, validation_data=val_gen.flow(X_test, y_test),
                                  callbacks=[lr_schduler, timer_log]).history
    history["time"] = timer_log.time_log
    # save-history
    with open(f"keras_mode_{mode}/trial_{trial}.dat", "wb") as fp:
        pickle.dump(history, fp)

def main():
    # output_dir
    output_dir = f"keras_mode_{mode}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # clearn directory
    files = glob.glob(output_dir+"/*")
    for f in files:
        os.remove(f)
    # train
    for i in range(nb_trials):
        print("Trial", i, "starts")
        train(i)

if __name__ == "__main__":
    model = create_model(3, 4)
    model.summary()
    exit()

    # consts
    nb_epochs = 100
    nb_trials = 5
    mode = 1  #2, 3
    # main
    main()
