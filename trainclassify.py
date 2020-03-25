"""
Train the MobileNet V2 model
"""

import os
import sys
import argparse
import pandas as pd

from mobilenet_v2 import MobileNetv2

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Conv2D, Reshape, Activation
from keras.models import Model
import tensorflow as tf

from keras import backend as K

from keras.utils import multi_gpu_model

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def generate(batch, size):
    """Data generation and augmentation

    # Arguments
        batch: Integer, batch size.
        size: Integer, image size.

    # Returns
        train_generator: train set generator
        validation_generator: validation set generator
        count1: Integer, number of train set.
        count2: Integer, number of test set.
    """

    #  Using the data Augmentation in traning data
    ptrain = 'dataclassify/train'
    pval = 'dataclassify/validation'

    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        ptrain,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    validation_generator = datagen2.flow_from_directory(
        pval,
        target_size=(size, size),
        batch_size=batch,
        class_mode='categorical')

    count1 = 0
    for root, dirs, files in os.walk(ptrain):
        for each in files:
            count1 += 1

    count2 = 0
    for root, dirs, files in os.walk(pval):
        for each in files:
            count2 += 1

    return train_generator, validation_generator, count1, count2


def fine_tune(num_classes, weights, model):
    """Re-build model with current num_classes.

    # Arguments
        num_classes, Integer, The number of classes of dataset.
        tune, String, The pre_trained model weights.
        model, Model, The model structure.
    """
    model.load_weights(weights,by_name=True)

    x = model.get_layer('Dropout').output
    x = Conv2D(num_classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)
    output = Reshape((num_classes,))(x)

    model = Model(inputs=model.input, outputs=output)

    return model

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

def train(batch, epochs, num_classes, size, weights, tclasses):
    """Train the model.

    # Arguments
        batch: Integer, The number of train samples per batch.
        epochs: Integer, The number of train iterations.
        num_classes, Integer, The number of classes of dataset.
        size: Integer, image size.
        weights, String, The pre_trained model weights.
        tclasses, Integer, The number of classes of pre-trained model.
    """

    train_generator, validation_generator, count1, count2 = generate(batch, size)
    log_dir = 'model_training/logclassify/'

    if weights != "False":
        model = MobileNetv2((size, size, 3), tclasses)
        model = fine_tune(num_classes, weights, model)
    else:
        model = MobileNetv2((size, size, 3), num_classes)

    #multigpu
    #model = multi_gpu_model(model, gpus=2)

    print(model.inputs)
    print(model.outputs)

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    opt = Adam()
    earlystop = EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    hist = model.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=count1 // batch,
        validation_steps=count2 // batch,
        epochs=epochs,
        callbacks=[earlystop, logging, checkpoint])

    if not os.path.exists('model_training'):
        os.makedirs('model_training')

    model.save_weights(log_dir + 'trained_weights_final.h5')
    model.save(log_dir + 'trained_classifymodel.h5')

    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])

    tf.train.write_graph(frozen_graph, log_dir, "my_model.pb", as_text=False)

    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model_training/histclassify.csv', encoding='utf-8', index=False)


#python trainclassify.py --classes=2 --size=224 --batch=128 --epochs=50 --weights=False --tclasses=0
#python trainclassify.py --classes=2 --size=224 --batch=128 --epochs=50 --weights=pretrainweight.weight --tclasses=pretrainclassnum
#python trainclassify.py --classes=3 --size=224 --batch=64 --epochs=50 --weights=trained_weights_final.h5 --tclasses=2
def main(argv):
    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--batch",
        default=32,
        help="The number of train samples per batch.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument(
        "--weights",
        default=False,
        help="Fine tune with other weights.")
    parser.add_argument(
        "--tclasses",
        default=0,
        help="The number of classes of pre-trained model.")

    args = parser.parse_args()

    train(int(args.batch), int(args.epochs), int(args.classes), int(args.size), args.weights, int(args.tclasses))


if __name__ == '__main__':
    main(sys.argv)
