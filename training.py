import os
import keras.callbacks
from generators import seq_generator, img_generator, get_data_list
from models import finetuned_resnet, CNN
from keras.optimizers import SGD
from dataPreprocessing import regenerate_data

CLASSES = 5
BATCH_SIZE = 20

def fit_model(model, train_data, test_data, weights_dir, input_shape, optic=False):

    try:
        if optic:
            train_generator = seq_generator(train_data, BATCH_SIZE, input_shape, CLASSES)
            test_generator = seq_generator(test_data, BATCH_SIZE, input_shape, CLASSES)
        else:
            train_generator = img_generator(train_data, BATCH_SIZE, input_shape, CLASSES)
            test_generator = img_generator(test_data, BATCH_SIZE, input_shape, CLASSES)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        print('Start fitting model')

        while True:
            callbacks = [keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True),
                         keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10, verbose=1, mode='auto'),
                         keras.callbacks.TensorBoard(log_dir='.\\logs\\try', histogram_freq=0, write_graph=True, write_images=True)]
            model.fit_generator(
                train_generator,
                steps_per_epoch=100,
                epochs=30,
                validation_data=test_generator,
                validation_steps=25,
                verbose=1,
                callbacks=callbacks
                )
            print('finished')

            data_dir = os.path.join(os.getcwd(), 'data')
            list_dir = os.path.join(data_dir, 'videoTrainTestlist')
            movie_dir = os.path.join(data_dir, 'Movie-dataset')
            regenerate_data(data_dir, list_dir, movie_dir)

    except KeyboardInterrupt:
        print("Training is intrrupted!")


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data')
    list_dir = os.path.join(data_dir, 'videoTrainTestlist')
    weights_dir = os.getcwd()

    video_dir = os.path.join(data_dir, 'Movie-Preprocessed-OF')
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)
    input_shape = (10, 216, 216, 3)
    weights_dest = os.path.join(weights_dir, 'finetuned_resnet_RGB_65_2.h5')
    model = finetuned_resnet(include_top=True, weights_dir=weights_dest)
    fit_model(model, train_data, test_data, weights_dest, input_shape)

    '''
    video_dir = os.path.join(data_dir, 'OF_data')
    weights_dest = os.path.join(weights_dir, 'temporal_cnn_2.h5')
    train_data, test_data, class_index = get_data_list(list_dir, video_dir)

    input_shape = (216, 216, 18)
    model = CNN(input_shape, weights_dest)
    fit_model(model, train_data, test_data, weights_dest, input_shape, optic=True)
    '''
