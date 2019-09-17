import os
import keras.callbacks
from keras.preprocessing.image import ImageDataGenerator
from generators import seq_generator, img_generator, get_data_list
from models import finetuned_resnet, CNN
from keras.optimizers import SGD

CLASSES = 5
BATCH_SIZE = 20

def fit_model(model, train_dir, test_dir, weights_dir, input_shape, optic=False):

    try:
        train_generator = ImageDataGenerator.flow_from_directory(train_dir,batch_size=20, class_mode='categorical')
        test_generator = ImageDataGenerator.flow_from_directory(test_dir, batch_size=20, class_mode='categorical')

        for data_batch, labels_batch in train_generator:
            print('data batch shape', data_batch.shape)
            print('labels batch shape', labels_batch.shape)
            break
        '''
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
            '''

    except KeyboardInterrupt:
        print("Training is intrrupted!")


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data')
    list_dir = os.path.join(data_dir, 'videoTrainTestlist')
    weights_dir = os.getcwd()
    video_dir = os.path.join(data_dir, 'Movie-dataset-preprocessed')

    train_data = os.path.join(video_dir, 'train')
    test_data = os.path.join(video_dir, 'test')

    print(train_data)
    print(test_data)

    input_shape = (216, 216, 3)
    weights_dest = os.path.join(weights_dir, 'finetuned_resnet_RGB_frames_1.h5')
    #model = finetuned_resnet(include_top=True, weights_dir=weights_dest)
    #fit_model(model, train_data, test_data, weights_dest, input_shape)
