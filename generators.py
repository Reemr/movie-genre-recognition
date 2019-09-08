import os
import numpy as np
import random
import scipy.misc


def seq_generator(datalist, batch_size, input_shape, num_classes):

    x_shape = (batch_size,) + input_shape
    y_shape = (batch_size, num_classes)

    while True:
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)

        for i in range(batch_size):
            step = random.randint(1, len(datalist) - 1)
            index = (index + step) % len(datalist)
            clip_dir, clip_class = datalist[index]
            batch_y[i, clip_class - 1] = 1
            clip_dir = os.path.splitext(clip_dir)[0] + '.npy'

            count = 0
            while not os.path.exists(clip_dir):
                count += 1
                if count > 20:
                    raise FileExistsError('Too many file missing')
                index = (index + 1) % len(datalist)
                clip_dir, class_idx = datalist[index]
            clip_data = np.load(clip_dir)
            if clip_data.shape != batch_x.shape[1:]:
                raise ValueError('The number of time sequence is nconsistent with the video data')
            batch_x[i] = clip_Data

        yield batch_x, batch_y


def img_generator(data_list, batch_size, input_shape, num_classes):

    batch_image_shape = (batch_size,) + input_shape[1:]
    batch_image = np.ndarray(batch_image_shape)

    video_gen = sequence_generator(data_list, batch_size, input_shape, num_classes)

    while True:
        batch_video, batch_label = next(video_gen)
        for idx, video in enumerate(batch_video):
            sample_frame_idx = random.randint(0, input_shape[0] - 1)
            sample_frame = video[sample_frame_idx]
            batch_image[idx] = sample_frame

        yield batch_image, batch_label

def get_data_list(list_dir, video_dir):
    '''
    Input parameters:
    list_dir: 'root_dir/data/ucfTrainTestlist'
    video_dir: directory that stores source train and test data

    Return value:
    test_data/train_data: list of tuples (clip_dir, class index)
    class_index: dictionary of mapping (class_name->class_index)
    '''
    train_dir = os.path.join(video_dir, 'train')
    test_dir = os.path.join(video_dir, 'test')
    testlisttxt = 'testlist.txt'
    trainlisttxt = 'trainlist.txt'

    testlist = []
    txt_path = os.path.join(list_dir, testlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            testlist.append(line[:line.rfind(' ')])

    trainlist = []
    txt_path = os.path.join(list_dir, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    class_index = dict()
    class_dir = os.path.join(list_dir, 'classInd.txt')
    with open(class_dir) as fo:
        for line in fo:
            class_number, class_name = line.split()
            class_number = int(class_number)
            class_index[class_name] = class_number

    train_data = []
    for i, clip in enumerate(trainlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(train_dir, clip)
        train_data.append((dst_dir, class_index[clip_class]))

    test_data = []
    for i, clip in enumerate(testlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(test_dir, clip)
        test_data.append((dst_dir, class_index[clip_class]))

    return train_data, test_data, class_index

if __name__ == "__main__":
