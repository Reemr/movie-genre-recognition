
import numpy as np
import scipy.misc
import os, cv2, random
import glob
import shutil
import time
import warnings
from collections import OrderedDict
import concurrent.futures


def optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=False):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError(dest_dir + ' already exists')
    os.mkdir(dest_dir)
    print(dest_dir, 'created')

    # create directory for training data
    dest_train_dir = os.path.join(dest_dir, 'train')
    if os.path.exists(dest_train_dir):
        print(dest_train_dir, 'already exists')
    else:
        os.mkdir(dest_train_dir)
        print(dest_train_dir, 'created')

    # create directory for testing data
    dest_test_dir = os.path.join(dest_dir, 'test')
    if os.path.exists(dest_test_dir):
        print(dest_test_dir, 'already exists')
    else:
        os.mkdir(dest_test_dir)
        print(dest_test_dir, 'created')

    dir_mapping = OrderedDict(
        [(train_dir, dest_train_dir), (test_dir, dest_test_dir)]) #the mapping between source and dest

    print('Start computing optical flows ...')
    for dir, dest_dir in dir_mapping.items():
        print('Processing data in {}'.format(dir))
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class of video
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                # print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process videos one by one
                file_dir = os.path.join(class_dir, filename)
                frames = np.load(file_dir)
                # note: store the final processed data with type of float16 to save storage
                processed_data = stack_optical_flow(frames, mean_sub).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
    print('Finish computing optical flows')



def stack_optical_flow(frames, mean_sub=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)
    num_sequences = frames.shape[0]
    output_shape = frame_shape + (2 * (num_sequences - 1),)  # stacked_optical_flow.shape is (216, 216, 18)
    flows = np.ndarray(shape=output_shape)

    for i in range(num_sequences - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = _calc_optical_flow(prev_gray, next_gray)
        flows[:, :, 2 * i:2 * i + 2] = flow

    if mean_sub:
        flows_x = flows[:, :, 0:2 * (num_sequences - 1):2]
        flows_y = flows[:, :, 1:2 * (num_sequences - 1):2]
        mean_x = np.mean(flows_x, axis=2)
        mean_y = np.mean(flows_y, axis=2)
        for i in range(2 * (num_sequences - 1)):
            flows[:, :, i] = flows[:, :, i] - mean_x if i % 2 == 0 else flows[:, :, i] - mean_y

    return flows


def _calc_optical_flow(prev, next_):
    flow = cv2.calcOpticalFlowFarneback(prev, next_, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    return flow



def combine_list_txt(list_dir):
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

    return trainlist, testlist


def process_frame(frame, img_size, x, y, mean=None, normalization=True, flip=True, random_crop=True):
    if not random_crop:
        frame = scipy.misc.imresize(frame, img_size)
    else:
        frame = frame[x:x+img_size[0], y:y+img_size[1], :]
    # flip horizontally
    if flip:
        frame = frame[:, ::-1, :]
    frame = frame.astype(dtype='float16')
    if mean is not None:
        frame -= mean
    if normalization:
        frame /= 255

    return frame


def process_clip(src_dir, dst_dir, seq_len, img_size, mean=None, normalization=True,
                 horizontal_flip=True, random_crop=True, consistent=True, continuous_seq=False):
    all_frames = []
    cap = cv2.VideoCapture(src_dir)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            all_frames.append(frame)
    # save all frames
    if seq_len is None:
        all_frames = np.stack(all_frames, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, all_frames)
    else:
        clip_length = len(all_frames)
        if clip_length <= 20:
            print(src_dir, ' has no enough frames')
        step_size = int(clip_length / (seq_len + 1))
        frame_sequence = []
        # select random first frame index for continuous sequence
        if continuous_seq:
            start_index = random.randrange(clip_length-seq_len)
        # choose whether to flip or not for all frames
        if not horizontal_flip:
            flip = False
        elif horizontal_flip and consistent:
            flip = random.randrange(2) == 1
        if not random_crop:
            x, y = None, None
        xy_set = False
        for i in range(seq_len):
            if continuous_seq:
                index = start_index + i
            else:
                index = i*step_size + random.randrange(step_size)
            frame = all_frames[index]
            # compute flip for each frame
            if horizontal_flip and not consistent:
                flip = random.randrange(2) == 1
            if random_crop and consistent and not xy_set:
                x = random.randrange(frame.shape[0]-img_size[0])
                y = random.randrange(frame.shape[1]-img_size[1])
                xy_set = True
            elif random_crop and not consistent:
                x = random.randrange(frame.shape[0]-img_size[0])
                y = random.randrange(frame.shape[1]-img_size[1])
            frame = process_frame(frame, img_size, x, y, mean=mean, normalization=normalization,
                                  flip=flip, random_crop=random_crop)
            frame_sequence.append(frame)
        frame_sequence = np.stack(frame_sequence, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0]+'.npy'
        np.save(dst_dir, frame_sequence)

    cap.release()

def preprocessing(list_dir, movie_dir, dest_dir, seq_len, img_size, overwrite=False, normalization=True,
                  mean_subtraction=True, horizontal_flip=True, random_crop=True, consistent=True, continuous_seq=False):
    '''
    Extract video data to sequence of fixed length, and save it in npy file
    :param list_dir:
    :param Movie_dir:
    :param dest_dir:
    :param seq_len:
    :param img_size:
    :param overwrite: whether overwirte dest_dir
    :param normalization: normalize to (0, 1)
    :param mean_subtraction: subtract mean of RGB channels
    :param horizontal_flip: add random noise to sequence data
    :param random_crop: cropping using random location
    :param consistent: whether horizontal flip, random crop is consistent in the sequence
    :param continuous_seq: whether frames extracted are continuous
    :return:
    '''
    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError('Destination directory already exists')
    os.mkdir(dest_dir)
    #trainlist = combine_list_txt(list_dir)
    trainlist, testlist = combine_list_txt(list_dir)
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)
    if mean_subtraction:
        mean = calc_mean(movie_dir, img_size).astype(dtype='float16')
        np.save(os.path.join(dest_dir, 'mean.npy'), mean)
    else:
        mean = None

    print('Preprocessing Movie data ...')
    for clip_list, sub_dir in [(trainlist, train_dir), (testlist, test_dir)]:
        for clip in clip_list:
            clip_name = os.path.basename(clip)
            #print("clip name = " + clip_name)
            clip_category = os.path.dirname(clip)
            #print("clip category = " + clip_category)
            category_dir = os.path.join(sub_dir, clip_category)
            #print("sub dir = " + sub_dir)
            #print("category dir = " + category_dir)
            src_dir = os.path.join(movie_dir, clip)
            #print("source = "+src_dir)
            dst_dir = os.path.join(category_dir, clip_name)
            #print("destination = "+dst_dir)
            #print()
            if not os.path.exists(category_dir):
                os.mkdir(category_dir)
            process_clip(src_dir, dst_dir, seq_len, img_size, mean=mean, normalization=normalization, horizontal_flip=horizontal_flip,
                         random_crop=random_crop, consistent=consistent, continuous_seq=continuous_seq)


    print('Preprocessing done ...')


def calc_mean(movie_dir, img_size):
    frames = []
    print('Calculating RGB mean ...')
    for dirpath, dirnames, filenames in os.walk(movie_dir):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    # successful read and frame should not be all zeros
                    if ret and frame.any():
                        if frame.shape != (240, 320, 3):
                            frame = scipy.misc.imresize(frame, (240, 320, 3))
                        frames.append(frame)
                cap.release()
    frames = np.stack(frames)
    mean = frames.mean(axis=0, dtype='int64')
    mean = scipy.misc.imresize(mean, img_size)
    print('RGB mean is calculated over', len(frames), 'video frames')
    return mean


def preprocess_flow_image(flow_dir):
    videos = os.listdir(flow_dir)
    for video in videos:
        video_dir = os.path.join(flow_dir, video)
        flow_images = os.listdir(video_dir)
        for flow_image in flow_images:
            flow_image_dir = os.path.join(video_dir, flow_image)
            img = scipy.misc.imread(flow_image_dir)
            if np.max(img) < 140 and np.min(img) > 120:
                print('remove', flow_image_dir)
                os.remove(flow_image_dir)


def regenerate_data(data_dir, list_dir, Movie_dir):
    start_time = time.time()
    sequence_length = 10
    image_size = (216, 216, 3)

    dest_dir_pre = os.path.join(data_dir, 'Movie-Preprocessed-OF')
    # generate sequence for optical flow
    preprocessing(list_dir, Movie_dir, dest_dir_pre, sequence_length, image_size, overwrite=True, normalization=False,
                  mean_subtraction=False, horizontal_flip=False, random_crop=True, consistent=True, continuous_seq=True)

    # compute optical flow data
    src_dir = dest_dir_pre
    dest_dir = os.path.join(data_dir, 'OF_data')
    optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=True)

    elapsed_time = time.time() - start_time
    print('Regenerating data takes:', int(elapsed_time / 60), 'minutes')


def preprocess_listtxt(list_dir, index, txt, txt_dest):

    index_dir = os.path.join(list_dir, index)
    txt_dir = os.path.join(list_dir, txt)
    dest_dir = os.path.join(list_dir, txt_dest)

    class_dict = dict()
    with open(index_dir) as fo:
        for line in fo:
            class_index, class_name = line.split()
            class_dict[class_name] = class_index
    #print(class_dict)
    #print()
    with open(txt_dir, 'r') as fo:
        lines = [line for line in fo]
    #print(lines)
    with open(dest_dir, 'w') as fo:
        for line in lines:
            class_name = os.path.dirname(line)
            class_index = class_dict[class_name]
            fo.write(line.rstrip('\n') + ' {}\n'.format(class_index))

def createListFiles(data_dir, src_name, dest_name):
    src_dir = os.path.join(data_dir, src_name)
    dest_dir = os.path.join(data_dir, dest_name )

    data_files = os.listdir(src_dir)

    classind = {}
    for c, fi in enumerate(data_files):
        classind[fi] = c

    if os.path.exists(dest_dir):
        print("path already exits")
    else:
        os.mkdir(dest_dir)

    ind = os.path.join(dest_dir, 'index.txt')
    with open(ind, 'w') as f:
        for k, v in classind.items():
            f.write('{}'.format(v) + ' ' + k + '\n')

    train_list = os.path.join(dest_dir, 'train.txt')
    test_list = os.path.join(dest_dir, 'test.txt')

    with open(train_list, 'w') as tr, open(test_list, 'w') as ts:
        for fol in classind.keys():
            data_path = os.path.join(src_dir, fol)
            data_list = os.listdir(data_path)

            data_list_size = len(data_list)
            div = round(0.8 * data_list_size)

            for i, fil in enumerate(data_list):
                file_path = os.path.join(fol, fil)
                #divid the data into train and test
                #the division can be defined here
                if i < div:
                    tr.write(file_path + '\n')
                else:
                    ts.write(file_path + '\n')


if __name__ == '__main__':

    sequence_length = 10
    image_size = (216, 216, 3)

    data_dir = os.path.join(os.getcwd(), 'data')
    list_dir = os.path.join(data_dir, 'videoTrainTestlist')
    movie_dir = os.path.join(data_dir, 'Movie-dataset')

    print("executed!")
    #frames_dir = os.path.join(data_dir, 'frames\\mean.npy')

    #createListFiles(data_dir, movie_dir_name, list_name)

    # add index number to testlist file
    #index = 'index.txt'

    #preprocess_listtxt(list_dir, index, 'train.txt', 'trainlist.txt')
    #preprocess_listtxt(list_dir, index, 'test.txt', 'testlist.txt')



    regenerate_data(data_dir, list_dir, movie_dir)
