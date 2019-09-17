import os
import cv2
import numpy as np
import time
import shutil
import scipy.misc


def extract_frames(src, dst):
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        frame_pos = cap.get(1) #the value 1 gets the frame pos
        succ, frame = cap.read()
        if not succ:
            break
        if frame.any():
            frame_res = cv2.resize(frame, (216,216))
            np.save(dst+'_%d'%frame_pos, frame_res)

    cap.release()

def get_list_videos(list_dir,txt):
    txt_path = os.path.join(list_dir, txt)

    with open(txt_path) as files:
        datalist = [file for file in files]

    return datalist

def process_videos(list_dir, movie_dir, dest_dir, txt, train_test_dir):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)


    datalist = get_list_videos(list_dir, txt)

    sub_dir = os.path.join(dest_dir, train_test_dir)

    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

    start_time = time.time()
    print("Extracting frames...")
    for i, clip in enumerate(datalist):
        clip_name = os.path.basename(clip)
        clip_categ = os.path.dirname(clip)
        categ_dir = os.path.join(sub_dir, clip_categ)
        src = os.path.join(movie_dir, clip)
        dst = os.path.join(categ_dir, os.path.splitext(clip_name)[0])
        if not os.path.exists(categ_dir):
            os.mkdir(categ_dir)
        video_time = time.time()
        extract_frames(src, dst)
        vid_elap = time.time() -video_time
        print(" video", i, "time", vid_elap / 60, 'minutes')
    elapsed_time = time.time()-start_time
    print("Total processing time:", (elapsed_time / 60), 'minutes')

if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'data')
    list_dir = os.path.join(data_dir, 'videoTrainTestlist')
    movie_dir = os.path.join(data_dir, 'Movie-dataset')
    dest_dir = os.path.join(data_dir, 'Movie-dataset-preprocessed')
    train_txt = 'train.txt'
    train = 'train'
    test_txt = 'test.txt'
    test = 'test'

    #process_videos(list_dir, movie_dir, dest_dir, train_txt, train)
    #process_videos(list_dir, movie_dir, dest_dir, test_txt, test)
