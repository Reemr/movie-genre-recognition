import os

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

def create_list_of_frames(src_dir, dest_dir, train_dir):
    index_file_dir = os.path.join(dest_dir, "index.txt")

    if train_dir:
        sub_dir = os.path.join(src_dir, 'train')
        list_file_dir = os.path.join(dest_dir, "list_train_data2.txt")
    else:
        sub_dir = os.path.join(src_dir, 'test')
        list_file_dir = os.path.join(dest_dir, "list_test_data2.txt")

    class_dict = dict()
    with open(index_file_dir) as fo:
        for line in fo:
            class_index, class_name = line.split()
            class_dict[class_name] = class_index

    with open(list_file_dir, 'w') as file_list:
        for class_name in class_dict.keys():
            data_path = os.path.join(sub_dir, class_name)
            list_data = os.listdir(data_path)

            for i, file_name in enumerate(list_data):
                file_path = os.path.join(data_path, file_name)

                file_list.write(file_path + '\n')


if __name__ == '__main__':

    data_dir = os.path.join(os.getcwd(), 'data')
    list_dir = os.path.join(data_dir, 'videoTrainTestlist')
    src_dir = os.path.join(data_dir, 'Movie-dataset-preprocessed')
    #createListFiles(data_dir, 'Movie_samples', 'listdata')

    index = 'index.txt'

    #preprocess_listtxt(list_dir, index, 'train.txt', 'trainlist.txt')
    #preprocess_listtxt(list_dir, index, 'test.txt', 'testlist.txt')

    create_list_of_frames(src_dir, list_dir, train_dir=False)
