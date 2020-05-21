from PIL import Image
import os

# origin_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/origin'
# label_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/label'
origin_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/origin'
label_folder_path = '/media/ding/Study/First_next_term/Lake Segmentation/dataset/label'

def get_data(mode):
    if mode == 'train' or mode == 'val' or mode == 'test':
        x_paths = []
        y_paths = []
        tmp_origin_folder_path = os.path.join(origin_folder_path, mode)
        tmp_label_folder_path = os.path.join(label_folder_path, mode)

        # os.walk helps to find all files in directory.
        for (path, dirname, files) in os.walk(tmp_origin_folder_path):
            for filename in files:
                x_paths.append(os.path.join(path, filename))

        # Find ground_truth file paths with x_paths.
        idx = len(tmp_origin_folder_path) #取出源图片路径，与下面的标签文件对应
        for x_path in x_paths:
            y_paths.append(tmp_label_folder_path + x_path[idx:-4] + '.png')

        return x_paths, y_paths
    else:
        print("Please call get_data function with arg 'train', 'val', 'test'.")
x_train_paths, y_train_paths = get_data('train')
print(x_train_paths)
print(y_train_paths)
x_val_paths, y_val_paths = get_data('val')
x_test_paths, y_test_paths = get_data('test')
num_data = len(y_train_paths)
# num_data = len(y_test_paths)
# num_data = len(y_val_paths)
print(num_data)
# for i in range(num_data):
#     I = Image.open(y_train_paths[i])
#     # print(y_train_paths)
#     # I.show()
#     L = I.convert('L')
#     # L.show()
#     L.save(y_train_paths[i][:-4]+ '-1' + '.png')