import os

# Ganti dengan path ke direktori dataset Anda
dataset_train = 'C:/bangkit/code/dataset/train'
dataset_valid = 'C:/bangkit/code/dataset/validation'
dataset_test = 'C:/bangkit/code/dataset/test'

# List semua subdirektori di dalam path
subdirectories = [d for d in os.listdir(dataset_train) if os.path.isdir(os.path.join(dataset_train, d))]
subdirectories1 = [d for d in os.listdir(dataset_valid) if os.path.isdir(os.path.join(dataset_valid, d))]
subdirectories2 = [d for d in os.listdir(dataset_test) if os.path.isdir(os.path.join(dataset_test, d))]

print(subdirectories)
print(subdirectories1)
print(subdirectories2)

def get_class_names(directory):
    return sorted(os.listdir(directory))

class_names = get_class_names(dataset_train)
print(class_names)