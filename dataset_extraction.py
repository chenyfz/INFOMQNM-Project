import zipfile
import os
from io import BytesIO
from multiprocessing import Pool


def extract_dataset(file_input, file_output):
    if os.path.isdir(file_output) is False:
        os.mkdir(file_output)
    with zipfile.ZipFile(file_input, 'r') as zip_dataset:
        p = Pool()
        for name in zip_dataset.namelist():
            if name.endswith('.zip'):
                print('-- Extracting {} --'.format(name))
                with zip_dataset.open(name) as sub_zipfile:
                    content = BytesIO(sub_zipfile.read())
                    p.apply_async(extract_sub_zip_file, args=(content, name, file_output))
        p.close()
        p.join()


def extract_sub_zip_file(content, name, file_output):
    with zipfile.ZipFile(content, 'r') as nested_zip:
        nested_zip.extractall(file_output + '/' + name.rsplit('.', 1)[0])
        print('-- [Done] Extracting {} --'.format(name))


if __name__ == '__main__':
    extract_dataset('./dataset-zip/k-emophone.zip', './dataset')
