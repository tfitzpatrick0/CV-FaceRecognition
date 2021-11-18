import os
import os.path

DIR = './cv_image_dataset'

for person in os.listdir(DIR):

    new_dir = os.path.join(DIR, person)

    # Print directory and # of images
    images = len([img for img in os.listdir(new_dir)])
    print('Directory: {dir}\nImages: {imgs}\n'.format(dir=new_dir, imgs=images))

    # Clean paths with only one image
    if len([img for img in os.listdir(new_dir)]) == 1:
        print('Deleting {path}'.format(path=new_dir))
        command = 'rm -r ' + new_dir
        os.system(command)