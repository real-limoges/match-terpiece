import numpy as np
import os
import sys
import boto3
from PIL import Image
from scipy.ndimage import imread
from scipy.misc import imresize
import cv2
s3 = boto3.resource('s3')
extension = '-thumb'

def generate_keys(bucket):
    '''
    INPUT: Bucket name (string) 
    OUTPUT: Yields keys from the s3 bucket.
    '''
    for key in s3.Bucket(bucket).objects.all():
        new_bucket = key.bucket_name + extension
        if check_key(key.key, new_bucket) == False:
            yield {'key': key.key, 'bucket_name': key.bucket_name}

def list_keys(bucket):
    new_bucket = bucket + extension
    keys=[]
    for key in s3.Bucket(bucket).objects.all():

        keys.append(key.key)
    return keys

def check_key(key, bucket_name):
    bucket = s3.Bucket(bucket_name)
    objs = list(bucket.objects.filter(Prefix=key))

    if len(objs)>0 and objs[0].key == key:
        return True
    return False


def transform(key, bucket_name):
    '''
    INPUTS: s3 Object - has bucket_name and key values
    OUTPUTS: None (Side Effects Only)

    Downloads item from the s3 bucket into ../images/. It then opens the
    image and provides basic transformations. Overwrites the original image
    on disk and then uploads the file to s3. Deletes image from local disk.
    '''
    path = os.path.abspath('../images/{}'.format(key))
    
    s3.Bucket(bucket_name).download_file(key, path) 
    new_bucket = bucket_name + extension
   
    img = cv2.imread(path)
    try:
        img = cv2.resize(img, (224, 224)).astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        cv2.imwrite('../images/foo3.jpg', img) 
        s3.Bucket(new_bucket).upload_file(path, key)
        os.remove(path)
        return None
    except Exception as e: 
        print e
        pass



if __name__ == '__main__':
    if len(sys.argv) == 2:
        old_bucket = sys.argv[1]
        new_bucket = old_bucket + '-thumb'
        s3.create_bucket(Bucket=new_bucket)

    else:
        print """Please provide an s3 bucket to transform"""
        sys.exit(1)
        
    set_o_keys = set(list_keys(old_bucket))
    set_n_keys = set(list_keys(new_bucket))
    diff_set = set_o_keys.difference(set_n_keys)
    #l_old_bucket = [old_bucket] * len(list_diff)
    
    del set_o_keys, set_n_keys
    
    for key in list(diff_set): transform(key, old_bucket)
