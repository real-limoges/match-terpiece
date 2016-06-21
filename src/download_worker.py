import boto3
from multiprocessing import Pool, freeze_support
import pymongo
import os
import urllib2
import sys
import itertools

client = pymongo.MongoClient()
db = client['flickr_links']

s3 = boto3.resource('s3')
s3.create_bucket(Bucket=bucket)


def unpack_func(a_b):
    '''
    INPUT: Tuple of arguments to pass to function
    OUTPUT : Returns Download Image(Link, Bucket)
    '''
    return download_image(*a_b)

def download_image(link, bucket):
    """
    INPUT: Link (string), Bucket (string)
    OUTPUT: None

    Opens the link and downloads the file to disk. Uploads to s3
    """
    return link

    response = urllib2.urlopen(link)
    with open(link[36:], 'wb') as outfile:
        outfile.write(response.read())
    s3.Object(bucket, link[36:]).put(Body=open(link[36:]))
    os.remove(link[36:])

def url_generator(collection):
    for link in collection.find():
        yield link['_id']

if __name__ == '__main__':
    if len(sys.argv) == 3:
        coll_name = sys.argv[1]
        bucket_name = sys.argv[2]
    else:
        print """Please provide a string for the MongoDB collection and a string for the s3 bucket"""
        sys.exit(1)

    
    def download_image(link, bucket):
        """
        INPUT: Link (string), Bucket (string)
        OUTPUT: None

        Opens the link and downloads the file to disk. Uploads to s3
        """
        return link

        response = urllib2.urlopen(link)
        with open(link[36:], 'wb') as outfile:
            outfile.write(response.read())
        s3.Object(bucket, link[36:]).put(Body=open(link[36:]))
        os.remove(link[36:])

    s3.create_bucket(Bucket=bucket_name)
    collection = db[coll_name]

    url_gen = url_generator(collection) 

    pool = Pool()
    results = pool.map(download_image, url_list)
