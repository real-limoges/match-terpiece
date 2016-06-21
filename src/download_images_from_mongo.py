import boto3
from multiprocessing import Pool, freeze_support
import pymongo
import os
import urllib2
import sys
import itertools

# Global Variables
client = pymongo.MongoClient()
db = client['flickr_links']
s3 = boto3.resource('s3')


def download_image(link):
    """
    INPUT: Link (string), Bucket (string)
    OUTPUT: None

    Opens the link and downloads the file to disk. Uploads to s3
    """
    print link, bucket_name
    return

    response = urllib2.urlopen(link)
    with open(link[36:], 'wb') as outfile:
        outfile.write(response.read())
    s3.Object(bucket_name, link[36:]).put(Body=open(link[36:]))
    os.remove(link[36:])


def url_generator(collection):
    '''
    INPUTS: MongoDB Collection Cursor
    OUTPUTS: Yields links from a given MongoDB collection
    '''
    for link in collection.find():
        yield link['_id']

if __name__ == '__main__':
    if len(sys.argv) == 3:
        coll_name = sys.argv[1]
        global bucket_name
        bucket_name = sys.argv[2]
    else:
        print """Please provide a string for the MongoDB collection and a string for the s3 bucket"""
        sys.exit(1)

    s3.create_bucket(Bucket=bucket_name)
    collection = db[coll_name]

    pool = Pool(2)
    results = pool.map(download_image, url_list)
