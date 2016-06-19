import boto3
import multiprocessing
import pymongo
import os
import urllib2

client = pymongo.MongoClient()
db = client['flickr_links']
collection = db['landscape group']

bucket = 'reals-landscapes-professional'
s3 = boto3.resource('s3')
s3.create_bucket(Bucket=bucket)


def download_image(link):
    response = urllib2.urlopen(link)
    with open(link[36:], 'wb') as outfile:
        outfile.write(response.read())
    s3.Object(bucket, link[36:]).put(Body=open(link[36:]))
    os.remove(link[36:])
    print link

if __name__ == '__main__':

    url_list = [link['_id'] for link in collection.find()]
    pool = multiprocessing.Pool()

    results = pool.map(download_image, url_list)
