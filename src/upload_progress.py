import boto3
import pymongo
import time
import sys

#Global Variables
s3 = boto3.resource('s3')
client = pymongo.MongoClient()
db = client['flickr_links']

def count_bucket(bucket):
    '''
    INPUT: s3 bucket
    OUTPUT: integer count of items in the bucket
    '''
    
    counter = 0
    for item in bucket.objects.all(): counter += 1
    return counter

def loop_until_term(collection, bucket):
    '''
    INPUT: MongoDB collection, s3 bucket
    OUTPUT: None
    
    Compares the number of items in the MongoDB collection to the number
    of items in the s3 bucket. Loop terminates when they are equal, as the
    task is done.
    '''
    full_count = collection.count()    
    s3_counter = count_bucket(bucket)

    while s3_counter != full_count:
        print "{num:2.2f}% done".format(num=(s3_counter*100.)/full_count)
        time.sleep(20)
        s3_counter = count_bucket(bucket)

if __name__ == '__main__':
    
    if len(sys.argv) == 3:
        coll_name = sys.argv[1]
        bucket_name = sys.argv[2]
    else:

        print """Incorrect number of parameters. Please enter a string for the MongoDB collection and a string for the s3 bucket."""
        sys.exit(1)
    
    collection = db[coll_name]
    bucket = s3.Bucket(bucket_name)

    loop_until_term(collection, bucket)
