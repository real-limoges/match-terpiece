import flickrapi
import time
import pymongo
import sys
import calendar

client = pymongo.MongoClient()
db = client['flickr_links']
with open('/Users/reallimoges/.flickr/key') as f:
    key = f.read().strip()
with open('/Users/reallimoges/.flickr/secret_key') as f:
    secret = f.read().strip()

flickr = flickrapi.FlickrAPI(key, secret, cache=True)

def scrape_tag_public(TAG, collection, min_date, max_date):
    '''
    INPUTS: TAG (string), collection (pymongo cursor), 
            min_date (string), max_date (string)
    OUTPUTS: Inserts into MongoDB new links

    Iterates through a search query on a given tag. Constructs
    a static link and stores it as the _id field in a Mongo Database
    '''
    
    # Figures out how many pages to iterate through
    pages = int(flickr.photos.search(tags=TAG, min_upload_date=min_date,
                max_upload_date=max_date).find('photos').get('pages'))

    for page in xrange(1, pages + 1):
        response = flickr.photos.search(tags=TAG, page=page)
        
        # Breaks out of function if no photos are present
        if len([photo for photo in response.find('photos')]) == 0: return
        
        added = 0
        for photo in response.find('photos'):
            link = generate_link(photo)
            if collection.find_one({'_id': link}) is None:
               collection.insert({'_id': link})
               added += 1

        print "Added {} New Documents from {} to {}".format(added,
                                                min_date, max_date)
        time.sleep(0.5)

def scrape_tag_group(GROUP, TAG, collection):
    '''
    INPUT: GROUP (string), TAG (string), colleciton (MongoDB collection)
    OUTPUT: Inserts into MongoDB new links

    Iterates through a search query on a given tag and group. Constructs
    a static link and stores it in the _id field in a Mongo Database
    '''
    
    # Figures out how many pages to iterate through
    pages = int(flickr.groups.pools.getPhotos(group_id=GROUP,
        tags=TAG).find('photos').get('pages'))
    
    for page in xrange(1, pages + 1):
        response = flickr.groups.pools.getPhotos(group_id=GROUP,
                tags=TAG, page=page)
        
        # Breaks out of function if no photos are present
        if len([photo for photo in response.find('photos')]) == 0: return

        added = 0

        for photo in response.find('photos'):
            link = generate_link(photo)
            if collection.find_one({'_id': link}) is None:
                collection.insert({'_id': link})
                added += 1

        print "Added {} New Documents".format(added)
       
        time.sleep(0.5)


def generate_link(photo):
    '''
    INPUT: Element Tree of a photo
    OUTPUTS : String (static link of photo)
    '''
    
    link = 'https://farm'
    link += str(photo.get('farm')) + '.staticflickr.com/'
    link += str(photo.get('server')) + '/'
    link += str(photo.get('id')) + '_'
    link += str(photo.get('secret')) + '_b.jpg'
    if collection.find_one({'_id': link}) is None:
        collection.insert({'_id': link})
    
    return link
    
def generate_dates():
    '''
    INPUTS: None
    OUTPUTS: List of Tuples (string, string)
    
    Returns a list of tuples. The first item in the tuple is the first
    day of the month. 
    '''
    
    dates=[]
    
    for year in xrange(2014, 2016):
        for month in xrange(1, 13):
            begin = '01'
            end = str(calendar.monthrange(year, month)[1])

            if len(str(month)) == 1: month='0' + str(month)

            begin=str(year) + '-' + str(month) + '-' + begin
            end=str(year) + '-' + str(month) + '-' + end

            dates.append((begin, end))
    return dates

if __name__ == '__main__':

    if len(sys.argv) == 2:
        TAG_NAME=sys.argv[1]
        coll_name = TAG_NAME + ' public'
        collection=db[coll_name]
        dates = generate_dates()
        for date in dates:
            begin, end = date
            scrape_tag_public(TAG_NAME, collection, begin, end)
    elif len(sys.argv) == 3:
        TAG_NAME=sys.argv[1]
        GROUP_ID=sys.argv[2]
        coll_name = TAG_NAME + ' group'
        collection = db[coll_name]
        scrape_tag_group(GROUP_ID, TAG_NAME, collection)
    else:
        print "not enough parameters. please provide a tag"
        sys.exit(1)
