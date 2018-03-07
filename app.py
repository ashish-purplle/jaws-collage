import argparse
import os
import base64
from detection import preprocess as prep
from recognition import compare as cmp
import urllib.request
import time
import pandas as pd
import numpy as np

para = cmp.loadModel('model/lightened_cnn/lightened_cnn', 0)

def process():
    dir = "/Users/admin/Work/jaws-recognition-service/main"
    items = os.listdir(dir)
    print(items)
    i = 0
    final = []
    resp = []
    elements = []
    image_urls = []
    # Append empty lists in first two indexes.
    elements.append([])
    for folder in items:
        res = {}

        # images = os.listdir(os.path.join(dir, folder))
        url = 'http://localhost:8888/main/' + folder
        image_urls.append(url)

        # for img in images:
        #     url = 'http://localhost:8888/main/' + img
        #     image_urls.append(url)
    # image_urls = ['http://localhost:8888/main/539266179/chetan_ashish.jpg','http://localhost:8888/main/539266232/faraz.jpg','http://localhost:8888/main/539266232/faraj_ashish_rahul.jpg']
    print(image_urls)
    df = pd.DataFrame(dtype=np.int8)
    hashtable = {}

    st = time.time()
    for i in range(len(image_urls)):
        img = urllib.request.urlopen(image_urls[i]).read()
        image_id = image_urls[i].split('/')[-1]
        image_64_encode = base64.encodestring(img)
        image_64_decode = base64.decodestring(image_64_encode)
        dst = time.time()
        detected_imgs = prep.init(image_64_decode)
        det =time.time()
        print("Time Taken In Detection on No. of faces ", (det - dst),len(detected_imgs))

        for index in range(len(detected_imgs)):
            print("For Face************",index)
            rst = time.time()
            detected_face_id =image_id + '_' + str(index)
            noOfFacesInImage = len(detected_imgs)
            response = cmp.compare_two_face(para,detected_imgs[index],hashtable,detected_face_id,image_id,noOfFacesInImage)
            ret = time.time()
            #print("is_already_exist",response)
            print("Time Taken In Recognition ", (ret - rst))

            if(response['is_already_exist'] == True):
                buildMatrix(response['classId'], image_id, noOfFacesInImage,df)
            elif(response['is_already_exist'] == False):
                buildHashTable(detected_face_id,detected_imgs[index],hashtable)
                buildMatrix(detected_face_id,image_id,noOfFacesInImage,df)

    #print(hashtable)
    df.fillna(0, inplace=True)
    print("**********dataFrame************")
    print(df)

    dff = get_most_frequent_faces(df)
    print("**********Most Frequent faces dataFrame************")
    print(dff)
    pics = get_pics_with_most_faces(dff)
    #analyse(df)
    print("**********Final Output************")
    print(pics)
    et = time.time()
    print("Final Time Taken In Detection And Recognition ", (et - st))

def buildHashTable(detected_face_id,detected_face,hashtable):
    hashtable[detected_face_id] = detected_face

def buildMatrix(faceId,imageId,noOfFaces,df):
    df.ix[faceId, imageId] = noOfFaces

def get_most_frequent_faces(df):
   df["count"] = df.astype(bool).sum(axis=1)
   df = df.sort_values(by=["count"], ascending=False)
   df = df.drop(df.index[3:])
   df = df.sort_values(by=["count"], ascending=True)
   df = df.drop("count", axis=1)
   return df

def get_pics_with_most_faces(df):
   rows = df.shape[0]
   pics = []
   for i in range(rows):
       column = df.ix[i].idxmax()
       pics.append(column)
       df.drop(column, axis=1, inplace=True)
   return pics



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument('--urls', type=int, default=1, dest='urls')
    # parser.add_argument('--userid', type=int, default=1, dest='userid')
    # parser.add_argument('--userid', type=int, default=1, dest='userid')
    # args = parser.parse_args()
    process()

