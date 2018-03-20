from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
import base64
from detection import preprocess as prep
from recognition import compare as cmp
import urllib.request
import time
import pandas as pd
import numpy as np
from logger import logger
import cv2
import argparse


para = cmp.loadModel('model/lightened_cnn/lightened_cnn', 0)

def process():
    dir = "/Users/admin/Work/jaws-recognition-service/main"
    items = os.listdir(dir)
    i = 0
    final = []
    resp = []
    elements = []
    # sn = 0
    # en = 0
    # if args.no >= 15:
    #     sn = args.no
    #     en = args.no +15
    # else:
    #     sn = 0
    #     en = 15
    # Append empty lists in first two indexes.
    elements.append([])
    st = time.time()

    # print("From ",sn )
    # print("To ",en )
    # print(items[sn:en])
    for folder in items:
        print("for folder",folder)
        res = {}
        d = os.path.join(dir,folder)
        dd = os.listdir(d)
        image_urls = []
        for image in dd:
            if image.find("csv") == -1:
                url = 'http://localhost:8888/main/' + folder + '/'+image
                image_urls.append(url)
            else:
                continue
        process1(image_urls,d,folder)
    et = time.time()
    logger.info("Total time Taken {}".format(((et-st)/60)))



def process1(image_urls,d,folder):
    df = pd.DataFrame(dtype=np.int8)
    hashtable = {}
    st = time.time()
    for i in range(len(image_urls)):
        print("**************" + image_urls[i] + "*************************")
        img = urllib.request.urlopen(image_urls[i]).read()

        image_id = image_urls[i].split('/')[-1]
        image_64_encode = base64.encodestring(img)
        image_64_decode = base64.decodestring(image_64_encode)
        # image_64_decode = cv2.resize(image_64_decode, (image_64_decode.shape[0] / 4, image_64_decode.shape[1] / 4))
        dst = time.time()
        detected_imgs = prep.init(image_64_decode)
        det = time.time()
        print("Time Taken In Detection on No. of faces ", (det - dst), len(detected_imgs))

        for index in range(len(detected_imgs)):
            print("For Face************", index)
            rst = time.time()
            detected_face_id = image_id + '_' + str(index)
            noOfFacesInImage = len(detected_imgs)
            response = cmp.compare_two_face(para, detected_imgs[index], hashtable, detected_face_id, image_id,
                                            noOfFacesInImage)
            ret = time.time()
            # print("is_already_exist",response)
            print("Time Taken In Recognition ", (ret - rst))

            if (response['is_already_exist'] == True):
                buildMatrix(response['classId'], image_id, noOfFacesInImage, df)
            elif (response['is_already_exist'] == False):
                buildHashTable(detected_face_id, detected_imgs[index], hashtable)
                buildMatrix(detected_face_id, image_id, noOfFacesInImage, df)

    # print(hashtable)
    df.fillna(0, inplace=True)
    print("**********dataFrame************")
    print(df)
    #print(image_urls[i].split('/')[-1])
    name = os.path.join(d,folder+'_matrix.csv')
    print(name)
    df.to_csv(name)

    dff = get_most_frequent_faces(df)
    print("**********Most Frequent faces dataFrame************")
    print(dff)
    name = os.path.join(d, folder + '_result.csv')
    dff.to_csv(name)
    pics = get_pics_with_most_faces(dff)
    for p in pics[0:3]:
        os.rename(os.path.join(d,p), os.path.join(d,'top_'+p))

    name = os.path.join(d, folder + '_result.csv')
    #pics.to_csv(name)
    # analyse(df)
    print("**********Final Output************")
    print(pics)
    et = time.time()
    logger.info("Time Taken In Detection And Recognition in folder {} {}".format(folder,((et - st) / 60)))
    # print("Time Taken In Detection And Recognition in folder "+folder, (et - st))

def buildHashTable(detected_face_id,detected_face,hashtable):
    hashtable[detected_face_id] = detected_face

def buildMatrix(faceId,imageId,noOfFaces,df):
    df.ix[faceId, imageId] = noOfFaces

# def get_most_frequent_faces(df):
#    df["count"] = df.astype(bool).sum(axis=1)
#    df = df.sort_values(by=["count"], ascending=False)
#    df = df.drop(df.index[3:])
#    df = df.sort_values(by=["count"], ascending=True)
#    df = df.drop("count", axis=1)
#    return df

def get_most_frequent_faces(df):
   df["count"] = df.astype(bool).sum(axis=1)
   df["max_faces"] = df.max(axis=1)
   df = df.sort_values(by=["count","max_faces"], ascending=[False,False])
   df = df.drop(["count","max_faces"], axis=1)
   df = df.reset_index(drop=True)
   return df

# def get_pics_with_most_faces(df):
#    rows = df.shape[0]
#    pics = []
#    for i in range(rows):
#        column = df.ix[i].idxmax()
#        pics.append(column)
#        df.drop(column, axis=1, inplace=True)
#    return pics

def get_pics_with_most_faces(df):
   rows= df.shape[0]
   pics = []
   for i in range(rows):

       if df.ix[i].sum()>0:
           column = df.ix[i].idxmax()
           pics.append(column)
           df.drop(column, axis=1, inplace=True)

   return pics



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('no', type=int)
    # # parser.add_argument('--userid', type=int, default=1, dest='userid')
    # # parser.add_argument('--userid', type=int, default=1, dest='userid')
    # args = parser.parse_args()
    # print("args",args)
    process()

