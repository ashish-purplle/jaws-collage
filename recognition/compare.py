import mxnet as mx
import numpy as np
import cv2
import time

def read2img(name1, name2, size, ctx):
    pair_arr = np.zeros((2, 1, size, size), dtype=float)
    imgA = cv2.cvtColor(name1, cv2.COLOR_BGR2GRAY);
    imgB = cv2.cvtColor(name2, cv2.COLOR_BGR2GRAY);
    img1 = np.expand_dims(cv2.resize(imgA, (size, size)), axis=0)
    img2 = np.expand_dims(cv2.resize(imgB, (size, size)), axis=0)
    assert (img1.shape == img2.shape == (1, size, size))
    pair_arr[0][:] = img1 / 255.0
    pair_arr[1][:] = img2 / 255.0
    return pair_arr


def group(data, num_r, num, kernel, stride, pad, layer):
    if num_r > 0:
        conv_r = mx.symbol.Convolution(data=data, num_filter=num_r, kernel=(1, 1), name=('conv%s_r' % layer))
        slice_r = mx.symbol.SliceChannel(data=conv_r, num_outputs=2, name=('slice%s_r' % layer))
        mfm_r = mx.symbol.maximum(slice_r[0], slice_r[1])
        conv = mx.symbol.Convolution(data=mfm_r, kernel=kernel, stride=stride, pad=pad, num_filter=num,
                                     name=('conv%s' % layer))
    else:
        conv = mx.symbol.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num,
                                     name=('conv%s' % layer))
    slice = mx.symbol.SliceChannel(data=conv, num_outputs=2, name=('slice%s' % layer))
    mfm = mx.symbol.maximum(slice[0], slice[1])
    pool = mx.symbol.Pooling(data=mfm, pool_type="max", kernel=(2, 2), stride=(2, 2), name=('pool%s' % layer))
    return pool


def lightened_cnn_b_feature():
    data = mx.symbol.Variable(name="data")
    pool1 = group(data, 0, 96, (5, 5), (1, 1), (2, 2), str(1))
    pool2 = group(pool1, 96, 192, (3, 3), (1, 1), (1, 1), str(2))
    pool3 = group(pool2, 192, 384, (3, 3), (1, 1), (1, 1), str(3))
    pool4 = group(pool3, 384, 256, (3, 3), (1, 1), (1, 1), str(4))
    pool5 = group(pool4, 256, 256, (3, 3), (1, 1), (1, 1), str(5))
    flatten = mx.symbol.Flatten(data=pool5)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512, name="fc1")
    slice_fc1 = mx.symbol.SliceChannel(data=fc1, num_outputs=2, name="slice_fc1")
    mfm_fc1 = mx.symbol.maximum(slice_fc1[0], slice_fc1[1])
    drop1 = mx.symbol.Dropout(data=mfm_fc1, p=0.7, name="drop1")
    return drop1


def lightened_cnn_b(num_classes=10575):
    drop1 = lightened_cnn_b_feature()
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=num_classes, name="fc2")
    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    return softmax


def compare(para,detected_face,hashtable,detected_face_id,image_id,noOfFacesInImage):
    _, model_args, model_auxs = para;
    ctx = mx.cpu();
    start = time.time()
    symbol = lightened_cnn_b_feature()
    response ={}
    if(len(hashtable) < 1):
        is_already_exist= False
        response['is_already_exist'] = False
        response['classId'] = detected_face_id
        # df = buildMatrix(detected_face_id, image_id, noOfFacesInImage)
        # if flag == True:
        #     return df
        # else:
        return response

    maxClassScore = 0
    maxClassId =""
    for key,val in hashtable.items():
        model_args['data'] = mx.nd.array(read2img(detected_face,val, 128, ctx), ctx)
        exector = symbol.bind(ctx, model_args, args_grad=None, grad_req="null", aux_states=model_auxs)
        exector.forward(is_train=False)
        exector.outputs[0].wait_to_read()
        output = exector.outputs[0].asnumpy()
        dis = np.dot(output[0], output[1]) / np.linalg.norm(output[0]) / np.linalg.norm(output[1])

        if (dis > maxClassScore):
            maxClassScore=dis
            maxClassId = key
        print("*************max score of Image with {} class name {}".format(key, dis))

    print("-------> Final maxClassId  {} and maxScore {} <----------".format(maxClassId, maxClassScore))

    if (maxClassScore > 0.60):
        is_already_exist = True
        response['is_already_exist'] = True
        response['classId'] = maxClassId
    else:
        is_already_exist =False
        response['is_already_exist'] = False
        response['classId'] = detected_face_id

    return response

    # df = buildMatrix(maxClassId,image_id,noOfFacesInImage)
    # if flag == True:
    #     return df
    # else:
    #     return is_already_exist


def compare_two_face(para,detected_face,hashtable,detected_face_id,image_id,noOfFacesInImage):
    return compare(para,detected_face,hashtable,detected_face_id,image_id,noOfFacesInImage)


def loadModel(modelpath, epoch):
    return mx.model.load_checkpoint(modelpath, epoch)





