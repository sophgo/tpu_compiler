import cv2
import os
import numpy as np
from scipy import interpolate
from imgaug import augmenters as iaa
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import pymlir

# params
DATA_ROOT = './'
RESIZE_SIZE = 112
IMAGE_SIZE = 32
CHANNEL = 3

def load_val_list():
    list = []
    f = open('wisecore_rgbir_val_list_noCleanOutFace.txt')
    lines = f.readlines()
    
    for line in lines:
        line = line.strip().split(' ')
        list.append(line)
    return list

def TTA_9_cropps(image, target_shape=(32, 32, 3)):
    image = cv2.resize(image, (RESIZE_SIZE, RESIZE_SIZE))
    
    width, height, d = image.shape
    target_w, target_h, d = target_shape
    
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    
    starts = [[start_x, start_y],
              
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],
              
              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]
    
    images = []
    
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index
        
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        
        if x + target_w >= RESIZE_SIZE:
            x = RESIZE_SIZE - target_w-1
        if y + target_h >= RESIZE_SIZE:
            y = RESIZE_SIZE - target_h-1
        
        zeros = image_[x:x + target_w, y: y+target_h, :]
        
        image_ = zeros.copy()
        
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        
    return images

def color_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    
    augment_img = iaa.Sequential([
        iaa.Fliplr(0),
    ])
    
    image =  augment_img.augment_image(image)
    #image = TTA_36_cropps(image, target_shape)
    #print('TTA 36 result ::::::::::::::')
    #print(image[0].shape)
    image = TTA_9_cropps(image, target_shape)
    #image = TTA_5_cropps(image, target_shape)
    #print('TTA 5 result ::::::::::::::')
    #print(image[0].shape)
    #print('color_augumentor output len : {}'.format(len(image)))
    return image

def ir_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    augment_img = iaa.Sequential([
        iaa.Fliplr(0),
    ])
    image =  augment_img.augment_image(image)
    image = TTA_9_cropps(image, target_shape)
    #image = TTA_9_cropps(image, target_shape)
    #image = TTA_5_cropps(image, target_shape)
    return image

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    tpr = 0 if (tp +fn==0) else float(tp) / float(tp +fn)
    fpr = 0 if (fp +tn==0) else float(fp) / float(fp +tn)
    
    acc = float(tp +tn ) /dist.shape[0]
    return tpr, fpr, acc

def calculate(threshold, dist, actual_issame):
    predict_issame = np.less(1-dist, 1-threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    return tp,fp,tn,fn

def TPR_FPR( dist, actual_issame, fpr_target = 0.001):
    print('fpr_target : {}'.format(fpr_target))
    # acer_min = 1.0
    # thres_min = 0.0
    # re = []
    
    # Positive
    # Rate(FPR):
    # FPR = FP / (FP + TN)
    
    # Positive
    # Rate(TPR):
    # TPR = TP / (TP + FN)
    
    thresholds = np.arange(0.0, 1.0, 0.001)
    nrof_thresholds = len(thresholds)
    
    fpr = np.zeros(nrof_thresholds)
    FPR = 0.0
    for threshold_idx, threshold in enumerate(thresholds):
        
        if threshold < 1.0:
            tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
            FPR = fp / (fp*1.0 + tn*1.0)
            TPR = tp / (tp*1.0 + fn*1.0)
            
        fpr[threshold_idx] = FPR
    
    if np.max(fpr) >= fpr_target:
        f = interpolate.interp1d(np.asarray(fpr), thresholds, kind= 'slinear', fill_value="extrapolate")
        threshold = f(fpr_target)
    else:
        threshold = 0.0
    
    tp, fp, tn, fn = calculate(threshold, dist, actual_issame)
    
    FPR = fp / (fp * 1.0 + tn * 1.0)
    TPR = tp / (tp * 1.0 + fn * 1.0)
    
    print('threshold : ' + str(threshold))
    print('FPR@' + str(fpr_target) +' - TPR : ' + str(FPR)+' - '+str(TPR))
    return FPR,TPR

val_list = load_val_list()

probs = []
labels = []
module = pymlir.module()
module.load('liveness-int8.mlir')

for i in tqdm(range(len(val_list))):
    color, ir, label =  val_list[i]
    color = cv2.imread(os.path.join(DATA_ROOT, color),1)
    #depth = cv2.imread(os.path.join(DATA_ROOT, depth),1)
    ir = cv2.imread(os.path.join(DATA_ROOT, ir),1)
    
    color = cv2.resize(color,(RESIZE_SIZE,RESIZE_SIZE))
    #depth = cv2.resize(depth,(RESIZE_SIZE,RESIZE_SIZE))
    ir = cv2.resize(ir,(RESIZE_SIZE,RESIZE_SIZE))
    
    color = color_augumentor(color, target_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),is_infer=True)
    #depth = depth_augumentor(depth, target_shape=(self.image_size, self.image_size, 3),is_infer=True)
    ir = ir_augumentor(ir, target_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),is_infer=True)
    
    # before concate, just a list, concate
    color = np.concatenate(color, axis=0)
    # depth = np.concatenate(depth, axis=0)
    ir = np.concatenate(ir, axis=0)
    
    n = len(color)
    image = np.concatenate([color.reshape([n,IMAGE_SIZE, IMAGE_SIZE, 3]),
                            #depth.reshape([n,self.image_size, self.image_size, 3]),
                            ir.reshape([n,IMAGE_SIZE, IMAGE_SIZE, 3])],
                            axis=3)
    
    image = np.transpose(image, (0, 3, 1, 2))
    image = image.astype(np.float32)
    image = image.reshape([n, CHANNEL * 2, IMAGE_SIZE, IMAGE_SIZE])
    image = image / 255.0

    out = np.zeros((image.shape[0],2))
    for i in range(image.shape[0]):
        temp = module.run(np.expand_dims(image[i,:,:,:], 0))
        out[i] = temp['fc2_dequant']
    
    labels.append(int(label))
    #print(labels)
    probs.append(np.mean(out, axis=0))
    # print(probs)
    
    #if i == 100: break

probs = np.vstack(probs)
probs = softmax(probs)
labels = np.vstack(labels)

#tpr, fpr, acc = calculate_accuracy(0.5, probs[:,1], labels[:,0])
#fpr, tpr_2 = TPR_FPR(probs[:, 1], labels[:,0], fpr_target=0.01)
#fpr, tpr_3 = TPR_FPR(probs[:, 1], labels[:,0], fpr_target=0.001)
#fpr, tpr_4 = TPR_FPR(probs[:, 1], labels[:,0], fpr_target=0.0001)
TPR_FPR(probs[:, 1], labels[:,0], fpr_target=0.0001)

roc_auc_score(labels[:,0], probs[:, 1])

