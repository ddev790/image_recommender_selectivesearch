import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import selectivesearch
import psycopg2

from collections import OrderedDict
from PIL import Image
from utils import FeatureExtractor
from psycopg2.extensions import AsIs

# For Reproducibilty
np.random.seed(42)

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats
    if boxes.dtype.kind == "i":
        boxes = boxes.astype('float')
 
    # initialize the list of picked indexes 
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    size = boxes[:, 4]

    # compute the area of the bounding boxes and sort the bounding boxes by the bottom-right 
    # y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of the bounding box and the 
        # smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype('int')


if __name__ == "__main__":
    input_image_path = sys.argv[1]
    fe = FeatureExtractor()  
    data = OrderedDict()
    con = None
    try:
        # connect to the PostgreSQL server
        con = psycopg2.connect("host='localhost' dbname='image_recommender' user='postgres' password='admin'")
        cur = con.cursor()
        cur.execute("""CREATE TABLE k (
                                    image_name text,
                                    x1 integer,
                                    x2 integer,
                                    x3 integer,
                                    x4 integer,
                                    y1 integer,
                                    y2 integer,
                                    y3 integer,
                                    y4 integer,
                                    features double precision[],
                                    enterprise_id text,
                                    model_name text,
                                    model_id text,
                                    segment_id integer,
                                    image_size integer
                                    )""");
        
        for idx, image in enumerate(glob.glob(input_image_path + '*.jpg')):  
            # loading image
            BGR_img = cv2.imread(image)
            img = cv2.cvtColor(BGR_img, cv2.COLOR_BGR2RGB)
            basename = os.path.splitext(os.path.basename(image))[0] 
            # perform selective search
            img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.8, min_size=10)
            candidates = set()
            for r in regions:
                # excluding same rectangle (with different segments)
                if r['rect'] in candidates:
                    continue
                # excluding regions smaller than 2000 pixels
                if r['size'] < 2000:
                    continue
                # distorted rects
                x, y, w, h = r['rect']
                if w / h > 1.2 or h / w > 1.2:
                    continue

                chosen = r['rect'] + (r['size'],)
                candidates.add(chosen)

            bounding_boxes = np.asarray(list(candidates))
            # twenty percent overlapping allowed
            pick = non_max_suppression(bounding_boxes, overlapThresh=0.2)
            # draw rectangles on the original image
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.imshow(img)
            for (x, y, w, h, size) in pick:  
                imgRect = img[y : y + h, x : x + w]
                segmented_image = Image.fromarray(imgRect)
                feature = fe.extract(segmented_image)    # get features from PIL image 
                (x, y, w, h, size) = (x.item(), y.item(), w.item(), h.item(), size.item())    # NumPy dtype to native Python dtype
                # BBOX
                (x1, y1) = (x, y)
                (x2, y2) = (x + w, y)
                (x3, y3) = (x + w, y + h)
                (x4, y4) = (x, y + h)
                # Image Name
                img_keys = ['image_name']
                img_vals = [image]
                X_keys = ['x{}'.format(i) for i in range(1, 5)]
                X_vals = [x1, x2, x3, x4]
                Y_keys = ['y{}'.format(i) for i in range(1, 5)]
                Y_vals = [y1, y2, y3, y4]
                # Features
                F_keys = ['features']
                F_vals = [feature.tolist()]
                # Enterprise Id
                enterprise_id_keys = ['enterprise_id']
                enterprise_id_vals = ['e_123']
                # Model Name
                model_name_keys = ['model_name']
                model_name_vals = ['watches']
                # Model Id
                model_id_keys = ['model_id']
                model_id_vals = ['m_456']
                # Segment Id
                segment_id_keys = ['segment_id']
                segment_id_vals = ['1234']
                # Image size
                image_size_keys = ['image_size']
                image_size_vals = [size]

                data.update(list(zip(img_keys, img_vals)) + list(zip(X_keys, X_vals)) + 
                            list(zip(Y_keys, Y_vals)) + list(zip(F_keys, F_vals)) + 
                            list(zip(enterprise_id_keys, enterprise_id_vals)) + 
                            list(zip(model_name_keys, model_name_vals)) + 
                            list(zip(model_id_keys, model_id_vals)) +
                            list(zip(segment_id_keys, segment_id_vals)) +
                            list(zip(image_size_keys, image_size_vals)))
                
                headers = list(data.keys())
                values = [data[col] for col in headers]
                cur.execute("INSERT INTO k (%s) values %s", (AsIs(','.join(headers)), tuple(values)))
                print(cur.mogrify("INSERT INTO k (%s) values %s", (AsIs(','.join(headers)), tuple(values))))
                # commit the changes
                con.commit()
                plt.close(fig)

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)

    finally:
        if con is not None:
            con.close()
