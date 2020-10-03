from flask import Flask
from flask import request, jsonify
import cv2
import numpy as np
import numpy.linalg as la
import json
import os
import glob
from random import randint

from classes.item_class import Item
from classes.cart_class import Cart
import database.product_db as db
from utils import process_frame

# Elastic Beanstalk looks for an 'application' that is callable by default
app = Flask(__name__)

# Health Check 
@app.route('/healthcheck')
def hello_world():
    return 'Your Server is working!'

# REST API
@app.route('/api/database', methods=['GET', 'POST'])
def database():
    if request.method == 'GET':
        products = db.getAllProducts()
        list = []
        for product in products:
            list.append({"name": product.getName()})
        hashtable = {"productList": list}
        content = json.dumps(hashtable)
        return content
    elif request.method == 'POST':
        content = request.json
        hashtable = json.loads(content)
        name = hashtable["name"]
        price = hashtable["price"]
        db.addProduct(name, price)
        return jsonify({'result': True})

@app.route('/api/database/<name>', methods=['GET', 'PUT', 'DELETE'])
def product(name):
    if request.method == 'DELETE':
        product = db.getProductPerName(name)
        db.deleteProduct(product)
        return jsonify({'result': True})
    elif request.method == 'PUT':
        content = request.json
        hashtable = json.loads(content)
        price = hashtable["price"]
        db.setProductPrice(name, price)
        return jsonify({'result': True})
    elif request.method == 'GET':
        product = db.getProductPerName(name)
        hashtable = {"name": name, "price": product.getPrice()}
        content = json.dumps(hashtable)
        return content

@app.route('/api/datalake', methods=['GET'])
def datalake():
    if request.method == 'GET':
        hashtable = {}
        for folder in glob.glob("dataLake/*"):
            name = os.path.basename(folder)
            hashtable[name] = {}
            for subfolder in glob.glob("%s/*"%folder):
                time = os.path.basename(subfolder)
                files = []
                for file in glob.glob("%s/*"%subfolder):
                    file = os.path.basename(file)
                    files.append(file)
                hashtable[name][time] = files
        content = json.dumps(hashtable)
        return content

@app.route('/api/datalake/<name>', methods=['GET', 'POST'])
def frame(name):
    if request.method == 'POST':
        content = request.json
        hashtable = json.loads(content)
        time_stamp = hashtable["time_stamp"]
        frame_type = hashtable["frame_type"]
        number = hashtable["number"]
        frame = hashtable["frame"]
        frame = cv2.UMat(np.array(frame, dtype=np.uint8))
        if not os.path.exists('dataLake/%s_%s'%(name, frame_type)):
            os.makedirs('dataLake/%s_%s'%(name, frame_type))
        if not os.path.exists('dataLake/%s_%s/%s'%(name, frame_type, time_stamp)):
            os.makedirs('dataLake/%s_%s/%s'%(name, frame_type, time_stamp))
        cv2.imwrite('dataLake/%s_%s/%s/%s.jpg'%(name, frame_type, time_stamp, number), frame)
        return ('', 204)
    if request.method == 'GET':
        content = request.json
        hashtable = json.loads(content)
        time_stamp = hashtable["time_stamp"]
        frame_type = hashtable["frame_type"]
        number = hashtable["number"]
        frame = cv2.imread('dataLake/%s_%s/%s/%s.jpg'%(name, frame_type, time_stamp, number))
        print('dataLake/%s_%s/%s/%s.jpg'%(name, frame_type, time_stamp, number))
        hashtable = {"frame": frame.tolist()}
        content = json.dumps(hashtable)
        return content

@app.route('/api/prediction', methods=['GET', 'POST'])
def prediction():
    content = request.json
    frame = json.loads(content)
    image = cv2.UMat(np.array(frame, dtype=np.uint8))

    # Define constants
    # CONF_THRESHOLD is confidence threshold. Only detection with confidence greater than this will be retained
    # NMS_THRESHOLD is used for non-max suppression
    CONF_THRESHOLD = 0.8
    NMS_THRESHOLD = 0.2

    # Read image from command line arguments
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    classes = []
    with open("classes.txt", "r") as f:
        classes = [cname.strip() for cname in f.readlines()]

    # Load the network with YOLOv3 weights and config using darknet framework
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg", "darknet")

    # Get the output layer names used for forward pass
    outNames = net.getUnconnectedOutLayersNames()

    # Set the input
    net.setInput(blob)

    # Run forward pass
    outs = net.forward(outNames)

    # Process output and draw predictions
    # Get all products ids which were identified
    products_id = process_frame(image, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    qtd = np.zeros(len(classes))
    cart = Cart()
    for item_id in products_id:
        productName = classes[item_id]
        item = Item(productName)
        cart.addProduct(item)
        qtd[item_id] += 1


    img = cv2.UMat.get(image).tolist()
    hashtable = {"cart": cart, "data": img}
    content = json.dumps(hashtable, default=lambda o: o.__dict__)
    #print(content)
    return content

@app.route('/api/get_ref', methods=['GET', 'POST'])
def get_ref():
    content = request.json
    img = json.loads(content)
    #L = 100
    img = np.array(img)
    d1, d2, d3 = img.shape
    img = color_norm(img)
    img = img.reshape((d1*d2, d3))
    #for pixel in img:
        #pixel[0] = int(L*pixel[0]/la.norm(pixel))
        #pixel[1] = int(L*pixel[1]/la.norm(pixel))
        #pixel[2] = int(L*pixel[2]/la.norm(pixel))
    mean = np.mean(img, axis=0)
    std = np.std(img, axis=0)
    hashtable = {"mean": mean.tolist(), "std": std.tolist()}
    content = json.dumps(hashtable)
    return content

def color_norm(img):
    L = 100
    for line in img:
        for pixel in line:
            if la.norm(pixel) != 0:
                pixel[0] = int(L*pixel[0]/la.norm(pixel))
                pixel[1] = int(L*pixel[1]/la.norm(pixel))
                pixel[2] = int(L*pixel[2]/la.norm(pixel))
    return img

@app.route('/api/segmentation', methods=['GET', 'POST'])
def segmentation():
    content = request.json
    hashtable = json.loads(content) #content.json()
    img = hashtable["img"]
    #import numpy as np
    img = np.array(img)
    mean = hashtable["mean"]
    std = np.array(hashtable["std"])
    img_map = []
    img_size = len(img)*len(img[0])
    img_dim = [len(img), len(img[0])]
    count = 0

    # normalize ilunmination
    if img is None:
        return '', 204
    img = color_norm(img)
    std = std*6
    print("mean: (%f, %f, %f), std: (%f, %f, %f)"%(mean[0], mean[1], mean[2], std[0], std[1], std[2]))

    # color filter
    for line in img:
        temp = []
        for pixel in line:
            #print(pixel)
            #print(mean)
            dev = abs(pixel - mean)
            if dev[0] < std[0] and dev[1] < std[1] and dev[2] < std[2]:
            #if pixel[0] > th and pixel[1] > th and pixel[2] > th:
                #pixel[0] = 0
                temp.append(-1)
                count += 1
            else: temp.append(0)
        img_map.append(temp)

    # region growing
    group = 1
    border_gp = []
    while count < img_size:
        # find an unclassified pixel
        while True:
            l = randint(0, img_dim[0]-1)
            w = randint(0, img_dim[1]-1)
            if img_map[l][w] == 0: break
        # check recursively neighbors
        stack = []
        stack.append((l, w))
        img_map[l][w] = group
        while len(stack) > 0:
            l, w = stack.pop()
            count += 1
            #print("count: %d, size: %d, stack: %d"%(count, img_size, len(stack)))
            try:
                if img_map[l-1][w] == 0:
                    stack.append((l-1, w))
                    img_map[l-1][w] = group 
                if img_map[l+1][w] == 0:
                    stack.append((l+1, w))
                    img_map[l+1][w] = group
                if img_map[l][w+1] == 0:
                    stack.append((l, w+1))
                    img_map[l][w+1] = group
                if img_map[l][w-1] == 0:
                    stack.append((l, w-1))
                    img_map[l][w-1] = group

                if img_map[l-1][w-1] == 0:
                    stack.append((l-1, w-1))
                    img_map[l-1][w-1] = group
                if img_map[l+1][w-1] == 0:
                    stack.append((l+1, w-1))
                    img_map[l+1][w-1] = group
                if img_map[l-1][w+1] == 0:
                    stack.append((l-1, w+1))
                    img_map[l-1][w+1] = group
                if img_map[l+1][w+1] == 0:
                    stack.append((l+1, w+1))
                    img_map[l+1][w+1] = group
            except:
                if len(border_gp) == 0 or border_gp[-1] != group: border_gp.append(group)
        group += 1

    #print(group)
    # temp modify img and identify groups
    objs = []
    groups = []
    for i in range(img_dim[0]):
        for j in range(img_dim[1]):
            pixel = img[i][j]
            gp = img_map[i][j]
            if gp in border_gp:
                pixel[1] = 0
            elif gp > 0:
                pixel[2] = 0
                if gp in groups:
                    objs[groups.index(gp)].append((i, j))
                else:
                    groups.append(gp)
                    objs.append([(i, j)])

    
    # create rectangle
    boxes = []
    max_value = 0
    if objs is None or objs == []:
        return '', 204
    for obj in objs:
        min_x = img_dim[1]
        min_y = img_dim[0]
        max_x = 0
        max_y = 0
        npix  = 0 
        for i, j in obj:
            npix += 1
            if j > max_x: max_x = j
            if j < min_x: min_x = j
            if i > max_y: max_y = i
            if i < min_y: min_y = i
        if npix > max_value:
            w = max_x - min_x
            h = max_y - min_y
            x = float(min_x + w/2)/img_dim[1]
            y = float(min_y + h/2)/img_dim[0]
            w = float(w)/img_dim[1]
            h = float(h)/img_dim[0]
            max_value = npix
            box = (x, y, w, h)

    #print(max_value)
    #print(box)
    x, y, w, h = box
    hashtable = {'x': x, 'y': y, 'w': w, 'h': h}
    content = json.dumps(hashtable)
    return content


# Run the application
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production application.
    app.debug = True
    app.run(host="0.0.0.0")
