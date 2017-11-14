import os
import random
import time

import requests
import sys

import simplejson
from flask import Flask, request, render_template, session, flash, redirect
from celery import Celery
import json

from psycopg2.extensions import AsIs
import psycopg2
from image_process import populate_db

app = Flask(__name__)

# Initialize Celery
celery = Celery(app.name, broker='redis://localhost:6379/0')
celery.conf.update(app.config)



@celery.task(name="segmentation")
def segmentation(image_url, enterprise_id, model_name, model_id):
    val = populate_db(image_url, enterprise_id, model_name, model_id)
    print(val)
    r = requests.post("/callback_url/", data=json.dumps(val))
    dataJSON = json.dumps({"image_url": image_url, "callback_url": "/callback_url/"})
    return dataJSON

@app.route('/segment/<enterprise_id>/<model_name>/<model_id>', methods=['POST'])
def insert_data(enterprise_id, model_name, model_id):
    if request.method == 'POST':
        content = request.json
        print(content)
        image_url = content["image_url"]

    # segmentation(image_url, enterprise_id, model_name, model_id)
    segmentation.delay(image_url, enterprise_id, model_name, model_id)
    return json.dumps({"image_url": image_url, "callback_url": "http://127.0.0.1:5000/callback_url/"})




# // celery process - insert into database - (url will be provided)

@app.route('/callback_url', methods=["POST"])
def callback_response(image_name):
    if request.method == 'POST':
        content = request.json
        print(content)
        # return dataJSON


@app.route('/features/<enterprise_id>/<model_name>/<model_id>/<segment_id>', methods=["GET"])
def features(enterprise_id, model_name, model_id, segment_id):
    dataJSON = {}
    con = None
    try:
        # connect to the PostgreSQL server
        con = psycopg2.connect("host='localhost' dbname='image_recommender' user='postgres' password='admin'")
        cur = con.cursor()
        query_select = "Select enterprise_id,model_name,segment_id,model_id,features from k where segment_id='" + segment_id + "'"
        cur.execute(query_select)
        columns = cur.description
        d = {}
        for value in cur.fetchall():
            for index, column in enumerate(value):
                k = columns[index][0]
                if k == 'segment_id':
                    d[k] = str(column)
                else:
                    d[k] = column

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        sys.exit(1)

    finally:
        if con is not None:
            con.close()

    return json.dumps(d)


if __name__ == "__main__":
    app.run(debug=True)
