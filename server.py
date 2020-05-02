#!/usr/bin/env python3

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path += [os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import fire
import json
import os
import numpy as np
import tensorflow as tf 
import tflex
import re
import time
import requests
import model, sample, encoder
import subprocess

from sanitizers.chatbot import sanitize as chat_sanitize
from flask import Flask, request, jsonify
from flask_apscheduler import APScheduler
from multiprocessing import Process

app = Flask(__name__)

class ModelController(object):
    def __init__(self, models):
        self.models = models
        self.active_model_key = None

    def acquire(self, key):
        self.active_model_key = key
        for k in self.models.keys():
            if k != key:
                self.models[k].close()
        
        self.models[key].start_session()
        return self.models[key]


class Model(object):
    def __init__(self, model_name, restore_from, sanitize):
        self.N_SAMPLES = 1
        self.BATCH_SIZE = 1
        self.model_name = model_name
        self.restore_from = restore_from
        self.do_sanitize = sanitize is not None
        self.sanitize = sanitize
        self.seed = None
        self.no_cuda = False # just hard code this for now
        self.inferences = []
        
        self.enc = encoder.get_encoder(self.model_name)
        self.hparams = model.default_hparams()
        with open(os.path.join('models', self.model_name, 'hparams.json')) as f:
            self.hparams.override_from_dict(json.load(f))

        self.sess = None
        self.sess_closed = False
        self.pattern = re.compile('[^\w]')
        self.last_used = time.time()
        self.model_expiration = 30 # seconds

    def has_session(self):
        return self.sess is not None and self.sess_closed is False

    def start_session(self):
        if self.has_session():
            return 

        self.graph = tf.Graph()
        config = tf.ConfigProto(device_count = {'GPU':0})
        self.sess = tflex.Session(graph=tf.Graph(), config=config if self.no_cuda else None)
        self.sess.__enter__()
        self.context = tf.placeholder(tf.int32, [self.BATCH_SIZE, None])
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.output = sample.sample_sequence(
            hparams=self.hparams, 
            length=120,#self.hparams.n_ctx // 2,
            context=self.context,
            batch_size=1,
            temperature=0.3, top_k=40, top_p=0.9, penalize=0.85
        )

        saver = tflex.Saver()
        ckpt = tflex.latest_checkpoint(self.restore_from)
    
        saver.restore(self.sess, ckpt)

    def close(self):
        if self.sess is not None:
            print("CLEANING MODEL")
            self.sess.close()
            self.sess_closed = True
            # del self.sess
            # self.sess = None
        else:
            print("...not cleaning")
        #self.inferences.clear()

    def clean(self):
        return
        now = time.time()
        print(f"self.sess exists {self.sess is not None}")
        if self.last_used > 0 and self.last_used + self.model_expiration < now:
            print(f"Cleaning up model {self.last_used} + {self.model_expiration} < {now}")
            self.close()
        else:
            print(f"Not cleaning up model {self.last_used} + {self.model_expiration} < {now}")
        
    def predict(self, raw_text):
        print(f"-------INPUT------\n{raw_text}\n--------------\n")
        self.start_session()
        self.last_used = time.time()
        print(f"last_used == {self.last_used}")

        print(f"GPU_available: {tf.test.is_gpu_available()}")
        if len(raw_text) > 1 and raw_text.endswith('\n'):
            raw_text = raw_text[:-1]

        context_tokens = self.enc.encode(raw_text)
        self.inferences.append(raw_text)

        generated = 0
        print(f">>>> Getting {self.N_SAMPLES // self.BATCH_SIZE} samples")
        for _ in range(self.N_SAMPLES // self.BATCH_SIZE):
            out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens for _ in range(self.BATCH_SIZE)]
            })[:, len(context_tokens):]
            print(f">>>> Returning {self.BATCH_SIZE}")
            for i in range(self.BATCH_SIZE):
                generated += 1
                text = self.enc.decode(out[i])                
                return self.sanitize(text) if self.do_sanitize else text


def initializer(config_path="server_config.json"):
        print(f"Initializing with {config_path}")

        with open(config_path, 'r') as cfile:
            data=cfile.read()

        obj = json.loads(data)
        obj['JOBS'] = [
            {
                'id': 'cleanup',
                'func': 'server:_call_clean',
                'trigger': 'interval',
                'seconds': 10,
            }
        ]
        obj['SCHEDULER_API_ENABLED'] = True

        print(f"Loaded obj {obj.keys()}")

        models = {}
        for item in obj['models']:
            print(f"Loading {item}")
            model = Model(item['model_name'],
                item['restore_from'],
                chat_sanitize if 'sanitizer' in item else None)
            models[item['key']] = model

        return (models, obj)

MODELS, CONFIG = fire.Fire(initializer)
CONTROLLER = ModelController(MODELS)
BASE_URL = f'http://{CONFIG["host"]}:{CONFIG["port"]}/'

def _infer(robj):
    global MODELS
    if robj is None:
        return "UNKNOWN"

    model_key = 'chatbot'
    if 'model_key' in robj:
        model_key = robj['model_key']

    if model_key not in MODELS:
        print(f"Could not find {model_key} in MODELS ({MODELS.keys()})")
        return "UNKNOWN"

    raw_text = robj['raw_text']
    result = CONTROLLER.acquire(model_key).predict(raw_text)
    return result

def _call_remote(url, robj):
    r = requests.post(url, json=robj)
    print(f'_call_remote: status_code == {r.status_code}')
    print(r.status_code)
    print(r.raw)

def _call_infer(robj):
    return _call_remote(f'{BASE_URL}/invocations', robj)

@app.route('/invocations', methods=['POST'])
def infer():
    return _infer(request.json)

@app.route('/ping')
def pong():
    return "Pong!"

@app.route('/prime', methods=['POST'])
def prime():
    robj = request.json
    if "raw_text" not in robj:
        robj["raw_text"] = "prime this bitch"

    p = Process(target=_call_infer, args=(robj,))
    p.start()
    
    return f"primed {robj['model_key']}"


@app.route('/models/list')
def list_models():
    robj = dict()
    robj['models'] = list()
    for k in MODELS.keys():
        robj['models'].append({ 'model_key': k, 'num_inferences' : len(MODELS[k].inferences) })

    robj['active_model'] = { 'key': CONTROLLER.active_model_key }

    return jsonify(robj)

@app.route('/gputemp')
def gputemp():
    result = subprocess.run(['nvidia-smi'], capture_output=True)
    temps = []
    output = str(result.stdout).split('\\n')
    print(f'There are {len(output)} lines')
    for line in output:
        if "%" in line:
            tmps = line.split(' ')
            temps.append(tmps[4])

    return jsonify(temps)

@app.route('/')
def hello():
    return "<html><body><h1 style='color:#AAAA00;font-family:consolas'>hello friend</style></body></html>"

if __name__ == '__main__':
    app.run(host=CONFIG['host'], port=CONFIG['port'], debug=CONFIG['debug'])
