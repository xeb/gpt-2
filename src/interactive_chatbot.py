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

import model, sample, encoder

PRIMARY_CONSTRUCT = "marcker"

SECONDARY_CONSTRUCTS = [
    { "id": 1,  "display": "Muhammad",   "alias": "yahia" },
    { "id": 2,  "display": "Chris",      "alias": "chriphil" },
    { "id": 3,  "display": "Elaine",     "alias": "+16028186512" },
    { "id": 4,  "display": "Ian",        "alias": "+17149326868" },
    { "id": 5,  "display": "Elaine (iCloud)",     "alias": "elainekbk@icloud.com" },
    { "id": 6,  "display": "Spencer",        "alias": "+14802398932" },
    { "id": 7,  "display": "Dad",        "alias": "+16026721386" },
]

def setup_chat():
    try:
        display_list = '\n'.join([f"{x['id']}-{x['display']}" for x in SECONDARY_CONSTRUCTS])
        person = input(f"Who do you want to talk to?\n{display_list}\n").strip()
        person_alias = ""
        for cnstr in SECONDARY_CONSTRUCTS:
            if cnstr["id"] == int(person):
                person_alias = cnstr["alias"]
        if person_alias == "":
            print("UNKNOWN SELECTION. Try again...")
            person_alias = setup_chat()

        return person_alias
    except:
        return setup_chat()

def person_mask(person_alias):
    for cnstr in SECONDARY_CONSTRUCTS:
        if cnstr["alias"] == person_alias:
            return cnstr["display"]

    return f"UNKNOWN ({person_alias})"

def output_chat(text):
    parsed_text = ""
    messages = text.split('\n')
    for message in messages:
        message = message.replace("<EOM>","").strip()
        if message == "" or message is None:
            continue
        
        message_parts = message.split('|')
        if message_parts[0] == "0": # from the person
            parsed_text = parsed_text + f"{person_mask(message_parts[1])}: "
        else: # from me!
            parsed_text = parsed_text + f"{PRIMARY_CONSTRUCT}: "

        if len(message_parts) < 3:
            print("WHATS HAPPENING!\n\n-----")
            print(f"Found {message_parts} parts which are: {message_parts}")
            print(message)
            print("------------\n")
        elif message_parts[2] is not None:
            parsed_text = parsed_text + message_parts[2] + "\n"

    return parsed_text

def interact_model(
    model_name='355M',
    restore_from='checkpoint/run1/',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=100,
    top_p=0.9,
    penalize=0.85,
    prompt=None
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    :penalize=0.0 : Float value controlling "used" penalty. Implements repetition
     reduction (similar to CTRL) if set to a value > 0. A decent setting might be 0.85
     with temperature 0.3 and top_k 40.
    """
    if batch_size is None:
        batch_size = 1

    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
    
    print(f">>> Model length is {length}")

    with tflex.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, 
            length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p, penalize=penalize
        )

        saver = tflex.Saver()
        if restore_from is None:
          restore_from = os.path.join('models', model_name)
        ckpt = tflex.latest_checkpoint(restore_from)
        print(f"{ckpt}")
        saver.restore(sess, ckpt)

        person_alias = setup_chat()

        while True:
            # Given a prompt (cmd or file)
            if prompt is not None: 
              if os.path.isfile(prompt):
                  with open(prompt) as f:
                      raw_text = f.read()
              else:
                  raw_text = prompt
              if len(raw_text) > 1 and raw_text.endswith('\n'):
                raw_text = raw_text[:-1]
            # Interactively ask for a prompt
            else:
                raw_text = input(f"Message (to {person_mask(person_alias)}) >>> ")
                if not raw_text:
                    raw_text="\n"
                print(f'{PRIMARY_CONSTRUCT}:', repr(raw_text))
                raw_text = f"1|{person_alias}|{raw_text}<EOM>"

            print(f'... sending {raw_text}')
            context_tokens = enc.encode(raw_text)
            generated = 0
            print(f">>>> Getting {nsamples // batch_size} samples")
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                print(f">>>> Returning {batch_size}")
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    # print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    # sys.stdout.write(output_chat(text))

                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    sys.stdout.write(text)

                    sys.stdout.flush()

            print("=" * 80)


# def debug_output():
#     with open("src/sample_output_3.txt",'r') as f:
#         text = f.read()
    
#     output_chat(text)
    
if __name__ == '__main__':
    # debug_output()
    fire.Fire(interact_model)
