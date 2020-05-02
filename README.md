# gpt-2 / "dixie"
This is a fork (of a fork) of OpenAI's GPT-2. It is modified to include a server that hosts multiple models in order to generate text. The purpose is to create a chatbot on Alexa. I've included a sanitizer for use in the "chatbot" scenario. Below is a bit about how I'm using this repository, but of course, YMMV. I called it "dixie" after "Dixie Flatline" from William Gibson's Neuromancer (who was a virtual personality construct).

## Training Data Format
First, get your training dataset in the following format:
```
1|mom|Hi! <EOM>
1|mom|How are you doing, Mom? <EOM>
0|mom|Hi Son. <EOM>
0|mom|I'm doing great, how are you? <EOM>
1|mom|Fantastic! <EOM>
```

This is a pipe deliminted format for defining chat sequences. The first character indicates if "you" sent the message. After the pipe separators you define the audience of the chat. Then the message content, then an ```<EOM>``` to note the end of the message.

It's best to get as much data as possible. Anything less than ~2MB is probably going to overfit.

## Setting Up Dependencies
Next, make sure you have CUDA 10.0 install on your system. Then you should create a virtualenv:
```
virtualenv --python=python3.7 venv
. venv/bin/activate
```

There are some dependencies to setup, so do:
```
pip install -r requirements.txt
```

Of course you need to download the model with:
```
python download_model.py 124M
```

There are other model weights available (e.g. 355M), so check your GPU capacity. You'll need 16GB for the 355M model.

## Training
I setup different training scripts prefixed with ```train-``` to kick off training for individual data sets. The format of combined.train is above. Here's an example of one of those scripts:
```
#!/bin/bash
DATA_PATH=/home/xeb/projects/dixie/data/chatbot/combined.train
GPT2_PATH=/home/xeb/projects/dixie-gpt-2
MODEL_NAME=124M
PYTHONPATH=src $GPT2_PATH/venv/bin/python3.7 $GPT2_PATH/train.py \
  --dataset $DATA_PATH \
  --model_name=$MODEL_NAME \
  --save_on_ctrlc \
  --batch_size 1
```

## Server
After you have a fine-tuned model ready to go, you should setup ```server_config.json``` from the sample, do something like:
```
cp server_config.json.sample server_config.json
```

And edit the file to point to the models you want to load. The ```server.py``` implementation will "hotswap" models. Yes, there are race-conditions, but oh well. This is definitely not for prod!

Now fire-up the server with:
```
./server.sh
```

You can run tests with cURL commands. Like the below with assumes ```server_config.json``` has a "general" model defined.
```
time curl -vvv -d'{"model_key":"general","raw_text":"How big is the earth?\n"}' -H"Content-Type: application/json" http://localhost:12345/invocations
```

---------

# gpt-2

Code from the paper ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

We have currently released small (117M parameter) and medium (345M parameter) versions of GPT-2.  While we have not released the larger models, we have [released a dataset](https://github.com/openai/gpt-2-output-dataset) for researchers to study their behaviors.

See more details in our [blog post](https://blog.openai.com/better-language-models/).

## Usage

This repository is meant to be a starting point for researchers and engineers to experiment with GPT-2.

### Some caveats

- GPT-2 models' robustness and worst case behaviors are not well-understood.  As with any machine-learned model, carefully evaluate GPT-2 for your use case, especially if used without fine-tuning or in safety-critical applications where reliability is important.
- The dataset our GPT-2 models were trained on contains many texts with [biases](https://twitter.com/TomerUllman/status/1101485289720242177) and factual inaccuracies, and thus GPT-2 models are likely to be biased and inaccurate as well.
- To avoid having samples mistaken as human-written, we recommend clearly labeling samples as synthetic before wide dissemination.  Our models are often incoherent or inaccurate in subtle ways, which takes more than a quick read for a human to notice.

### Work with us

Please [let us know](mailto:languagequestions@openai.com) if you’re doing interesting research with or working on applications of GPT-2!  We’re especially interested in hearing from and potentially working with those who are studying
- Potential malicious use cases and defenses against them (e.g. the detectability of synthetic text)
- The extent of problematic content (e.g. bias) being baked into the models and effective mitigations

## Development

See [DEVELOPERS.md](./DEVELOPERS.md)

## Contributors

See [CONTRIBUTORS.md](./CONTRIBUTORS.md)

## Fine tuning on custom datasets

To retrain GPT-2 117M model on a custom text dataset:

```
PYTHONPATH=src ./train.py --dataset <file|directory|glob>
```

If you want to precompute the dataset's encoding for multiple runs, you can instead use:

```
PYTHONPATH=src ./encode.py <file|directory|glob> /path/to/encoded.npz
PYTHONPATH=src ./train.py --dataset /path/to/encoded.npz
```

### Gradient Checkpointing

https://github.com/openai/gradient-checkpointing is included to reduce the memory requirements of the model, and can be enabled by `--memory_saving_gradients`. The checkpoints are currently chosen manually (poorly) by just adding layer 10 to the 'checkpoints' collection in model.py. `--memory_saving_gradients` is enabled by default for training the 345M model.

### Validation loss

Set `--val_every` to a number of steps `N > 0`, and "validation" loss against a fixed sample of the dataset will be calculated every N steps to get a better sense of training progress. N around 200 suggested. You can set `--val_dataset` to choose a separate validation dataset, otherwise it defaults to a sample from the train dataset (so not a real cross-validation loss!).

### Optimizer

You can use SGD instead of Adam with `--optimizer sgd`. This also helps conserve memory when training the 345M model. Note: the learning rate needs to be adjusted for SGD, due to not having Adam's gradient normalization (0.0006 seems to be a good number from some experiments).

### Multi gpu (out of date)

To do distributed on multiple GPUs or machines using Horovod:

```
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -x PYTHONPATH=src \
    -mca pml ob1 -mca btl ^openib \
    /home/jovyan/gpt-2/train-horovod.py --dataset encoded.npz
```

## GPT-2 samples

| WARNING: Samples are unfiltered and may contain offensive content. |
| --- |

While we have not yet released GPT-2 itself, you can see some samples from it in the `gpt-2-samples` folder.
We show unconditional samples with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.
We show conditional samples, with contexts drawn from `WebText`'s test set, with default settings (temperature 1 and no truncation), with temperature 0.7, and with truncation with top_k 40.

## Citation

Please use the following bibtex entry:
```
@article{radford2019language,
  title={Language Models are Unsupervised Multitask Learners},
  author={Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya},
  year={2019}
}
```

## Future work

We may release code for evaluating the models on various benchmarks.

We are still considering release of the larger models.

## License

[MIT](./LICENSE)
