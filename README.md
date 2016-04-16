# shufl
GI15 - Information Retrieval Data Mining Group Project (Group 37)

This project aims to address the coldstart problem in music recommendation and
is meant to be an excercise in Machine Learning. It is largely based on
Sander Dieleman's blog post which can be found here:
http://benanne.github.io/2014/08/05/spotify-cnns.html

We use the MagnaTagATune dataset consisting of approximately 26,000 of 29
seconds music clips encoded in 16 kHz, 32kbps, mono mp3. Each of them is
annotated with a combination of 188 tags, such as 'classical', 'guitar',
'slow', etc. 

First, a 'ground truth' model is obtained using gensim doc2vec algorithm: given
a set of tags for each clip, a 40 vector latent representation is produced.

The mp3 samples are then converted to their (mel) spectrogram representation
using librosa.

Finally a convolutional neural network that follows Sander Dieleman's
architecture is built and trained with Theano and Lasagne on an Amazon AWS GPU
instance.

#Instalation
These instructions are for a computer running MAC OS X and assume no previous
python development present.

Install homebrew package manager (http://brew.sh/):
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

Install python using brew:
```bash
brew install python
```

Make sure /usr/local/bin is in the first line of /etc/paths so that when you
type python the brewed version is used

Move it there if its not:
```bash
sudo nano /etc/paths
```

Install virtualenv so you can sandbox your development environment
```bash
pip install virtualenv
```

Make sure you dont install packages to your global python environment unless
explicitly stated
```bash
$ nano ~/.bashrc
export PIP_REQUIRE_VIRTUALENV=true

gpip(){
    PIP_REQUIRE_VIRTUALENV="" pip "$@"
}
```

Clone shufl to your place of choice
```bash
git clone https://github.com/iogakos/shufl.git
cd shufl
```

Create virtual envirnment for shufl:
```bash
virtualenv venv-shufl
```

Activate the virtual environment:
```bash
source venv-shufl/bin/activate
```

## Install dependancies:
librosa (https://bmcfee.github.io/librosa/install.html)
```bash
pip install librosa
brew install libsamplerate
pip install scikits.samplerate
```

Lasagne + Theano (http://lasagne.readthedocs.org/en/latest/user/installation.html):
```bash
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```
gensim (https://radimrehurek.com/gensim/install.html):
```bash
pip install --upgrade gensim
```

Download audio samples from MagnaTagATune dataset
(http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
```bash
cd ~/Download
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
```

You need to concatenate the partial archive first
```bash
cat mp3.zip.* > magna.zip
unzip magna.zip -d SHUFL_ROOT/scripts/data/magna
```

Some mp3 files are broken/empty - you can find them:
```bash
cd SHUFL_ROOT/scripts/data/magna
$ du -ah SHUFL_ROOT/scripts/data/magna | grep -v "/$" | sort  | head -n 20
  0B	./6/norine_braun-now_and_zen-08-gently-117-146.mp3
  0B	./8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
  0B	./9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
```

Remove these files as they are zero byte long:
```bash
rm ./6/norine_braun-now_and_zen-08-gently-117-146.mp3
rm ./8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
rm ./9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
```

## cloud installation
TODO: writing off the top of my head, verify

follow the stanford guide at http://cs231n.github.io/aws-tutorial/ to setup a gpu instance with Theano and Lasagne preinstalled

ssh into your machine

follow install dependencies

libsamplerate needs to be installed from source

scikits.samplerate might need libsamplerate0-dev
```bash
sudo apt-get install libsamplerate0-dev
pip install scikits.samplerate
```

getting ffmpeg on ubuntu (to be able to load mp3s for librosa)
http://stackoverflow.com/questions/29125229/how-to-reinstall-ffmpeg-clean-on-ubuntu-14-04

```bash
sudo apt-get --purge remove ffmpeg
sudo apt-get --purge autoremove

sudo apt-get install ppa-purge
sudo ppa-purge ppa:jon-severinsson/ffmpeg

sudo add-apt-repository ppa:mc3man/trusty-media
sudo apt-get update
sudo apt-get dist-upgrade

sudo apt-get install ffmpeg
```

Update lasagne and theano to newest versions (see install dependancies)

You will need to get the newest cuDNN (v5) from
https://developer.nvidia.com/cudnn. Register and download the linux archive.

"install" (taken from https://gist.github.com/joyofdata/11e936d0603dd7dd63f6)
```bash
scp -i  ~/.ssh/aws.pem ~/Downloads/cudnn.tar.gz ubuntu@ec2-52-17-84-162.eu-west-1.compute.amazonaws.com:/home/ubuntu/downloads

gzip -d file.tar.gz
tar xf file.tar

echo -e "\nexport LD_LIBRARY_PATH=/home/ubuntu/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH" >> ~/.bashrc

sudo cp cudnn.h /usr/local/cuda-7.0/include
sudo cp libcudnn* /usr/local/cuda-7.0/lib64
```

The persistent (root) filesystem on the aws instance is only 12GB, of which a
lot is taken up by python and some other preinstalled packages. I would
recommend removing caffe and torch if you dont plan to use them and keep the
zipped mp3 samples in ~/downloads so you dont need to redownload it each time

Once booted you have an additional 60 GB of storage under /mnt folder (root
access only) that gets wiped out on reboot. change the owner to ubuntu for
easier access:
```bash
sudo chown -R ubuntu:ubuntu /mnt
```

Extract the mp3s:
```bash
unzip ~/downloads/mp3.zip -d /mnt/data/magna/
```

and remove the broken files:
```bash
rm /mnt/data/magna/6/norine_braun-now_and_zen-08-gently-117-146.mp3
rm /mnt/data/magna/8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
rm /mnt/data/magna/9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
```

To make theano run on gpu by default:
```bash
echo -e '[global]\nfloatX=float32\ndevice=gpu\n\n[mode]=FAST_RUN\n\n[lib]\ncnmem=0.9\n\n[nvcc]\nfastmath=True\n\n[cuda]\nroot=/usr/local/cuda' > ~/.theanorc
```

# Usage

The following includes a walkthrough execution of Shufl's environment starting
from preprocessing the dataset until getting recommendations from the trained
model

The following steps assume you have changed into the `scripts` directory of
Shufl.

First we prepare a line separated file, each line containing the clip id in the
beginning of the line and its corresponding (space separated) tags like the
following
```
58870 vocals foreign voice female_vocal vocal indian singing notenglish women
```

Execute the following script in order to build this file:
```bash
python prepare_tags.py
```

Then, we move from tags to latent vector space representation for each clip
using the Doc2Vec model available by the gensim library
```bash
python train_tags_model.py
```

This will save the Doc2Vec model in a file called `d2vmodel.doc2vec` under
`data` directory. Having switched to the latent vector space, we are now ready
to prepare a fully fledged data for training, validating and testing. By
default the dataset preparation script assumes executing on a local environment
that will only include 1000 samples for training. If you want to prepare a
dataset for a production enviroment, you can use the `-p` flag that takes into
account the whole dataset generating pickle files of approximately `15GB`
(requires ~1.5h in an 8-Core machine). We assume running on a commodity machine
at the moment, thus we prepare the dataset simply by executing:
```bash
python prepare_dataset.py
```

This script generated the following files briefly explained below:
```
711M  mels.pickle # serialized mel spectrograms for training/validation
70M   mels-test.pickle # serialized mel spectrograms for testing
11M   tags.pickle # serialized tags for training/validation
900K  tags-test.pickle # serialized tags for testing
9.4K  clips # line separated file, each line containing the corresponding clip ids
```

Having prepared the dataset we are now ready to move to training and evaluation
using the `shufly.py` script. Following we show different execution modes
supported by the utility, and we later explain each of the flags futher:
```
[ioannisgakos|scripts] python shufl.py -h
usage: shufl.py [-h] [-e N] [-m <mode>] [-t TRACK_ID] [-c] [-p] [-s]

GI15 2016 - Shufl

optional arguments:
  -h, --help            show this help message and exit
  -e N, --epochs N      Number of training epochs (default: 500)
  -m <mode>, --mode <mode>
                        Set to `train` for training, `val` for validating.
                        `user` for user mode (Default `train`)
  -t TRACK_ID, --track-id TRACK_ID
                        Return the closest vectors for a specific track id
                        from the MagnaTagATune dataset
  -c, --checkpoint      Continue training from the last epoch checkpoint
  -p, --production      Load training specific options appropriate for
                        production
  -s, --shuffle         Shuffle the training data (caution: loads the whole
                        datas in memory)
```


