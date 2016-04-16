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

#Installation
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
supported by the utility, and we later explain each of the flags further:
```
$ python shufl.py -h
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
`-e` controls how many epochs the network will go through while training. As a
note, we trained our model using 21600 samples on an EC2 instance equipped
with a GPU, where each epoch took ~40s resulting in 200 epochs training in
~2.5h.

`-m` controls the execution mode. `train` mode starts training assuming the
already prepared dataset exists. `val` mode enters validation mode, which means
no training is applied, only the validation dataset is fed in the network.
`user` mode is used to fetch the closest predictions of clip id based on an
already trained model. The `user` mode must be followed by the `-t` flag which
defines the target clip id in `CLIP_X` format, where `X` is the id of the track
in the MagnaTagATune dataset.

At the end of each training epoch, `shufl.py` saves the parameters of a
trained model under `data/shufl.pickle`. The `-c` flag provides a checkpoint
support, which when used continues training of the model from the last
persisted epoch.

In order to train the fully fledged model, we set up an EC2 instance on AWS so
we incorporated production environment support with the `-p` flag, to easily
switch between development and production environment. The environments are
being configured but a configuration file under `config/shufl.cfg` and
following is the config file we used
```
[production]
data = 21642  # the total number of training/validation/test samples
mel_x = 599   # the length of the time domain of the mel spectrograms
mel_y = 128   # the number of frequency bins for each sample
latent_v = 40 # the size of the latent vector (1,40)

epochs = 200   # number of training epochs
train_data = 18000  # number of training samples
val_data = 2000     # number of validation samples
test_data = 1600    # number of test samples

train_batchsize = 500 # training minibatch size
val_batchsize = 500   # validation minibatch size
test_batchsize = 100  # test minibatch size

[local]
data=1000
mel_x=599
mel_y=128
latent_v=40

epochs=2
train_data = 800
val_data = 100
test_data = 100

train_batchsize=100
val_batchsize=10
test_batchsize=10
```

`-s` controls shuffling of the training data as they are being fed in the
InputLayer. Shuffling helps the model tackle the problem of moving too fast
towards specific patterns occuring in the dataset. The MagnaTagATune dataset
contains multiple samples of the same track/artist in alphabetical order, so we
wanted to avoid our model learning filters that might contain bias
towards specific instruments for example, which would make it hard for the
network to move away from a genre while going through the training dataset.
Shuffling introduces a tradeoff between the quality of the learnt filters and
memory requirements, as it requires loading the whole dataset in memory which
is about ~6Gb of deserialized vectors, assuming a production environment. We
tackled this issue on the EC2 instance where we were capable of loading the
whole dataset in memory.

Having built this utility, we can easily switch between training and validation
modes and some basic examples are provided below, again assuming a local
execution environment.

By default, given no flags, `shufl.py` trains a model using configuration
parameters for a local execution environment, starting training from scratch
with no shuffling:
```
$ python shufl.py -e 200
Loading local configuration
Loading data...
Building model and compiling functions...
Entered training mode
Starting training from scratch...
Epoch 1 of 200 took 57.074s
  training loss:    0.175772
  validation loss:    0.150191
  validation accuracy:    84.98 %
Saving model...
Epoch 2 of 200 took 60.318s
  training loss:    0.170642
  validation loss:    0.143875
  validation accuracy:    85.61 %
Saving model...
...
```

In order to execute `shufl.py` in a production environment you can use the
following command
```bash
nohup python -u shufl.py --production -m train -s > log.txt.1 2> log.err.1 < /dev/null &
```

This will spawn a process redirecting output to a non-tty, allowing you to
terminate an SSH session without kill the training process.

As stated above, `user` mode provides a means of evaluation a track as follows:
```
$ python shufl.py -m user -t CLIP_2

query: CLIP_2 'classical','strings','opera','violin'
CLIP_10204 'classical','strings','violin','cello'
CLIP_51984 'classical','strings','violin','cello'
CLIP_8713 'drums','electronic','fast','techno','beat'
CLIP_54119 'classical','strings','violin','cello'
CLIP_54125 'classical','strings','violin','cello'
CLIP_51680 'classical','strings','violin','cello'
CLIP_54825 'classical','strings','violin','cello'
CLIP_41176 'classical','strings','violin','cello'
CLIP_49620 'classical','guitar','strings'
CLIP_54824 'classical','strings','violin','cello'
```

You can extract the clip ids using:
```
$ python shufl.py -m user -t CLIP_2 | grep -o 'CLIP_\d\+' | tr '\n' ' '
CLIP_2 CLIP_53212 CLIP_10204 CLIP_51984 CLIP_41176 CLIP_51680 CLIP_54125 CLIP_49620 CLIP_37278 CLIP_33383 CLIP_54825 
```

And using the `clips2mp3.py` script fetch the filenames
```
$ python clips2mp3.py CLIP_2 CLIP_53212 CLIP_10204 CLIP_51984 CLIP_41176 CLIP_51680 CLIP_54125 CLIP_49620 CLIP_37278 CLIP_33383 CLIP_54825
data/magna/f/american_bach_soloists-j_s__bach_solo_cantatas-01-bwv54__i_aria-30-59.mp3
data/magna/c/vito_paternoster-cd1bach_sonatas_and_partitas_for_solo_violin-15-sonata_seconda_in_re_minore__andante-204-233.mp3
data/magna/1/vito_paternoster-cd1bach_cello_suites-02-suite_i_in_sol_maggiore__allemande-88-117.mp3
data/magna/c/vito_paternoster-cd1bach_sonatas_and_partitas_for_solo_violin-14-sonata_seconda_in_re_minore__fuga-291-320.mp3
data/magna/2/vito_paternoster-cd2bach_cello_suites-09-suite_iv_in_mi_bemolle_maggiore__courante-59-88.mp3
data/magna/9/vito_paternoster-cd2bach_sonatas_and_partitas_for_solo_violin-14-partita_terza_in_la_maggiore__bouree-30-59.mp3
data/magna/1/vito_paternoster-cd1bach_cello_suites-16-suite_vi_in_re_magiore__sarabande-88-117.mp3
data/magna/a/jacob_heringman-black_cow-13-bakfark_non_accedat_ad_te_malum_secunda_pars-320-349.mp3
data/magna/d/processor-insomnia-08-shiraio_pig-117-146.mp3
data/magna/9/vito_paternoster-cd2bach_sonatas_and_partitas_for_solo_violin-07-sonata_terza_in_fa_maggiore__fuga-552-581.mp3
data/magna/1/vito_paternoster-cd1bach_cello_suites-17-suite_vi_in_re_magiore__gavotte_i_e_ii-59-88.mp3
```
