# shufl
GI15 - Information Retrieval Data Mining Group Project

This project aims to address the coldstart problem in music recommendation and is ment to be an excercise in Machine Learning. It is largely based on benanne's blog post which can be found here: http://benanne.github.io/2014/08/05/spotify-cnns.html

We use MagnaTagATune dataset consisting of approximately 26,000 of 29 seconds music clips encoded in 16 kHz, 32kbps, mono mp3. Each of them is annotated with a combination of 188 tags, such as 'classical', 'guitar', 'slow', etc. 

First, a 'ground truth' model is obtained using gensim doc2vec algorithm: given a set of tags for each clip, a 40 vector latent representation is produced.

Then the mp3 samples are converted to their (mel) spectrogram representations using librosa.

Finally a convolutional neural network that follows benanne's architecture is built and trained with Theano and Lasagne on an Amazon AWS GPU instance.

#Instalation
These instructions are for a computer running MAC OS X and assume no previous python development present

install homebrew package manager (http://brew.sh/):
```bash
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

install python using brew:
```bash
brew install python
```

make sure /usr/local/bin is in the first line of /etc/paths so that when you type python the brewed version is used
move it there if its not:
```bash
sudo nano /etc/paths
```

install virtualenv so you can sandbox your development environment
```bash
pip install virtualenv
```

make sure you dont install packages to your global python environment unless explicitly stated
```bash
$ nano ~/.bashrc
export PIP_REQUIRE_VIRTUALENV=true

gpip(){
    PIP_REQUIRE_VIRTUALENV="" pip "$@"
}
```

clone shufl to your place of choice
```bash
git clone https://github.com/iogakos/shufl.git
cd shufl
```

create virtual envirnment for shufl:
```bash
virtualenv venv-shufl
```

activate the virtual environment:
```bash
source venv-shufl/bin/activate
```

## install dependancies:
librosa (https://bmcfee.github.io/librosa/install.html)
```bash
pip install librosa
brew install libsamplerate
pip install scikits.samplerate
```

lasagne + theano (http://lasagne.readthedocs.org/en/latest/user/installation.html):
```bash
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```
gensim (https://radimrehurek.com/gensim/install.html):
```bash
pip install --upgrade gensim
```

download audio samples from MagnaTagATune dataset (http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
```bash
cd ~/Download
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.001
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.002
curl http://mi.soi.city.ac.uk/datasets/magnatagatune/mp3.zip.003
```

you need to concatenate the partial archive first
```bash
cat mp3.zip.* > magna.zip
unzip magna.zip -d SHUFL_ROOT/scripts/data/magna
```

some mp3 files are broken/empty - you can find them:
```bash
cd SHUFL_ROOT/scripts/data/magna
$ du -ah SHUFL_ROOT/scripts/data/magna | grep -v "/$" | sort  | head -n 20
  0B	./6/norine_braun-now_and_zen-08-gently-117-146.mp3
  0B	./8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
  0B	./9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
```

remove these files:
```bash
rm ./6/norine_braun-now_and_zen-08-gently-117-146.mp3
rm ./8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3
rm ./9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3
```


# usage
