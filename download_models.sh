#!/bin/bash
VERSION=0.9.3

mkdir models
cd models
wget "https://github.com/mozilla/DeepSpeech/releases/download/v$VERSION/deepspeech-$VERSION-models.pbmm"
wget "https://github.com/mozilla/DeepSpeech/releases/download/v$VERSION/deepspeech-$VERSION-models.scorer"
cd ..

