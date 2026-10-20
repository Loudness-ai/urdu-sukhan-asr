#!/usr/bin/env bash
# MUSAN (OpenSLR 17, CC BY 4.0): music, noise and speech-babble recordings, used as no-speech examples.
cd "$(dirname "$0")/.."
mkdir -p data/musan
for i in 1 2 3 4 5; do
  curl -sSL -C - -o data/musan/musan.tar.gz https://openslr.elda.org/resources/17/musan.tar.gz && break
  sleep 30
done
tar xzf data/musan/musan.tar.gz -C data/musan && rm -f data/musan/musan.tar.gz && echo "musan ok"
