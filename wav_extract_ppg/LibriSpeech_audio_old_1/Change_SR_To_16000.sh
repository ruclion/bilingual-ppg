#!/usr/bin/env bash
for x in ./*.wav
  do
    b=${x##*/}
    sox $b -r 16000 -c 1 tmp_$b
    rm -rf $b
    mv tmp_$b $b
  done

