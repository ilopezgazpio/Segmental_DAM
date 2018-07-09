#!/bin/bash

path=/home/..../SNLI/snli_1.0
glove=/home/..../glove


python process-SNLI.py --data_folder ${path} --out_folder ${path}

for name in glove.6B.300d glove.42B.300d glove.840B.300d
do
    python preprocess.py \
	--srcfile ${path}/src-train.txt \
	--targetfile ${path}/targ-train.txt \
	--labelfile ${path}/label-train.txt \
	--srcvalfile ${path}/src-dev.txt \
	--targetvalfile ${path}/targ-dev.txt \
	--labelvalfile ${path}/label-dev.txt \
	--srctestfile ${path}/src-test.txt \
	--targettestfile ${path}/targ-test.txt \
	--labeltestfile ${path}/label-test.txt \
	--outputfile ${path}/$name \
	--glove ${glove}/${name}.txt
done

for name in glove.6B.300d glove.42B.300d glove.840B.300d
do
    python get_pretrain_vecs.py \
	--glove ${glove}/${name}.txt \
	--outputfile ${path}/${name}.data.hdf5 \
	--dictionary ${path}/${name}.word.dict
done

