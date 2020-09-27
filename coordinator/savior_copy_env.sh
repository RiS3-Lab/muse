#!/bin/sh
PREV_ENV=$1
NEW_ENV=$2
PROG=$3

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo "$0 prev_env new_env prog"
    exit
fi

mkdir $NEW_ENV

cp -a $PREV_ENV/labelmap.csv $NEW_ENV/
cp -a $PREV_ENV/*.edge $NEW_ENV/
cp -a $PREV_ENV/*.filter $NEW_ENV/
cp -a $PREV_ENV/*.reach* $NEW_ENV/
cp -a $PREV_ENV/*.dma.bc $NEW_ENV/
cp -a $PREV_ENV/$PROG-opt.bc $NEW_ENV/
cp -a $PREV_ENV/muse-$PROG.bc $NEW_ENV/
cp -a $PREV_ENV/*.dom $NEW_ENV/
cp -a $PREV_ENV/*.direct $NEW_ENV/
cp -a $PREV_ENV/$PROG $NEW_ENV/
cp -a $PREV_ENV/afl-$PROG $NEW_ENV/
cp -a $PREV_ENV/muse-$PROG $NEW_ENV/
cp -a $PREV_ENV/config.cfg $NEW_ENV/
cp -a $PREV_ENV/in/ $NEW_ENV
