#!/bin/bash
# python3 -m tests.{metric,image,polynomial} -h
# brew install expect
cd $(dirname $0)/..
# read -r -p "Enter alternative module name: " ALT
ALT=$(keyring get mlop alternative)
if [ "$1" == "metric" ] || [ "$1" == "image" ]; then
    unbuffer python -m tests.$1 $2 $3 $4 | sed "s/$ALT/alt/g"
else
    echo "$0 [metric|image] [r|m|w] [if|store] [e|d]"
fi
