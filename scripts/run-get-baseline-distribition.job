#!/bin/bash
  
D=$1
P=$2

# not using /dev/shm due to systemd "RemoveIPC" issue
# https://askubuntu.com/questions/884127/16-04-lts-and-dev-shm-files-disappearing
# https://superuser.com/questions/1117764/why-are-the-contents-of-dev-shm-is-being-removed-automatically
# https://stackoverflow.com/questions/58911494/why-may-dev-shm-folder-be-periodically-cleaned-in-ubuntu-18-04
T=$(mktemp -d /tmp/${USER}-${D}-${P}-XXXXX)

# folder check and exit trap adapted from
# https://stackoverflow.com/questions/4632028/how-to-create-a-temporary-directory

# check if tmp dir was created
if [[ ! "$T" || ! -d "$T" ]]; then
    echo "Could not create temp dir"
    exit 1
fi

# Make sure it gets removed even if the script exits abnormally.
trap "exit 1"      HUP INT PIPE QUIT TERM
trap 'rm -rf "$T"' EXIT

source $HOME/tri-training/mtb-tri-training/config/locations.sh

cd $HOME/tri-training/mtb-tri-training/workdirs

echo $(hostname) $(date) >> distr-${D}-${P}.start

find | $HOME/tri-training/mtb-tri-training/scripts/get-baseline-distribution.py --distribution $D --part $P --tmp-dir $T \
    2> distr-${D}-${P}-stderr.txt  \
    >  distr-${D}-${P}-stdout.txt

touch distr-${D}-${P}.end

rm -rf $T

