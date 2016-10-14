#!/bin/bash

# how2run: ./git_pull_all.sh .
# how2run: ./git_pull_all.sh /path2/gitrepos
if [ $# -lt 1 ]; then
echo Usage: $0  /path2_gitrepos
exit 1
fi

git_root=$1
cd $git_root || exit 1

echo "going to pull every git repo in $git_root"

for d in `ls -d *`; do
    echo Doing dir $d
    cd $d && /usr/bin/git pull 
    cd $git_root

done

