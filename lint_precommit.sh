#!/bin/bash

my_precommit() {
    repo_dir=$(git rev-parse --show-toplevel)

    autoflake --in-place -r $repo_dir/$1
    isort $repo_dir/$1 --length-sort
    black $repo_dir/$1

    git add $repo_dir/$1
}

filenames=$( git diff --name-only --cached )
for filename in $filenames; 
    do my_precommit $filename; 
done