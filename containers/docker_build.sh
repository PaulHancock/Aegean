#! /usr/bin/env bash

function usage {
        echo "Usage: $(basename $0) [git branch name or commit hash]" 2>&1
        exit 1
}

if [[ ${#} -eq 0 ]]; then
   version='main';
else
    version=$1;
fi

echo "Building with branch/checkout: ${version}"
# build the container and update tag
docker build --build-arg gitargs="@${version}" . -f Dockerfile -t "paulhancock/aegean:${version}" && \
echo "Run: docker push paulhancock/aegean:${version}"