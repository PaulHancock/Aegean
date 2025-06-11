#! /usr/bin/env bash

if [[ ${#} -eq 0 ]]; then
   version='main';
else
    version=$1;
fi

echo "Building with branch/checkout: ${version}"
# build the container and update tag
docker build --build-arg gitargs="@${version}" . -f Dockerfile -t "paulhancock/aegean:${version}" && \
echo "Run: docker push paulhancock/aegean:${version}"
