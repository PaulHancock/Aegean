# use the following format to change the major version number
# git tag -a v2.5 -m 'Version 2.5'
# git push --tags
tag=`git describe --tag --long`
sed -i .bak "s/^version = '.*'/version = '${tag}'/g" aegean.py
