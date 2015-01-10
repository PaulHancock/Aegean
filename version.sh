tag=`git describe --tag --long`
sed -i .bak "s/^version = '.*'/version = '${tag}'/g" aegean.py
