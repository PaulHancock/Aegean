# use the following format to change the major version number
# git tag -a v2.5 -m 'Version 2.5'
# git push --tags
tag=`git describe --tag --long`
files="aegean.py BANE.py MIMAS.py SR6.py"
for f in ${files}
do
sed -i.bak "0,/__version__/{s/__version__ = '.*'/__version__ = '${tag}'/}" ${f}
date=`git log -n 1 --pretty="%cd" ${f}`
shortdate=`python d2v.py ${date}`
sed -i.bak "0,/__date__/{s/__date__ = '.*'/__date__ = '${shortdate}'/}" ${f}
echo git add ${f}
done
echo git commit -m \"update version/date information\"

