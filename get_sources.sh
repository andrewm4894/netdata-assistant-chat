# Cron this script to run daily

# clone repos
cd tmp
git clone https://github.com/netdata/learn.git
git clone https://github.com/netdata/blog.git

# copy to data folder
cd ../data/
cp -rf ../tmp/learn/docs/* ./docs/
cp -rf ../tmp/blog/blog/* ./blog/

