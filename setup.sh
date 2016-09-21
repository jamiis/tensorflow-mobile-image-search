# exmaple usage:
#    $ ./get_caltech_256.sh ~/src/datasets
#
# arguments:
#    $1 (optional): data directory where caltech-256 will reside



### CALTECH-256 IMAGE DATASET ###
# default datasets directory
DATASETS_DIR=~/datasets/

# conditionally override DATASETS_DIR if first argument provided to script
if [ ! -z "$1" ]; then DATASETS_DIR=$1; fi

# make datasets directory if doesn't exist
mkdir -p $DATASETS_DIR
echo $DATASETS_DIR

CALTECH_ARCHIVE=256_ObjectCategories.tar
CALTECH_DIR=256_ObjectCategories
CALTECH_DIR_RENAME="caltech-256"
CALTECH_URL="http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"

# download caltech-256 dataset and unarchive
wget $CALTECH_URL
tar -xf $CALTECH_ARCHIVE
mv $CALTECH_DIR "$DATASETS_DIR/$CALTECH_DIR_RENAME"
rm $CALTECH_ARCHIVE



### DOWNLOAD RETRAIN.PY ###
wget https://github.com/tensorflow/tensorflow/raw/bae5a573207bdfdc941878c00316dc9c24c3bcb3/tensorflow/examples/image_retraining/retrain.py
