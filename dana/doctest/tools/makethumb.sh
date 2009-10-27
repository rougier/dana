#! /bin/bash
#
# Don't forge to chmod 755 makethumb.sh
#
# Example:
#
#  Images are in /tmp
#  Thumbnails need to be created in /usr/share/images/thumbs
#  
#  > cd /usr/share/images/thumbs
#  > convert_all.sh /tmp/*.jpg


NOARGS=65
if [ $# -eq 0 ]
then
    echo "Usage: `basename $0` filename" >&2
    exit 1
fi  

density="1.5"
for file in $*; do
    if [ -e "$file" ]; then
        fullname=`basename $file`
        basename=${fullname%%.[^.]*}

        echo -n Processing $basename...
        convert ${file} \
            -size 128x128                 \
            -thumbnail '128x128>'         \
            -bordercolor white  -border 6 \
            -bordercolor grey60 -border 1 \
            -background  none   -rotate 0 \
            -background  black  \( +clone -shadow 60x4+4+4 -channel A -evaluate multiply ${density} +channel \) +swap \
            -background  none   -flatten \
            -depth 8 -quality 95 ${basename}.png
       echo "done"
    fi
done
