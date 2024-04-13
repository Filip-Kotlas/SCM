#!/bin/bash
# task 3 subtask 6
# Filip Kotlas

if [ ! -e "gif.zip" ]; then
    wget http://imsc.uni-graz.at/haasegu/Lectures/SciComp/SS20/gif.zip
    unzip gif.zip

    mkdir gif_new
    cd gif
    for NAME in *.gif; do
        mm=${NAME:0:2}
        dd=${NAME:2:2}
        yy=${NAME:4:2}

        yyyy="20$yy"

        NEW_NAME="${yyyy}_${mm}_${dd}.gif"
        mv "$NAME" "$NEW_NAME"
        ADRESS_WITH_GIF="../gif_new/$NEW_NAME"
        convert $NEW_NAME -resize 50% "${ADRESS_WITH_GIF::-3}png"
    done
else
    echo "Folder with pictures is already downloaded and converted."
fi

