import os
import json
from io import BytesIO
import base64
from PIL import Image
import socket

# WE ARE NOT USING THIS APPROACH
hostname= str(socket.gethostname())
def get_all_imagefiles(source, ext):
    img_paths = []
    for roots, dir, files in os.walk(source):
        for file in files:
            if file.endswith(ext):
                file_abs_path = os.path.join(roots, file)

                img_paths.append(file_abs_path)
    return img_paths


def get_bin_string_from_image(filepath):
    with open(filepath, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_string = encoded_bytes.decode('ascii')
        #print("TT", type(encoded_string))
        return encoded_string


def string_to_bytes(encoded_string):
    recovered_bytes = encoded_string.encode("ascii")
    return recovered_bytes

def convert_mongo_string_to_img(mongo_string):
    # RECONSTRUCTING ENCODED STRING
    recovered = mongo_string.encode("ascii")

    im = Image.open(BytesIO(base64.b64decode(recovered)))
    #im.save('data/image1.tif', 'TIFF')
    return im


def read_all_files(img_paths, album_name, op_path="/s/"+hostname+"/a/nobackup/galileo/sapmitra/"):

    total = len(img_paths)

    op_file = os.path.join(op_path , album_name + "-" + hostname+ ".json")

    if os.path.exists(op_file):
        print("REMOVED OLD DUMP")
        os.remove(op_file)

    #op_file = "/s/lattice-177/a/nobackup/galileo/sapmitra/" + album_name + str(file_num) + ".json"
    for i in range(0, total):
        image_document = {}

        filepath = img_paths[i]
        tokens = filepath.split('/')
        ln = len(tokens)
        filename = tokens[ln-1]
        # QUAD HASH OF THE ENTIRE IMAGE
        image_hash = tokens[ln - 3]

        img_bin_str = get_bin_string_from_image(filepath)
        #print("TYPE:", type(img_bin_str))
        image_document['quad'] = image_hash
        image_document['data'] = img_bin_str
        image_document['name_str'] = filename

        if i%20 == 0:
            print(image_hash,str(i+1)+"/"+str(total))

        with open(op_file, 'a') as outfile:
            json.dump(image_document, outfile)


if __name__ == '__main__':
    album_name = "co-3month"
    all_files = get_all_imagefiles("/s/"+hostname+"/a/nobackup/galileo/stip-images/"+album_name+"/", "-3.tif")
    print("TOTAL TIFS:",len(all_files))
    read_all_files(all_files,album_name, "/s/"+hostname+"/a/nobackup/galileo/sapmitra/")

