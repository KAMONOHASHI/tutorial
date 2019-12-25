import os
import glob
from PIL import Image

files = glob.glob('/kqi/input/**/**.png', recursive=True)

dname = os.environ["DATA_NAME"]

for f in files:
    img = Image.open(f)
    img_resize = img.resize((128,128))
    fname = os.path.basename(f)
    ftitle, fext = os.path.splitext(fname)
    new_data_path = '/kqi/output/' + dname + '_' + ftitle
    os.mkdir(new_data_path)
    img_resize.save(new_data_path  + '/' + ftitle + '_resized' + fext)
