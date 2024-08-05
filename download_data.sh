mkdir -p ./data
mkdir -p ./data/datasets
mkdir -p ./data/weights
mkdir -p ./data/weights/yolov8
mkdir -p ./data/weights/gdrnpp

wget -O gdrnpp_ycbv_weights.zip "https://owncloud.tuwien.ac.at/index.php/s/fkCygRgrV9C7zDH/download"
unzip gdrnpp_ycbv_weights.zip
cp -r gdrnpp_ycbv_weights.pth ./data/weights/gdrnpp/output/gdrn/ycbv/gdrnpp_ycbv_weights.pth
rm -r gdrnpp_ycbv_weights.zip gdrnpp_ycbv_weights.pth

wget -O gdrnpp_ycbv_models.zip "https://owncloud.tuwien.ac.at/index.php/s/gUxThY2caSsvix2/download"
unzip gdrnpp_ycbv_models.zip
cp -r models ./data/datasets/ycbv
rm -r gdrnpp_ycbv_models.zip models

# # Ycb ichores weights and models
wget -O ycb_ichores.zip "https://owncloud.tuwien.ac.at/index.php/s/qTajYeAhCghzRl3/download"
unzip ycb_ichores.zip

unzip ycb_ichores/yolov8_weights/ycb_ichores.zip
cp -r train10/weights/last.pt ./data/weights/yolov8/ycb_ichores.pt
rm -r train10 ycb_ichores/yolov8_weights

cp -r ycb_ichores/gdrn_weight/model_final.pth ./data/weights/gdrnpp/output/gdrn/ycb_ichores/gdrnpp_ycb_ichores_weights.pth
rm -r ycb_ichores/gdrn_weight 

cp -r ycb_ichores/models ./data/datasets/ycb_ichores
rm -r ycb_ichores.zip ycb_ichores