# From the public folder https://disk.yandex.ru/d/VDWjzzpifhpBTw
# download the file medical_mask_overlay/CelebA-HQ-img-256-256.zip
# into the project root (the folder where this script is located)
# and run this script to setup the environment

mkdir mask_overlay_datasets
mkdir mask_overlay_datasets/CelebA-HQ-img-256-256
mkdir mask_overlay_datasets/CelebA-HQ-img-256-256-labels
mkdir mask_overlay_datasets/CelebA-HQ-img-256-256-masked
mkdir mask_overlay_datasets/CelebA-HQ-img-256-256-merged

unzip CelebA-HQ-img-256-256.zip -d mask_overlay_datasets

