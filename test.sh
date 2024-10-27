#!/usr/bin/sh
for file in tests/*; do
    category=$(echo $file | sed "s;.*/\(.*\)\..*;\1;")
    echo "Ожидаем: $category; получаем: $(car-recog $file custom_resnet18.pth -np normalize_params.json)"
done
