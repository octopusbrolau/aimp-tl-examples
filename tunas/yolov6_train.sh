cd YOLOv6

models=('yolov6t' 'yolov6t' 'yolov6s'  'yolov6m' 'yolov6l' )
device=${2:-0}
batch_size=16
if [[ " ${models[@]} " =~ " $1 " ]]; then
    python tools/train.py --weights  ../../data/weights/${1:0:7}.pt  --conf-file ../configs/yolo/models/$1.py --data-path ../configs/yolo/data/yolov6_tunas_data.yaml --batch-size $batch_size  --output-dir ../../data/runs/train/  --name $1  --eval-interval 1  --epochs 200 --device ${device}   --patience 20  --exist_ok 
else
    echo "Please provide correct model name in ${models[@]}"
fi



