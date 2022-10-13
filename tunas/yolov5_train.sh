
models=('yolov5n' 'yolov5s' 'yolov5s-T7'  'yolov5m' 'yolov5m-T7' 'yolov5l' 'yolov5x')
device=${2:-0}
batch_size=16
if [[ " ${models[@]} " =~ " $1 " ]]; then
    cd yolov5
    python train.py --weights  ../../data/weights/${1:0:7}.pt --cfg ../configs/yolo/models/$1.yaml --data ../configs/yolo/data/yolov5_tunas_data.yaml  --hyp ../configs/yolo/hyps/hyp_$1.yaml --batch-size $batch_size  --project ../../data/runs/train  --name $1  --epochs 100 --device ${device}  --seed 2022 --exist-ok --multi-scale --patience 20
else
    echo "Please provide correct model name in ${models[@]}"
fi


