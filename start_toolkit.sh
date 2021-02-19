# For documentation, please refer to "doc/tasks.md"

dataDir="/home/ubuntu/data"

methodName=mrcnn50_nm
scale=1.0
# "nm" is short for "no mask"

python -m sap_toolkit.server \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--overwrite \
	--out-dir "$dataDir/Exp/Argoverse-HD/output/${methodName}_s${scale}_evalserver_forecast/val" \
	--eval-config "./config.json"