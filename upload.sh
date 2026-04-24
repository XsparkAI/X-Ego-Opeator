export CCR_HOST=ccr-29eug8s3-pub.cnc.bj.baidubce.com
export CCR_NAMESPACE=xspark
export IMAGE_NAME=xego-hand_analysis
export IMAGE_TAG=20260423-test-11
export FULL_IMAGE=$CCR_HOST/$CCR_NAMESPACE/$IMAGE_NAME:$IMAGE_TAG

sudo docker tag xego-hand_analysis:latest "$FULL_IMAGE"
sudo docker push "$FULL_IMAGE"