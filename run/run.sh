# RUN_ID=$(python -c 'import string; import random; print("".join(random.choices(string.ascii_lowercase + string.digits, k=7)))')
# mkdir -p /tmp/train_gpt/$RUN_ID
# # cp -r ~/code/bert /tmp/train_gpt/$RUN_ID
# rsync -av --exclude 'create_dataset' --exclude 'test_checkpoints' --exclude "fake_data" ~/code/bert /tmp/train_gpt/$RUN_ID


# tar -czf /tmp/puma.tar.gz -C /tmp/puma/$RUN_ID .
# echo "uploading to s3"
# s5cmd --endpoint-url http://vast1.me-corp.lan --numworkers 4 cp /tmp/puma.tar.gz "$aws_dir"code/puma.tar.gz


docker build . -f docker/Dockerfile -t  artifactory.sddc.mobileye.com/dl-algo-docker-release/pqdsbm:corgipt
docker push artifactory.sddc.mobileye.com/dl-algo-docker-release/pqdsbm:corgipt
python benchmark_workflow.py