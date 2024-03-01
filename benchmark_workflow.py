from argo_client import ArgoClient

from hera.task import Task
from hera.resources import Resources
from hera.env import SecretEnv, Env
from hera.volumes import EmptyDirVolume
from hera.toleration import Toleration

IMAGE = "artifactory.sddc.mobileye.com/dl-algo-docker-release/pqdsbm:corgipt"

def get_vast_env():
    return [
    SecretEnv(name="AWS_ACCESS_KEY_ID", secret_name="op-s3-storage", secret_key="accessKey"),
    SecretEnv(
        name="AWS_SECRET_ACCESS_KEY",
        secret_name="op-s3-storage",
        secret_key="secretKey",
    ),
    Env(name="S3_USE_HTTPS", value="0"),
    Env(name="S3_VERIFY_SSL", value="0"),
    SecretEnv(name="S3_ENDPOINT", secret_name="op-s3-storage", secret_key="endpointurl"),
]

def get_volumes():
    volumes = [EmptyDirVolume(size="100Gi")]
    return volumes

def trigger_workflow():
    argo_client = ArgoClient()
    resources = Resources(cpu_limit=32, cpu_request=32, memory_limit="256Gi", memory_request="256Gi", gpus=4)
    tolerations = [Toleration(key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")]
    node_selectors = {"gpu": "nvidia-a100"}
    tasks = [
        Task(
            name="read-data",
            command=["python", "train_gpt.py"],
            image=IMAGE,
            resources=resources,
            env=get_vast_env(),
            volumes=get_volumes(),
            tolerations=tolerations,
            node_selectors=node_selectors
        )
    ]
    argo_client.submit_workflow("corgi-gpt-", *tasks)

if __name__ == "__main__":
    trigger_workflow()

