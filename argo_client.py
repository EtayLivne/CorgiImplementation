import os
import uuid
from pathlib import Path

import yaml
from hera.task import Task
from hera.workflow import Workflow
from hera.workflow_service import WorkflowService

class TokenMissing(BaseException):
    pass


class ArgoClient:
    ARGO_TOKEN_PATH_INSIDE_POD = "/run/secrets/kubernetes.io/serviceaccount/token"
    ARGO_SERVER = "http://zorro-api.angie.mobileye.com"
    ZORRO_CONFIG_PATH = Path.home() / ".zorro.yaml"

    def __init__(self):
        self.zorro_config = self._get_zorro_config()
        self.token = self._get_token()
        self.workflow_service = WorkflowService(
            host=self.ARGO_SERVER, verify_ssl=False, token=self.token
        )
        self._workflow_name_set = False
        self._current_user = None

    def _get_zorro_config(self):
        try:
            with self.ZORRO_CONFIG_PATH.open("r") as fp:
                zorro_config = yaml.safe_load(fp)
        except FileNotFoundError:
            raise TokenMissing(
                f"Could not find zorro config at {self.ZORRO_CONFIG_PATH}, please run `zorro auth` and try again"
            )
        return zorro_config

    def _get_token(self):
        if self.in_argo_environment:
            token = self.get_running_pod_auth_token()
        else:
            token = self.zorro_config["ZORRO_TOKEN"]
        return token

    def get_running_pod_auth_token(self):
        with open(self.ARGO_TOKEN_PATH_INSIDE_POD, "r") as file:
            argo_token = file.read()
        return argo_token

    @property
    def in_argo_environment(self):
        return "ARGO_NODE_ID" in os.environ

    def workflow_name(self, name):
        if self._workflow_name_set:
            return self._workflow_name
        id = str(uuid.uuid4()).split("-")[0]
        self._workflow_name = f"{name}-{self.user}-{id}"
        self._workflow_name_set = True
        return self._workflow_name

    @property
    def user(self):
        if self._current_user:
            return self._current_user
        self._current_user = os.environ.get("USER", "")
        return self._current_user

    def status(self, name):
        return self.workflow_service.get_workflow_status(name)

    def _get_datacenter_value(self) -> str:
        return "mobileye-on-prem"

    def submit_workflow(self, name, *tasks: Task):
        service_account = None
        data_center = self._get_datacenter_value()
        workflow = Workflow(
            name=self.workflow_name(name),
            service=self.workflow_service,
            labels={"USER": self.user, "extendedRun": "true", "dataCenter": data_center},
            service_account_name=service_account,
        )
        workflow.add_tasks(*tasks)
        workflow.create()
        return workflow.name