import json
from typing import Dict, Optional
import boto3

def configure_boto3(region: str, access: str, secret: str, token: Optional[str]=None):
    s = boto3.Session(aws_access_key_id=access, aws_secret_access_key=secret, aws_session_token=token, region_name=region)
    ident = s.client("sts").get_caller_identity()
    return s, ident["Account"], ident["Arn"]

class ClaudeBedrock:
    def __init__(self, boto_sess, model: str, region: str="us-east-1"):
        self.client = boto_sess.client("bedrock-runtime", region_name=region)
        self.model = model

    def complete(self, system: str, user: str, max_tokens=6000, temp=0.2) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
            "max_tokens": max_tokens, "temperature": temp
        }
        resp = self.client.invoke_model(modelId=self.model, body=json.dumps(body))
        payload = json.loads(resp["body"].read())
        parts = payload.get("content", [])
        return "".join([p.get("text", "") for p in parts if p.get("type") == "text"])