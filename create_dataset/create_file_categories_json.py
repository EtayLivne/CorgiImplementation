import subprocess
import json


CATEGORIES = [
    "academic",
    "code",
    "judicial",
    "medical",
    "news",
    "poetry",
    "quotes",
    "recipes",
    "reddit",
    "reviews",
    "tweets"
]

vast5 = "s5cmd --endpoint-url http://vast1.me-corp.lan --profile op".split(" ")
json_data = dict()
for c in CATEGORIES:
    command =  f"s5cmd --endpoint-url http://vast1.me-corp.lan --profile op ls s3://mobileye-team-angie/users/etay/nltk_data/{c}/ | wc -l"
    # json_data[c] = subprocess.run(vast5 + ["ls", f"s3://mobileye-team-angie/users/etay/nltk_data/{c}/", "|", "wc", "-l"], capture_output=True, text=True, shell=True)
    json_data[c] = int((subprocess.run(command, shell=True, capture_output=True, text=True).stdout).rstrip())

with open("/homes/etayl/code/bert/file_categories.json", "w") as handler:
    json.dump(json_data, handler, indent=4)