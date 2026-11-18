"""Download the VLM weights (pinned revision) into the Hugging Face cache."""
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17')
print('DONE', p)
