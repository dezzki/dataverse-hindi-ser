import os, requests, sys
print("=== DIAG2 START ===")
prox={k:v for k,v in os.environ.items() if "proxy" in k.lower() or "PROXY" in k}
print("proxy env:", prox)
for label,url in [("HF","https://huggingface.co"),("pypi","https://pypi.org/simple/")]:
    for stream in (True, False):
        try:
            r=requests.get(url, timeout=30, stream=stream)
            print("REQ", label, "stream=", stream, "->", r.status_code)
            r.close()
        except Exception as e:
            print("REQ-FAIL", label, "stream=", stream, repr(e)[:160])
try:
    from huggingface_hub import hf_hub_download
    p=hf_hub_download("dundar-gunluk/emo-small","config.json")
    print("HF_HUB_OK", p)
except Exception as e:
    print("HF_HUB_FAIL", repr(e)[:160])
print("=== DIAG2 END ===")
