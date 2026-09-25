import subprocess, torch, os
print("=== GPU PROBE ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
    try:
        x=torch.randn(1000,1000,device="cuda").sum().item()
        print("cuda op OK:", round(x,2))
    except Exception as e:
        print("cuda op FAIL:", repr(e)[:120])
else:
    try:
        out=subprocess.run(["nvidia-smi"],capture_output=True,text=True,timeout=30)
        print("nvidia-smi rc", out.returncode, out.stdout[:150])
    except Exception as e:
        print("nvidia-smi FAIL", repr(e)[:120])
print("=== GPU PROBE END ===")
