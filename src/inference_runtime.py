"""
Choose where YOLO runs, and whether that run uses FP16 or cuDNN auto-tuning.

Both scripts import this so --device, --half, and --cudnn-benchmark mean the
same thing. TensorRT export stays in detect_stream.py: an .engine file can
run inference, and it cannot train.

FP16 stores each tensor value in 16 bits instead of 32. On an RTX or GTX 16
series GPU (and newer) that cuts the memory traffic roughly in half and
raises throughput. The CPU build of PyTorch used by this project has no
FP16 inference path, so a CPU run stays FP32.

cuDNN benchmark is a different switch. For a fixed image size it times the
available convolution algorithms once, then keeps the fastest one. The first
frames are slower. Later frames reuse that choice. It does nothing on CPU.
"""

import torch


def resolve_device(requested):
    """
    Turn --device into a device string Ultralytics accepts.

    "auto" uses cuda:0 when this machine has an NVIDIA GPU, and cpu when it
    does not. A numbered GPU that is not plugged in also falls back to cpu,
    so a script copied from a GPU machine still starts here.
    """
    text = str(requested or "auto").strip()
    lowered = text.lower()
    cuda = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda else 0

    if lowered in {"auto", ""}:
        if cuda:
            print(f"[Device] auto -> cuda:0 ({torch.cuda.get_device_name(0)})")
            return "0"
        print("[Device] auto -> cpu. CUDA is not available on this machine.")
        return "cpu"

    if lowered == "cpu":
        print("[Device] Using cpu.")
        return "cpu"

    index_text = lowered[5:] if lowered.startswith("cuda:") else lowered
    if index_text.isdigit():
        index = int(index_text)
        if not cuda or index >= count:
            print(
                f"[Device] cuda:{index} is not available "
                f"({count} CUDA device(s)). Falling back to cpu."
            )
            return "cpu"
        print(f"[Device] Using cuda:{index} ({torch.cuda.get_device_name(index)}).")
        return str(index)

    if not cuda:
        print(f"[Device] '{text}' needs CUDA, which is not available. Falling back to cpu.")
        return "cpu"
    print(f"[Device] Using {text}.")
    return text


def resolve_half(requested, device):
    """
    FP16 only when the user asked for it and the chosen device is a GPU.

    On CPU the flag is ignored and the caller keeps FP32. That is the clean
    fallback: the same command still runs, it just does not pretend to use
    half precision.
    """
    if not requested:
        return False
    if str(device).lower() == "cpu":
        print("[Half] FP16 inference needs an NVIDIA GPU. This run stays FP32 on cpu.")
        return False
    print("[Half] FP16 inference enabled. Tensor math uses 16-bit floats.")
    return True


def enable_cudnn_benchmark(requested, device):
    """
    Let cuDNN pick the fastest convolution for this image size.

    The choice is valid while every frame uses the same imgsz, which is how
    both scripts call YOLO. A CPU device leaves the flag off.
    """
    if not requested:
        return False
    if str(device).lower() == "cpu" or not torch.cuda.is_available():
        print("[cuDNN] Benchmark auto-tuning needs a CUDA GPU. Leaving it off.")
        return False
    torch.backends.cudnn.benchmark = True
    print(
        "[cuDNN] benchmark=True. The first frames time the conv algorithms "
        "for this imgsz and keep the fastest."
    )
    return True


def configure_runtime(device, half, cudnn_benchmark):
    """Resolve all three flags. Returns (device, use_half, cudnn_on)."""
    chosen = resolve_device(device)
    use_half = resolve_half(half, chosen)
    cudnn_on = enable_cudnn_benchmark(cudnn_benchmark, chosen)
    return chosen, use_half, cudnn_on
