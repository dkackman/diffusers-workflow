from .qr_code import get_qrcode_image

def dispatch_task(task_name, kwargs):
    task_name = task_name.lower()

    if task_name == "qr_code":
        return [get_qrcode_image(**kwargs)]
    
    raise ValueError(f"Unknown task {task_name}")