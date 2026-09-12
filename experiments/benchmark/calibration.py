"""Training-data calibration of the encoder projection's single BatchNorm."""
import hashlib
import torch


@torch.no_grad()
def calibrate_projection_bn(model, batches):
    modules = [m for m in model.projector.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if len(modules) != 1 or not modules[0].track_running_stats:
        raise ValueError("Expected one encoder projection BatchNorm with running statistics")
    bn = modules[0]
    modes = {m:m.training for m in model.modules()}
    count, mean, m2 = 0, None, None
    def collect(module, inputs):
        nonlocal count, mean, m2
        x = inputs[0].detach().double()
        if x.ndim != 2 or not torch.isfinite(x).all():
            raise ValueError("Expected finite two-dimensional BatchNorm inputs")
        n = len(x)
        current_mean = x.mean(0)
        current_m2 = (x-current_mean).square().sum(0)
        if count == 0:
            count, mean, m2 = n, current_mean, current_m2
        else:
            delta = current_mean-mean
            total = count+n
            m2 = m2+current_m2+delta.square()*(count*n/total)
            mean = mean+delta*(n/total)
            count = total
    model.eval()
    hook = bn.register_forward_pre_hook(collect)
    try:
        device = next(model.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            for batch in batches:
                model.encode(dict(batch))
        if count < 2:
            raise ValueError("Need at least two calibration observations")
        variance = m2/(count-1)
        if not torch.isfinite(variance).all():
            raise ValueError("Nonfinite calibration variance")
        bn.running_mean.copy_(mean)
        bn.running_var.copy_(variance)
        digest = hashlib.sha256(bn.running_mean.cpu().numpy().tobytes()+bn.running_var.cpu().numpy().tobytes()).hexdigest()
        return {"observations":count,"statistics_sha256":digest,
                "mean_running_variance":float(bn.running_var.mean())}
    finally:
        hook.remove()
        for module, training in modes.items():
            module.training = training
