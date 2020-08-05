def scale_tensors(*tensors, scale_factor=1):
    return [t * scale_factor for t in tensors]
