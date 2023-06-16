import os


def set_visible_gpu(gpu_id: int):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def set_max_threads(num_max_cores: int):
    os.environ["OMP_NUM_THREADS"] = str(num_max_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_max_cores)
    os.environ["MKL_NUM_THREADS"] = str(num_max_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_max_cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_max_cores)

