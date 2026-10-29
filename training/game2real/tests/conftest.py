import torch


def pytest_sessionstart(session):
    # Tiny correctness tests are slower with a many-core BLAS thread pool.
    torch.set_num_threads(2)
