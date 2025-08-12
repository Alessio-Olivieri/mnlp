import bitsandbytes as bnb
from bitsandbytes.cuda_setup.main import get_compute_capabilities
print("bnb version:", bnb.__version__)
print("compute caps:", get_compute_capabilities())