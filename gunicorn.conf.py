# gunicorn.conf.py
import os

workers = 1
worker_class = 'sync'
worker_connections = 1000
timeout = 120
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Reduce memory usage
def post_fork(server, worker):
    import gc
    gc.collect()