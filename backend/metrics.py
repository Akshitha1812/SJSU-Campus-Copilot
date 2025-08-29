import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware


REQUESTS = Counter("requests_total", "Total requests", ["route"])
ERRORS = Counter("errors_total", "Errors", ["route"])
LATENCY = Histogram("latency_ms", "Latency (ms)", ["stage"])
INFLIGHT = Gauge("inflight_requests", "In-flight requests")


def with_metrics(route_name: str):
    def deco(fn):
        @wraps(fn)
        def wrapped(*a, **kw):
            INFLIGHT.inc()
            REQUESTS.labels(route=route_name).inc()
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            except Exception:
                ERRORS.labels(route=route_name).inc()
                raise
            finally:
                LATENCY.labels(stage=route_name).observe(
                    (time.perf_counter()-t0)*1000)
            INFLIGHT.dec()
        return wrapped
    return deco


def mount_metrics(app):
    app.wsgi_app = DispatcherMiddleware(
        app.wsgi_app, {"/metrics": make_wsgi_app()})
    return app
