import time
import schedule

# Holds tuples of (time_string, function, args, kwargs)
registered_tasks = []

def daily(time_str):
    """Decorator to register a function to run daily at the given time."""
    def decorator(func):
        registered_tasks.append((time_str, func, [], {}))
        return func
    return decorator

def add_daily_task(time_str, func, *args, **kwargs):
    """Programmatically register a daily task."""
    registered_tasks.append((time_str, func, args, kwargs))


def _setup():
    """Internal: schedule all registered tasks."""
    for time_str, func, args, kwargs in registered_tasks:
        schedule.every().day.at(time_str).do(func, *args, **kwargs)


def run():
    """Run the scheduler loop."""
    # Import tasks to ensure decorators are executed
    import trading_app.tasks  # noqa: F401
    _setup()
    while True:
        schedule.run_pending()
        time.sleep(1)
