from runner import Runner
from threading import Thread
from logging import getLogger, basicConfig, RootLogger, INFO


def runner_thread(logger: type[RootLogger]):
    runner = Runner(logger)
    runner.initialize_capture()
    runner.run()
    runner.stop_capture()
    print(f"Total events: {len(runner.events)}")


basicConfig(filename="out.log", encoding="utf-8", level=INFO)

logger = getLogger()

t = Thread(target=runner_thread, args=(logger,))
t.start()
