
class ModuleEventListener():
    def on_batch(func, *args, **kwargs):
        def on_batch_(*args, **kwargs):
            return func(*args, **kwargs)

        return on_batch_

    def on_epoch(func, *args, **kwargs):
        def on_epoch_(*args, **kwargs):
            return func(*args, **kwargs)
        return on_epoch_
