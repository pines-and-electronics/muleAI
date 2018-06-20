from parts.base import BasePart

class MockMode(BasePart):
    input_keys = ()
    output_keys = ('mode',)


    def __init__(self):
        self.mode = {'recording': False}
        self.counter = 1

    def start(self):
        pass

    def transform(self, state):
        state['mode'] = self.mode
        if self.counter % 20 == 0:
            self.mode['recording'] = not self.mode['recording']
        self.counter += 1

    def stop(self):
        pass



