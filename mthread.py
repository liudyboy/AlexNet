import threading


class send_raw_data_thread(threading.Thread):
    def __init__(self, name, X, Y, destination):
        threading.Thread.__init__(self)
        self.name = name

    def run(self):
