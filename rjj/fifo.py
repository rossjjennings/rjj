import os
import time
import asyncio
try:
    import trio
except ImportError:
    have_trio = False
else:
    have_trio = True

class AsyncPipe:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self

    def __exit__(self, err_type, err_value, traceback):
        self.file.close()

    def _read(self, future):
        item = self.file.read()
        if not future.cancelled():
            future.set_result(item)

    def _write(self, item, future):
        self.file.write(item)
        if not future.cancelled():
            future.set_result(None)

    async def read(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        loop.add_reader(self.file.fileno(), self._read, future)
        try:
            item = await future
        finally:
            loop.remove_reader(self.file.fileno())
        return item

    async def write(self, item):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        loop.add_writer(self.file.fileno(), self._write, item, future)
        try:
            item = await future
        finally:
            loop.remove_writer(self.file.fileno())

if have_trio:
    class TrioPipe:
        def __init__(self, filename, mode='r'):
            self.filename = filename
            self.mode = mode

        def __enter__(self):
            self.file = open(self.filename, self.mode)
            return self

        def __exit__(self, err_type, err_value, traceback):
            self.file.close()

        async def read(self):
            await trio.lowlevel.wait_readable(self.file)
            return self.file.read()

        async def write(self, item):
            await trio.lowlevel.wait_writable(self.file)
            return self.file.write(item)
