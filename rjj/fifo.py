import os
import time
import asyncio

class FIFO:
    def __init__(self, filename, encoding='utf-8', buffer_size=4096, poll_interval=0.1):
        self.filename = filename
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.poll_interval = poll_interval

    def __enter__(self):
        self.fd = os.open(self.filename, os.O_NONBLOCK)
        return self

    def __exit__(self, err_type, err_value, traceback):
        os.close(self.fd)

    def __iter__(self):
        return self.read_until_disconnect()

    def read(self, block_until_connect=True):
        if self.encoding is None:
            result = b''
        else:
            result = ''
        while True:
            try:
                contents = os.read(self.fd, self.buffer_size)
            except BlockingIOError:
                if result:
                    return result
                time.sleep(self.poll_interval)
            else:
                if contents:
                    if self.encoding is not None:
                        contents = contents.decode(self.encoding)
                    result += contents
                else:
                    if result:
                        return result
                    elif block_until_connect:
                        time.sleep(self.poll_interval)
                    else:
                        raise EOFError("Pipe is empty, no writers connected")

    def read_until_disconnect(self, block_until_connect=True):
        yield self.read(block_until_connect=block_until_connect)
        while True:
            try:
                yield self.read(block_until_connect=False)
            except EOFError:
                break

    def read_forever(self):
        while True:
            yield self.read()

class AsyncFIFO:
    def __init__(self, filename, encoding='utf-8', buffer_size=4096, poll_interval=0.1):
        self.filename = filename
        self.encoding = encoding
        self.buffer_size = buffer_size
        self.poll_interval = poll_interval

    def __enter__(self):
        self.fd = os.open(self.filename, os.O_NONBLOCK)
        return self

    def __exit__(self, err_type, err_value, traceback):
        os.close(self.fd)

    def __aiter__(self):
        return self.read_until_disconnect()

    async def read(self, yield_until_connect=True):
        if self.encoding is None:
            result = b''
        else:
            result = ''
        while True:
            try:
                contents = os.read(self.fd, self.buffer_size)
            except BlockingIOError:
                if result:
                    return result
                await asyncio.sleep(self.poll_interval)
            else:
                if contents:
                    if self.encoding is not None:
                        contents = contents.decode(self.encoding)
                    result += contents
                else:
                    if result:
                        return result
                    elif yield_until_connect:
                        await asyncio.sleep(self.poll_interval)
                    else:
                        raise EOFError("Pipe is empty, no writers connected")

    async def read_until_disconnect(self, yield_until_connect=True):
        yield await self.read(yield_until_connect=yield_until_connect)
        while True:
            try:
                yield await self.read(yield_until_connect=False)
            except EOFError:
                break

    async def read_forever(self):
        while True:
            yield await self.read()
