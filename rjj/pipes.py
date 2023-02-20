import os
import time
import asyncio
from rjj.lowlevel_async import wait_readable, wait_writable

class AsyncPipe:
    def __init__(self, filename, mode='r', encoding='utf-8', blocksize=4096):
        self.filename = filename
        self.flags = os.O_NONBLOCK
        if mode == 'r':
            self.flags |= os.O_RDONLY
        elif mode == 'w':
            self.flags |= os.O_WRONLY
            self.flags |= os.O_CREAT
        elif mode in ['rw', 'wr']:
            self.flags |= os.O_RDWR
            self.flags |= os.O_CREAT
        else:
            raise ValueError(f"Unrecognized mode '{mode}'")
        self.encoding = encoding
        self.blocksize = blocksize

    def __enter__(self):
        self.fd = os.open(self.filename, self.flags)
        return self

    def __exit__(self, err_type, err_value, traceback):
        os.close(self.fd)

    async def __aiter__(self):
        """
        Yield messages asynchronously as they come in from the pipe,
        returning when the pipe is closed.
        """
        message = b''
        # wait until the pipe is opened by a writer
        await wait_readable(self.fd)
        while True:
            try:
                item = os.read(self.fd, self.blocksize)
            except BlockingIOError:
                # pipe is empty, yield message and wait for another
                if self.encoding is not None:
                    message = message.decode(self.encoding)
                yield message
                message = b''
                await wait_readable(self.fd)
            else:
                if not item:
                    # pipe is closed, return
                    break
                else:
                    message += item
        if self.encoding is not None:
            message = message.decode(self.encoding)
            yield message


    async def read(self, wait_closed=True):
        """
        Read data from the pipe.

        Parameters
        ----------
        wait_closed: Whether to wait to return until the pipe is closed.
                     If `False`, return as soon as the pipe is blocked.
        """
        content = b''
        # wait until the pipe is opened by a writer
        await wait_readable(self.fd)
        while True:
            try:
                item = os.read(self.fd, self.blocksize)
            except BlockingIOError:
                # pipe is blocked
                if wait_closed:
                    await wait_readable(self.fd)
                else:
                    break
            else:
                if not item:
                    # pipe is closed
                    break
                else:
                    content += item
        if self.encoding is not None:
            content = content.decode(self.encoding)
        return content

    async def write(self, item):
        await wait_writable(self.fd)
        if self.encoding is not None:
            item = item.encode(self.encoding)
        return os.write(self.fd, item)
