import asyncio
import sniffio

def resolve_future(future):
    """
    Callback function used to signal to an awaiting async function
    that a resource is ready, without returning a value.
    """
    if not future.cancelled():
        future.set_result(None)

async def wait_readable(obj):
    """
    An equivalent of trio.lowlevel.wait_readable(), i.e., an async
    funtion that will yield to the event loop until the operating system 
    reports that the given file descriptor or file-like object is available
    for reading.
    
    The hard parts are delegated to the asyncio event loop, using the
    `asyncio.add_reader()` function. The main purpose of this function is
    to adapt that callback-based API to use async/await syntax.
    """
    library = sniffio.current_async_library()
    if library == "trio":
        return await trio.lowlevel.wait_readable(obj)
    elif library == "asyncio":
        pass
    else:
        raise RuntimeError(f"'{library}' is not a supported async library")

    if hasattr(obj, 'fileno'):
        fd = obj.fileno()
    else:
        fd = obj
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    loop.add_reader(fd, resolve_future, future)
    try:
        await future
    finally:
        loop.remove_reader(fd)

async def wait_writable(obj):
    """
    An equivalent of trio.lowlevel.wait_writable(), i.e., an async
    funtion that will yield to the event loop until the operating system 
    reports that the given file descriptor or file-like object is available
    for writing.
    
    The hard parts are delegated to the asyncio event loop, using the
    `asyncio.add_writer()` function. The main purpose of this function is
    to adapt that callback-based API to use async/await syntax.
    """
    library = sniffio.current_async_library()
    if library == "trio":
        return await trio.lowlevel.wait_writable(obj)
    elif library == "asyncio":
        pass
    else:
        raise RuntimeError(f"'{library}' is not a supported async library")

    if hasattr(obj, 'fileno'):
        fd = obj.fileno()
    else:
        fd = obj
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    loop.add_writer(fd, resolve_future, future)
    try:
        await future
    finally:
        loop.remove_writer(fd)
