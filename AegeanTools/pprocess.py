#!/usr/bin/env python

"""
A simple parallel processing API for Python, inspired somewhat by the thread
module, slightly less by pypar, and slightly less still by pypvm.


Copyright (C) 2005, 2006, 2007, 2008, 2009, 2013, 2014,
              2016 Paul Boddie <paul@boddie.org.uk>
Copyright (C) 2013 Yaroslav Halchenko <debian@onerussian.com>

Adapted to work with python 3.7 by Paul Hancock 2019
see <http://www.boddie.org.uk/python/pprocess.html> for the original version

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = "0.5.4-aegean"

import os
import sys
import select
import socket
import platform
import errno
from time import sleep
from warnings import warn
import pickle
import logging

logging.basicConfig(format="%(process)d:%(module)s:%(levelname)s %(message)s")
log = logging.getLogger("pprocess")
log.setLevel(logging.WARNING)

# Special values.
report_all_errs = False

class Undefined: pass

# Communications.

class AcknowledgementError(Exception):
    pass

class Channel:

    "A communications channel."

    def __init__(self, pid, read_pipe, write_pipe):

        """
        Initialise the channel with a process identifier 'pid', a 'read_pipe'
        from which messages will be received, and a 'write_pipe' into which
        messages will be sent.
        """

        self.pid = pid
        self.read_pipe = read_pipe
        self.write_pipe = write_pipe

    def __del__(self):

        # Since signals don't work well with I/O, we close pipes and wait for
        # created processes upon finalisation.

        self.close()

    def close(self):

        "Explicitly close the channel."

        if self.read_pipe is not None:
            self.read_pipe.close()
            self.read_pipe = None
        if self.write_pipe is not None:
            self.write_pipe.close()
            self.write_pipe = None
        #self.wait(os.WNOHANG)

    def wait(self, options=0):

        "Wait for the created process, if any, to exit."

        if self.pid != 0:
            try:
                os.waitpid(self.pid, options)
            except OSError:
                pass

    def _send(self, obj):

        "Send the given object 'obj' through the channel."

        pickle.dump(obj, self.write_pipe)
        self.write_pipe.flush()

    def send(self, obj):

        """
        Send the given object 'obj' through the channel. Then wait for an
        acknowledgement. (The acknowledgement makes the caller wait, thus
        preventing processes from exiting and disrupting the communications
        channel and losing data.)
        """

        self._send(obj)
        if self._receive() != "OK":
            raise AcknowledgementError(obj)

    def _receive(self):

        "Receive an object through the channel, returning the object."

        obj = pickle.load(self.read_pipe)
        if isinstance(obj, Exception):
            raise obj
        else:
            return obj

    def receive(self):

        """
        Receive an object through the channel, returning the object. Send an
        acknowledgement of receipt. (The acknowledgement makes the sender wait,
        thus preventing processes from exiting and disrupting the communications
        channel and losing data.)
        """

        try:
            obj = self._receive()
            return obj
        finally:
            self._send("OK")

class PersistentChannel(Channel):

    """
    A persistent communications channel which can handle peer disconnection,
    acting as a server, meaning that this channel is associated with a specific
    address which can be contacted by other processes.
    """

    def __init__(self, pid, endpoint, address):
        Channel.__init__(self, pid, None, None)
        self.endpoint = endpoint
        self.address = address
        self.poller = select.poll()

        # Listen for connections before this process is interested in
        # communicating. It is not desirable to wait for connections at this
        # point because this will block the process.

        self.endpoint.listen(1)

    def close(self):

        "Close the persistent channel and remove the socket file."

        Channel.close(self)
        try:
            os.unlink(self.address)
        except OSError:
            pass

    def _ensure_pipes(self):

        "Ensure that the channel is capable of communicating."

        if self.read_pipe is None or self.write_pipe is None:

            # Accept any incoming connections.

            endpoint, address = self.endpoint.accept()
            self.read_pipe = endpoint.makefile("rb", 0)
            self.write_pipe = endpoint.makefile("wb", 0)

            # Monitor the write pipe for error conditions.

            fileno = self.write_pipe.fileno()
            self.poller.register(fileno, select.POLLOUT | select.POLLHUP | select.POLLNVAL | select.POLLERR)

    def _reset_pipes(self):

        "Discard the existing connection."

        fileno = self.write_pipe.fileno()
        self.poller.unregister(fileno)
        self.read_pipe = None
        self.write_pipe = None
        self.endpoint.listen(1)

    def _ensure_communication(self, timeout=None):

        "Ensure that sending and receiving are possible."

        while 1:
            self._ensure_pipes()
            fileno = self.write_pipe.fileno()
            fds = self.poller.poll(timeout)
            for fd, status in fds:
                if fd != fileno:
                    continue
                if status & (select.POLLHUP | select.POLLNVAL | select.POLLERR):

                    # Broken connection: discard it and start all over again.

                    self._reset_pipes()
                    break
            else:
                return

    def _send(self, obj):

        "Send the given object 'obj' through the channel."

        self._ensure_communication()
        Channel._send(self, obj)

    def _receive(self):

        "Receive an object through the channel, returning the object."

        self._ensure_communication()
        return Channel._receive(self)

# Management of processes and communications.

class Exchange:

    """
    A communications exchange that can be used to detect channels which are
    ready to communicate. Subclasses of this class can define the 'store_data'
    method in order to enable the 'add_wait', 'wait' and 'finish' methods.

    Once exchanges are populated with active channels, use of the principal
    methods of the exchange typically cause the 'store' method to be invoked,
    resulting in the processing of any incoming data.
    """

    def __init__(self, channels=None, limit=None, reuse=0, continuous=0, autoclose=1):

        """
        Initialise the exchange with an optional list of 'channels'.

        If the optional 'limit' is specified, restrictions on the addition of
        new channels can be enforced and observed through the 'add_wait', 'wait'
        and 'finish' methods. To make use of these methods, create a subclass of
        this class and define a working 'store_data' method.

        If the optional 'reuse' parameter is set to a true value, channels and
        processes will be reused for waiting computations, but the callable will
        be invoked for each computation.

        If the optional 'continuous' parameter is set to a true value, channels
        and processes will be retained after receiving data sent from such
        processes, since it will be assumed that they will communicate more
        data.

        If the optional 'autoclose' parameter is set to a false value, channels
        will not be closed automatically when they are removed from the exchange
        - by default they are closed when removed.
        """

        self.limit = limit
        self.reuse = reuse
        self.autoclose = autoclose
        self.continuous = continuous

        self.waiting = []
        self.readables = {}
        self.removed = []
        self.poller = select.poll()

        for channel in channels or []:
            self.add(channel)

    # Core methods, registering and reporting on channels.

    def add(self, channel):

        "Add the given 'channel' to the exchange."

        fileno = channel.read_pipe.fileno()
        self.readables[fileno] = channel
        self.poller.register(fileno, select.POLLIN | select.POLLHUP | select.POLLNVAL | select.POLLERR)

    def active(self):

        "Return a list of active channels."

        return list(self.readables.values())

    def ready(self, timeout=None):

        """
        Wait for a period of time specified by the optional 'timeout' in
        milliseconds (or until communication is possible) and return a list of
        channels which are ready to be read from.
        """

        fds = self.poller.poll(timeout)
        readables = []
        self.removed = []

        for fd, status in fds:
            channel = self.readables[fd]
            removed = 0

            # Remove ended/error channels.

            if status & (select.POLLHUP | select.POLLNVAL | select.POLLERR):
                self.remove(channel)
                self.removed.append(channel)
                removed = 1

            # Record readable channels.

            if status & select.POLLIN:
                if not (removed and self.autoclose):
                    readables.append(channel)

        return readables

    def remove(self, channel):

        """
        Remove the given 'channel' from the exchange.
        """

        fileno = channel.read_pipe.fileno()
        del self.readables[fileno]
        self.poller.unregister(fileno)
        if self.autoclose:
            channel.close()
            channel.wait()

    # Enhanced exchange methods involving channel limits.

    def unfinished(self):

        "Return whether the exchange still has work scheduled or in progress."

        return self.active() or self.waiting

    def busy(self):

        "Return whether the exchange uses as many channels as it is allowed to."

        return self.limit is not None and len(self.active()) >= self.limit

    def add_wait(self, channel):

        """
        Add the given 'channel' to the exchange, waiting if the limit on active
        channels would be exceeded by adding the channel.
        """

        self.wait()
        self.add(channel)

    def wait(self):

        """
        Test for the limit on channels, blocking and reading incoming data until
        the number of channels is below the limit.
        """

        # If limited, block until channels have been closed.

        while self.busy():
            self.store()

    def finish(self):

        """
        Finish the use of the exchange by waiting for all channels to complete.
        """

        while self.unfinished():
            self.store()

    def store(self, timeout=None):

        """
        For each ready channel, process the incoming data. If the optional
        'timeout' parameter (a duration in milliseconds) is specified, wait only
        for the specified duration if no channels are ready to provide data.
        """

        # Either process input from active channels.

        if self.active():
            for channel in self.ready(timeout):
                try:
                    self.store_data(channel)
                    self.start_waiting(channel)
                except (IOError, OSError) as exc:
                    self.remove(channel)
                    warn("Removed channel %r due to exception: %s" % (channel, exc))

        # Or schedule new processes and channels.

        else:
            while self.waiting and not self.busy():
                details = self.waiting.pop()

                # Stop actively scheduling if resources are exhausted.

                if not self.start_new_waiting(details):
                    if not self.active():
                        sleep(1)
                    break

    def store_data(self, channel):

        """
        Store incoming data from the specified 'channel'. In subclasses of this
        class, such data could be stored using instance attributes.
        """

        raise NotImplementedError("store_data")

    # Support for the convenience methods.

    def _get_waiting(self, channel):

        """
        Get waiting callable and argument information for new processes, given
        the reception of data on the given 'channel'.
        """

        # For continuous channels, no scheduling is requested.

        if self.waiting and not self.continuous:

            # Schedule this callable and arguments.

            callable, args, kw = self.waiting.pop()

            # Try and reuse existing channels if possible.

            if self.reuse:

                # Re-add the channel - this may update information related to
                # the channel in subclasses.

                self.add(channel)
                channel.send((args, kw))

            # Return the details for a new channel.

            else:
                return callable, args, kw

        # Where channels are being reused, but where no processes are waiting
        # any more, send a special value to tell them to quit.

        elif self.reuse:
            channel.send(None)

        return None

    def _set_waiting(self, callable, args, kw):

        """
        Support process creation by returning whether the given 'callable' has
        been queued for later invocation.
        """

        if self.busy():
            self.waiting.insert(0, (callable, args, kw))
            return 1
        else:
            return 0

    def _get_channel_for_process(self, channel):

        """
        Support process creation by returning the given 'channel' to the
        creating process, and None to the created process.
        """

        if channel.pid == 0:
            return channel
        else:
            self.add_wait(channel)
            return None

    # Methods for overriding, related to the convenience methods.

    def start_waiting(self, channel):

        """
        Start a waiting process given the reception of data on the given
        'channel'.
        """

        details = self._get_waiting(channel)

        if details is not None:
            self.start_new_waiting(details)

    def start_new_waiting(self, details):

        """
        Start a waiting process with the given 'details', obtaining a new
        channel.
        """

        callable, args, kw = details
        channel = self._start(callable, *args, **kw)

        # Monitor any newly-created process.

        if channel:
            self.add(channel)
            return True

        # Push the details back onto the end of the waiting list.

        else:
            self.waiting.append(details)
            return False

    # Convenience methods.

    def start(self, callable, *args, **kw):

        """
        Create a new process for the given 'callable' using any additional
        arguments provided. Then, monitor the channel created between this
        process and the created process.
        """

        if self._set_waiting(callable, args, kw):
            return False

        channel = self._start(callable, *args, **kw)

        # Monitor any newly-created process.

        if channel:
            self.add_wait(channel)
            return True

        # Otherwise, add the details to the waiting list unconditionally.

        else:
            self.waiting.insert(0, (callable, args, kw))
            return False

    def create(self):

        """
        Create a new process and return the created communications channel to
        the created process. In the creating process, return None - the channel
        receiving data from the created process will be automatically managed by
        this exchange.
        """

        channel = create()
        return self._get_channel_for_process(channel)

    def manage(self, callable):

        """
        Wrap the given 'callable' in an object which can then be called in the
        same way as 'callable', but with new processes and communications
        managed automatically.
        """

        return ManagedCallable(callable, self)

    def _start(self, callable, *args, **kw):

        """
        Create a new process for the given 'callable' using any additional
        arguments provided. Return any successfully created channel or None if
        no process could be created at the present time.
        """

        try:
            return start(callable, *args, **kw)
        except OSError as exc:
            if exc.errno != errno.EAGAIN:
                raise
            else:
                return None

class Persistent:

    """
    A mix-in class providing methods to exchanges for the management of
    persistent communications.
    """

    def start_waiting(self, channel):

        """
        Start a waiting process given the reception of data on the given
        'channel'.
        """

        details = self._get_waiting(channel)
        if details is not None:
            callable, args, kw = details
            self.add(start_persistent(channel.address, callable, *args, **kw))

    def start(self, address, callable, *args, **kw):

        """
        Create a new process, located at the given 'address', for the given
        'callable' using any additional arguments provided. Then, monitor the
        channel created between this process and the created process.
        """

        if self._set_waiting(callable, args, kw):
            return

        start_persistent(address, callable, *args, **kw)

    def create(self, address):

        """
        Create a new process, located at the given 'address', and return the
        created communications channel to the created process. In the creating
        process, return None - the channel receiving data from the created
        process will be automatically managed by this exchange.
        """

        channel = create_persistent(address)
        return self._get_channel_for_process(channel)

    def manage(self, address, callable):

        """
        Using the given 'address', publish the given 'callable' in an object
        which can then be called in the same way as 'callable', but with new
        processes and communications managed automatically.
        """

        return PersistentCallable(address, callable, self)

    def connect(self, address):

        "Connect to a process which is contactable via the given 'address'."

        channel = connect_persistent(address)
        self.add_wait(channel)

class ManagedCallable:

    "A callable managed by an exchange."

    def __init__(self, callable, exchange):

        """
        Wrap the given 'callable', using the given 'exchange' to monitor the
        channels created for communications between this and the created
        processes. Note that the 'callable' must be parallel-aware (that is,
        have a 'channel' parameter). Use the MakeParallel class to wrap other
        kinds of callable objects.
        """

        self.callable = callable
        self.exchange = exchange

    def __call__(self, *args, **kw):

        "Invoke the callable with the supplied arguments."

        self.exchange.start(self.callable, *args, **kw)

class PersistentCallable:

    "A callable which sets up a persistent communications channel."

    def __init__(self, address, callable, exchange):

        """
        Using the given 'address', wrap the given 'callable', using the given
        'exchange' to monitor the channels created for communications between
        this and the created processes, so that when it is called, a background
        process is started within which the 'callable' will run. Note that the
        'callable' must be parallel-aware (that is, have a 'channel' parameter).
        Use the MakeParallel class to wrap other kinds of callable objects.
        """

        self.callable = callable
        self.exchange = exchange
        self.address = address

    def __call__(self, *args, **kw):

        "Invoke the callable with the supplied arguments."

        self.exchange.start(self.address, self.callable, *args, **kw)

class BackgroundCallable:

    """
    A callable which sets up a persistent communications channel, but is
    unmanaged by an exchange.
    """

    def __init__(self, address, callable):

        """
        Using the given 'address', wrap the given 'callable'. This object can
        then be invoked, but the wrapped callable will be run in a background
        process. Note that the 'callable' must be parallel-aware (that is, have
        a 'channel' parameter). Use the MakeParallel class to wrap other kinds
        of callable objects.
        """

        self.callable = callable
        self.address = address

    def __call__(self, *args, **kw):

        "Invoke the callable with the supplied arguments."

        start_persistent(self.address, self.callable, *args, **kw)

# Abstractions and utilities.

class Map(Exchange):

    "An exchange which can be used like the built-in 'map' function."

    def __init__(self, *args, **kw):
        Exchange.__init__(self, *args, **kw)
        self.init()

    def init(self):

        "Remember the channel addition order to order output."

        self.channel_number = 0
        self.channels = {}
        self.results = []
        self.current_index = 0

    def add(self, channel):

        "Add the given 'channel' to the exchange."

        Exchange.add(self, channel)
        self.channels[channel] = self.channel_number
        self.channel_number += 1

    def start(self, callable, *args, **kw):

        """
        Create a new process for the given 'callable' using any additional
        arguments provided. Then, monitor the channel created between this
        process and the created process.
        """

        self.results.append(Undefined) # placeholder
        Exchange.start(self, callable, *args, **kw)

    def create(self):

        """
        Create a new process and return the created communications channel to
        the created process. In the creating process, return None - the channel
        receiving data from the created process will be automatically managed by
        this exchange.
        """

        self.results.append(Undefined) # placeholder
        return Exchange.create(self)

    def __call__(self, callable, sequence):

        "Wrap and invoke 'callable' for each element in the 'sequence'."

        if not isinstance(callable, MakeParallel):
            wrapped = MakeParallel(callable)
        else:
            wrapped = callable

        self.init()

        # Start processes for each element in the sequence.

        for i in sequence:
            self.start(wrapped, i)

        # Access to the results occurs through this object.

        return self

    def store_data(self, channel):

        "Accumulate the incoming data, associating results with channels."

        data = channel.receive()
        self.results[self.channels[channel]] = data
        del self.channels[channel]

    def __iter__(self):
        return self

    def next(self):
        """
        Workaround for python2 not understanding that it should look for __next__
        """
        return self.__next__()

    def __next__(self):

        "Return the next element in the map."

        try:
            return self._next()
        except IndexError:
            pass

        while self.unfinished():
            self.store()
            try:
                return self._next()
            except IndexError:
                pass
        else:
            raise StopIteration

    def __getitem__(self, i):

        "Return element 'i' from the map."

        try:
            return self._get(i)
        except IndexError:
            pass

        while self.unfinished():
            self.store()
            try:
                return self._get(i)
            except IndexError:
                pass
        else:
            raise IndexError(i)

    # Helper methods for the above access methods.

    def _next(self):
        result = self._get(self.current_index)
        self.current_index += 1
        return result

    def _get(self, i):
        result = self.results[i]
        if result is Undefined or isinstance(i, slice) and Undefined in result:
            raise IndexError(i)
        return result

class Queue(Exchange):

    """
    An exchange acting as a queue, making data from created processes available
    in the order in which it is received.
    """

    def __init__(self, *args, **kw):
        Exchange.__init__(self, *args, **kw)
        self.queue = []


    def store_data(self, channel):

        "Accumulate the incoming data, associating results with channels."

        data = channel.receive()
        self.queue.insert(0, data)

    def __iter__(self):
        return self

    def next(self):
        """
        Workaround for python2 not understanding that it should look for __next__
        """
        return self.__next__()

    def __next__(self):

        "Return the next element in the queue."

        if self.queue:
            return self.queue.pop()

        while self.unfinished():
            self.store()
            if self.queue:
                return self.queue.pop()
        else:
            raise StopIteration

    def __len__(self):

        "Return the current length of the queue."

        return len(self.queue)

class MakeParallel:

    "A wrapper around functions making them able to communicate results."

    def __init__(self, callable):

        """
        Initialise the wrapper with the given 'callable'. This object will then
        be able to accept a 'channel' parameter when invoked, and to forward the
        result of the given 'callable' via the channel provided back to the
        invoking process.
        """

        self.callable = callable

    def __call__(self, channel, *args, **kw):

        "Invoke the callable and return its result via the given 'channel'."

        channel.send(self.callable(*args, **kw))

class MakeReusable(MakeParallel):

    """
    A wrapper around functions making them able to communicate results in a
    reusable fashion.
    """

    def __call__(self, channel, *args, **kw):

        "Invoke the callable and return its result via the given 'channel'."

        channel.send(self.callable(*args, **kw))
        t = channel.receive()
        while t is not None:
            args, kw = t
            channel.send(self.callable(*args, **kw))
            t = channel.receive()

# Persistent variants.

class PersistentExchange(Persistent, Exchange):

    "An exchange which manages persistent communications."

    pass

class PersistentQueue(Persistent, Queue):

    "A queue which manages persistent communications."

    pass

# Convenience functions.

def BackgroundQueue(address):

    """
    Connect to a process reachable via the given 'address', making the results
    of which accessible via a queue.
    """

    queue = PersistentQueue(limit=1)
    queue.connect(address)
    return queue

def pmap(callable, sequence, limit=None):

    """
    A parallel version of the built-in map function with an optional process
    'limit'. The given 'callable' should not be parallel-aware (that is, have a
    'channel' parameter) since it will be wrapped for parallel communications
    before being invoked.

    Return the processed 'sequence' where each element in the sequence is
    processed by a different process.
    """

    mymap = Map(limit=limit)
    return mymap(callable, sequence)

# Utility functions.

_cpuinfo_fields = "processor", "physical id", "core id"

def _get_number_of_cores():

    """
    Return the number of distinct, genuine processor cores. If the platform is
    not supported by this function, None is returned.
    """

    try:
        f = open("/proc/cpuinfo")
        try:
            processors = set()

            # Use the _cpuinfo_field values as "digits" in a larger unique
            # core identifier.

            processor = [None, None, None]

            for line in f:
                for i, field in enumerate(_cpuinfo_fields):

                    # Where the field is found, insert the value into the
                    # appropriate location in the processor identifier.

                    if line.startswith(field):
                        t = line.split(":")
                        processor[i] = int(t[1].strip())
                        break

                # Where a new processor description is started, record the
                # identifier.

                if line.startswith("processor") and processor[0] is not None:
                    processors.add(tuple(processor))
                    processor = [None, None, None]

            # At the end of reading the file, add any unrecorded processors.

            if processor[0] is not None:
                processors.add(tuple(processor))

            return len(processors)

        finally:
            f.close()

    except OSError:
        return None

def _get_number_of_cores_solaris():

    """
    Return the number of cores for OpenSolaris 2008.05 and possibly other
    editions of Solaris.
    """

    f = os.popen("psrinfo -p")
    try:
        return int(f.read().strip())
    finally:
        f.close()

_system_profiler_field = "Total Number of Cores:"

def _get_number_of_cores_macosx():

    "Return the number of cores for Mac OS X."

    f = os.popen("/usr/sbin/system_profiler -detailLevel full SPHardwareDataType")
    try:
        for line in f:
            line = line.strip()
            if line.startswith(_system_profiler_field):
                return int(line[len(_system_profiler_field):].strip())

        return None

    finally:
        f.close()

# Low-level functions.

def create_socketpair():

    """
    Create a new process, returning a communications channel to both the
    creating process and the created process.
    """

    parent, child = socket.socketpair()
    for s in [parent, child]:
        s.setblocking(1)

    pid = os.fork()
    if pid == 0:
        parent.close()
        return Channel(pid, child.makefile("rb", 0), child.makefile("wb", 0))
    else:
        child.close()
        return Channel(pid, parent.makefile("rb", 0), parent.makefile("wb", 0))

def create_pipes():

    """
    Create a new process, returning a communications channel to both the
    creating process and the created process.

    This function uses pipes instead of a socket pair, since some platforms
    seem to have problems with poll and such socket pairs.
    """

    pr, cw = os.pipe()
    cr, pw = os.pipe()

    pid = os.fork()
    if pid == 0:
        os.close(pr)
        os.close(pw)
        return Channel(pid, os.fdopen(cr, "r", 0), os.fdopen(cw, "w", 0))
    else:
        os.close(cr)
        os.close(cw)
        return Channel(pid, os.fdopen(pr, "r", 0), os.fdopen(pw, "w", 0))

# Configure the interprocess communications and core-counting functions.

if platform.system() == "SunOS":
    create = create_pipes
    get_number_of_cores = _get_number_of_cores_solaris
elif platform.system() == "Darwin":
    create = create_socketpair
    get_number_of_cores = _get_number_of_cores_macosx
else:
    create = create_socketpair
    get_number_of_cores = _get_number_of_cores

def create_persistent(address):

    """
    Create a new process, returning a persistent communications channel between
    the creating process and the created process. This channel can be
    disconnected from the creating process and connected to another process, and
    thus can be used to collect results from daemon processes.

    In order to be able to reconnect to created processes, the 'address' of the
    communications endpoint for the created process needs to be provided. This
    should be a filename.
    """

    parent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    child = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    child.bind(address)

    for s in [parent, child]:
        s.setblocking(1)

    pid = os.fork()
    if pid == 0:
        parent.close()
        return PersistentChannel(pid, child, address)
    else:
        child.close()
        #parent.connect(address)
        return Channel(pid, parent.makefile("rb", 0), parent.makefile("wb", 0))

def connect_persistent(address):

    """
    Connect via a persistent channel to an existing created process, reachable
    at the given 'address'.
    """

    parent = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    parent.setblocking(1)
    parent.connect(address)
    return Channel(0, parent.makefile("rb", 0), parent.makefile("wb", 0))

def exit(channel):

    """
    Terminate a created process, closing the given 'channel'.
    """

    channel.close()
    os._exit(0)

def start(callable, *args, **kw):

    """
    Create a new process which shall start running in the given 'callable'.
    Additional arguments to the 'callable' can be given as additional arguments
    to this function.

    Return a communications channel to the creating process. For the created
    process, supply a channel as the 'channel' parameter in the given 'callable'
    so that it may send data back to the creating process.
    """
    channel = create()
    if channel.pid == 0:
        try:
            try:
                callable(channel, *args, **kw)
            except:
                # force errs to be reported as they occur instead of when we access the results
                exc_type, exc_value, exc_traceback = sys.exc_info()
                if report_all_errs:
                    log.error(exc_value, exc_info=True)
                channel.send(exc_value)
        finally:
            exit(channel)
    else:
        return channel

def start_persistent(address, callable, *args, **kw):

    """
    Create a new process which shall be reachable using the given 'address' and
    which will start running in the given 'callable'. Additional arguments to
    the 'callable' can be given as additional arguments to this function.

    Return a communications channel to the creating process. For the created
    process, supply a channel as the 'channel' parameter in the given 'callable'
    so that it may send data back to the creating process.

    Note that the created process employs a channel which is persistent: it can
    withstand disconnection from the creating process and subsequent connections
    from other processes.
    """

    channel = create_persistent(address)
    if channel.pid == 0:
        close_streams()
        try:
            try:
                callable(channel, *args, **kw)
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                channel.send(exc_value)
        finally:
            exit(channel)
    else:
        return channel

def close_streams():

    """
    Close streams which keep the current process attached to any creating
    processes.
    """

    os.close(sys.stdin.fileno())
    os.close(sys.stdout.fileno())
    os.close(sys.stderr.fileno())

def waitall():

    "Wait for all created processes to terminate."

    try:
        while 1:
            os.wait()
    except OSError:
        pass

# vim: tabstop=4 expandtab shiftwidth=4
