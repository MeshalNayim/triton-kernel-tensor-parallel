from mpi4py import MPI
import numpy as np

class Communicator(object):
    def __init__(self, comm: MPI.Comm):
        self.comm = comm
        self.total_bytes_transferred = 0

    def Get_size(self):
        return self.comm.Get_size()

    def Get_rank(self):
        return self.comm.Get_rank()

    def Barrier(self):
        return self.comm.Barrier()

    def Allreduce(self, src_array, dest_array, op=MPI.SUM):
        assert src_array.size == dest_array.size
        src_array_byte = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_array_byte * 2 * (self.comm.Get_size() - 1)
        self.comm.Allreduce(src_array, dest_array, op)

    def Allgather(self, src_array, dest_array):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Allgather(src_array, dest_array)

    def Reduce_scatter(self, src_array, dest_array, op=MPI.SUM):
        src_array_byte = src_array.itemsize * src_array.size
        dest_array_byte = dest_array.itemsize * dest_array.size
        self.total_bytes_transferred += src_array_byte * (self.comm.Get_size() - 1)
        self.total_bytes_transferred += dest_array_byte * (self.comm.Get_size() - 1)
        self.comm.Reduce_scatter_block(src_array, dest_array, op)

    def Split(self, key, color):
        return __class__(self.comm.Split(key=key, color=color))

    def Alltoall(self, src_array, dest_array):
        nprocs = self.comm.Get_size()

        assert src_array.size % nprocs == 0, (
            "src_array size must be divisible by the number of processes"
        )
        assert dest_array.size % nprocs == 0, (
            "dest_array size must be divisible by the number of processes"
        )

        send_seg_bytes = src_array.itemsize * (src_array.size // nprocs)
        recv_seg_bytes = dest_array.itemsize * (dest_array.size // nprocs)

        
        self.total_bytes_transferred += send_seg_bytes * (nprocs - 1)
        self.total_bytes_transferred += recv_seg_bytes * (nprocs - 1)

        self.comm.Alltoall(src_array, dest_array)

    def myAllreduce(self, src_array, dest_array, op=MPI.SUM):
        """
        A manual implementation of all-reduce using a reduce-to-root
        followed by a broadcast.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.
        Your implementation should respect the passed reduction operator.
        The required operators for this assignment are MPI.MIN, MPI.SUM,
        and MPI.MAX.

        Each non-root process sends its data to process 0, which applies the
        reduction operator (by default, summation). Then process 0 sends the
        reduced result back to all processes.

        The transfer cost is computed as:
          - For non-root processes: one send and one receive.
          - For the root process: (n-1) receives and (n-1) sends.
        """
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        src_bytes = src_array.itemsize * src_array.size
        self.total_bytes_transferred += src_bytes * 2 * (size - 1)

        if rank == 0:
            dest_array[:] = src_array
            temp = np.empty_like(src_array)
            for i in range(1, size):
                self.comm.Recv(temp, source=i)
                if op == MPI.SUM:
                    dest_array += temp
                elif op == MPI.MIN:
                    np.minimum(dest_array, temp, out=dest_array)
                elif op == MPI.MAX:
                    np.maximum(dest_array, temp, out=dest_array)
            for i in range(1, size):
                self.comm.Send(dest_array, dest=i)
        else:
            self.comm.Send(src_array, dest=0)
            self.comm.Recv(dest_array, source=0)

    def myAlltoall(self, src_array, dest_array):
        """
        A manual implementation of all-to-all where each process sends a
        distinct segment of its source array to every other process.

        Do not call built-in MPI collective operations inside this method.
        Use point-to-point communication such as Send, Recv, or Sendrecv.

        It is assumed that the total length of src_array (and dest_array)
        is evenly divisible by the number of processes.

        The algorithm loops over the ranks:
          - For the local segment (when destination == self), a direct copy is done.
          - For all other segments, the process exchanges the corresponding
            portion of its src_array with the other process via Sendrecv.

        The total data transferred is updated for each pairwise exchange.
        """
        nprocs = self.comm.Get_size()
        rank = self.comm.Get_rank()

        src_flat = src_array.reshape(-1)
        dest_flat = dest_array.reshape(-1)
        seg_size = src_flat.size // nprocs
        seg_bytes = src_array.itemsize * seg_size

        for j in range(nprocs):
            src_seg = np.ascontiguousarray(src_flat[j * seg_size:(j + 1) * seg_size])
            if j == rank:
                dest_flat[j * seg_size:(j + 1) * seg_size] = src_seg
            else:
                recv_buf = np.empty(seg_size, dtype=src_array.dtype)
                self.comm.Sendrecv(src_seg, dest=j, recvbuf=recv_buf, source=j)
                dest_flat[j * seg_size:(j + 1) * seg_size] = recv_buf
                self.total_bytes_transferred += seg_bytes * 2
