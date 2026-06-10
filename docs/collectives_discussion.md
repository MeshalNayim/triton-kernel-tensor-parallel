# Custom collectives vs. MPI built-ins (timing discussion)

myAllreduce uses a reduce-to-root-then-broadcast approach, which requires 2*(n-1) point-to-point messages and serializes all communication through rank 0, making it slower than MPI's built-in Allreduce which uses a recursive halving/doubling or ring algorithm with O(log n) communication rounds and better bandwidth utilization.
Similarly, myAlltoall performs n-1 sequential Sendrecv calls per process, whereas MPI's Alltoall is implemented in optimized native code and can overlap multiple transfers simultaneously using hardware-level scheduling.
As a result, both manual implementations are expected to be slower than their MPI counterparts, with the gap widening as the number of processes or message size increases.
