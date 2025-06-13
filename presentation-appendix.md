
Based on the Habana Document: https://docs.habana.ai/en/latest/Management_and_Monitoring/Network_Configuration/index.html

### 1. How many on-chip ports are actually used for Scale-Up? 

| On-chip 100 GbE RoCE v2 NICs | Used for **Scale-Up** | Left for **Scale-Out** (to the TOR switch) |
| ---------------------------- | --------------------- | ------------------------------------------ |
| 24 per Gaudi 2 die           | **21**                | 3                                          |

### 2. Per-port line-rate
```
100 Gbit/s  = 12.5 GByte/s   (divide by 8)
```

### 3. Bandwidth per Gaudi 2 (all 21 Scale-Up ports)
| Direction   | Formula             | Result                    |
| ----------- | ------------------- | ------------------------- |
| One-way     | 21 ports × 100 Gbps | **2.1 Tb/s ≈ 262.5 GB/s** |
| Full-duplex | 2 × (2.1 Tb/s)      | **4.2 Tb/s ≈ 525 GB/s**   |

Exactly the same figures are called out in the documentation: “21 × 100 Gbps = 262.5 GB/s one-direction, 525 GB/s bidirectional.” 



### 4. Bandwidth between any two Gaudi 2s inside the box
The server’s topology wires three 100 GbE links between every pair of accelerators (21 ports ÷ 7 peers = 3 links).


| Per-pair one-way   |  3 × 100 Gbps = 300 Gbps  = 37.5 GB/s |
|--------------------|---------------------------------------|
| Per-pair bi-dir    | 75 GB/s                               | 


### 5. Aggregate bandwidth for an 8-Gaudi 2 HLS-2 server
Be careful not to double-count links.

1. Count physical links

    * Each Gaudi 2 contributes 21 Scale-Up ports → 8 × 21 = 168 ports.

    * Each physical link consumes two ports → 168 / 2 = 84 physical links.

2. Sum the link rates

    * 84 links × 100 Gbps = 8.4 Tb/s aggregate one-way fabric bandwidth.

    * In bytes/second that is ≈ 1.05 TB/s one-way (≈ 2.1 TB/s full-duplex).

That 1 TB/s-class figure is why Habana marketing often says “~1 TB/s of scale-up bandwidth per HLS-2 node.”

| Scope                              | One-way BW                | Bi-directional BW        |
| ---------------------------------- | ------------------------- | ------------------------ |
| Single Gaudi 2 (21 ports)          | **2.1 Tb/s (262.5 GB/s)** | **4.2 Tb/s (525 GB/s)**  |
| Any Gaudi2 ↔ Gaudi2 pair           | **300 Gb/s (37.5 GB/s)**  | **600 Gb/s (75 GB/s)**   |
| Whole 8-device node (unique links) | **8.4 Tb/s (1.05 TB/s)**  | **16.8 Tb/s (2.1 TB/s)** |


Those are the theoretical peaks, the real HCCL collective throughput will be a bit lower due to protocol overhead, store-and-forward latencies, congestion control, etc., but the math above is what you plug into roof-line or scaling-efficiency analyses.


## HCCL test result analysis:

launch:
```
ENABLE_CONSOLE=true HCCL_COMM_ID=127.0.0.1:5555 python3 run_hccl_demo.py --nranks 8 --node_id 0 --size 1024m --test all_reduce --loop 1000 --ranks_per_node 8

Welcome to HCCL demo
Setting HCCL demo attributes:
clean                = False
list_tests           = False
doc                  = False
nranks               = 8
ranks_per_node       = 8
scaleup_group_size   = None
node_id              = 0
mpi                  = False
test                 = all_reduce
size                 = 1024m
size_range           = None
size_range_inc       = 1
loop                 = 1000
test_root            = 0
ranks_list           = None
data_type            = float
custom_comm          =
no_correctness       = False
reduction_op         = sum
scaleout_bw          = None
result_csv           =
ignore_mpi_errors    = False
no_color             = True
data_csv             =
The user did not set --scaleup_group_size. It is set to 8
HCCL demo runs in pure mode
Affinity: Creating affinity files...
Affinity: Running in pure mode.
Affinity: Running the following command line: MPI_ENABLED=0 NUMA_MAPPING_DIR=/tmp/affinity_topology_output bash common_list_affinity_topology.sh
If isolated_cores is empty, consider all cores as isolated. It will impact affinity level.
HCCL demo test command line:
HCCL_DEMO_TEST=all_reduce HCCL_DATA_TYPE=float HCCL_DEMO_TEST_SIZE=1073741824 HCCL_DEMO_TEST_LOOP=1000 HCCL_REDUCTION_OP=sum HCCL_DEMO_TEST_ROOT=0 HCCL_DEMO_MPI_REQUESTED=0 MPI_ENABLED=0 NUMA_MAPPING_DIR=/tmp/affinity_topology_output ID=0 HCCL_RANK=0 HCCL_NRANKS=8 HCCL_RANKS_PER_NODE=8 HCCL_SCALEUP_GROUP_SIZE=8 ./hccl_demo
......
Setting auto-affinity
filename = /tmp/affinity_topology_output/.habana_moduleID3
20 21 22 23 24 25 26 27 28 29
moduleID=3 affinity set to = 0000000000000000000011111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
......
rank=3 size=1073741824 <float> Input Buffer [3 11 19 27 ...] Output Buffer [28 92 156 220 ...] which is fine.
rank=5 size=1073741824 <float> Input Buffer [5 13 21 29 ...] Output Buffer [28 92 156 220 ...] which is fine.
.......
[BENCHMARK] hcclAllReduce(dataSize=1073741824, count=268435456, dtype=float, iterations=1000)
[BENCHMARK]     NW Bandwidth   : 258.200959 GB/s
[BENCHMARK]     Algo Bandwidth : 147.543405 GB/s
#########################################
Core affinity optimization result:
Physical core  - enabled
NUMA           - enabled
Core isolation - not enabled
```

The above ~1 GiB, 8-rank all-reduce run is actually a sanity-check of the earlier calculation:

| metric (per-rank)  | what it counts                                              | expected on Gaudi 2                                                 | test result                   |
| ------------------ | ----------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------- |
| **NW Bandwidth**   | raw bytes that hit the wire (all 21 x 100 GbE links) ÷ time | 21 ports × 12.5 GB/s ≈ **262.5 GB/s** one-way  | **258.2 GB/s** (98 % of peak) |
| **Algo Bandwidth** | “useful” payload of the collective ÷ time                   | NW BW ÷ (2·(n-1)/n) → 262.5 / 1.75 ≈ **150 GB/s**                   | **147.5 GB/s**                |

hccl_demo implements the classic ring all-reduce. so,
Calculated as: Algo BW × scaling factor
For all_reduce: scaling factor = 2×(n-1)/n = 2×7/8 = 1.75 which accounts for the ring-allreduce algorithm requiring each rank to send/receive data multiple times.

## Scaling-factor
For any collec let **S (bytes)** be the user payload that each rank cares about (the tensor you pass to the op).
Let **T (bytes)** be the total traffic that actually crosses the NIC fabric when HCCL executes the collective once.

We define the **scaling-factor** as: 
      $$f(n) = {T\over S}$$ (bytes put on the wire per useful byte)

The larger $f(n)$ is, the more network work the collective has to do for the same amount of application data, so the measured “Algo-BW” is simply 
   $${AlgoBW} = {{NWBW}\over f(n)}$$

Below are the factors for the algorithms HCCL uses today on Gaudi-2 (ring or pairwise-exchange; no topology tricks). *n = number of ranks in the communicator.*

| Collective            | Description                                                                      | HCCL algorithm                                     | Scaling-factor **f(n)** | Example **n = 8** | Notes                                                                                                             |
| --------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------- | ----------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| **broadcast**         | One sender, everyone receives                                                    | Pipelined **ring** fan-out                         | $(n-1)/n$               | 7/8  ≈ 0.875      | Every chunk makes $n-1$ hops around the ring; each rank forwards what it has already received.                    |
| **all-reduce**        | All ranks contribute, all get the result                                         | 2-phase **ring** <br>(reduce-scatter + all-gather) | $2\frac{(n-1)}{n}$      | $2·7/8 = 1.75$    | Same factor you measured: NW-traffic is 1.75 × the tensor size.                                                   |
| **reduce-scatter**    | Reduce, but each rank keeps one slice                                            | 1-phase **ring**                                   | $(n-1)/n$               | 0.875             | Only the reduce-scatter half of all-reduce.                                                                       |
| **all-gather**        | Everyone gets the concatenation of all slices                                    | 1-phase **ring**                                   | $(n-1)/n$               | 0.875             | Only the gather half of all-reduce.                                                                               |
| **all-to-all**        | Every rank sends a distinct piece to every other rank                            | **pairwise-exchange** steps                        | $n-1$                   | 7.0               | Each rank must transmit its full tensor to each of $n-1$ peers and receive as much—a bandwidth-intensive pattern. |
| **reduce**            | Many-to-one reduction into the root                                              | Pipelined **tree**                                 | $(n-1)/n$               | 0.875             | A k-ary tree has $(n-1)$ sends of size $S/n$ each. Tree height gives the same factor as ring.                     |
| **send / recv**       | Direct point-to-point                                                            | –                                                  | **1**                   | 1.0               | Only the sender transmits; nothing is forwarded.                                                                  |
| **scale\_validation** | HCCL “bandwidth validation test” (many rank pairs fire `send_recv` concurrently) | independent **send\_recv** streams                 | **1** (per stream)      | 1.0               | The tool just runs multiple p2p streams to saturate the fabric; no extra hops.                                    |


