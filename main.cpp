#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#define GPU_BUFFER_SIZE 1024

int main(int argc, char *argv[]) {
    // Step 1: Allocate memory on the GPU
    char *d_buf = NULL;
    cudaError_t cuda_err = cudaMalloc((void **)&d_buf, GPU_BUFFER_SIZE);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cuda_err));
        return EXIT_FAILURE;
    }
    // Optionally, initialize the GPU memory (e.g., cudaMemset)

    // Step 2: Set up RDMA resources
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    if (!dev_list) {
        perror("Failed to get IB devices list");
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }
    // Open the first available device (in a production code, choose the correct device)
    struct ibv_context *context = ibv_open_device(dev_list[0]);
    if (!context) {
        fprintf(stderr, "Couldn't open device\n");
        ibv_free_device_list(dev_list);
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }
    // Allocate a protection domain (PD)
    struct ibv_pd *pd = ibv_alloc_pd(context);
    if (!pd) {
        fprintf(stderr, "Couldn't allocate PD\n");
        ibv_close_device(context);
        ibv_free_device_list(dev_list);
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }

    // Step 3: Register the GPU memory with RDMA
    // Note: Make sure the GPU memory is registered using IBV_ACCESS flags that suit your operation.
    struct ibv_mr *mr = ibv_reg_mr(pd, d_buf, GPU_BUFFER_SIZE,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        fprintf(stderr, "Couldn't register MR\n");
        ibv_dealloc_pd(pd);
        ibv_close_device(context);
        ibv_free_device_list(dev_list);
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }

    // Steps 4 & 5: Set up Completion Queue (CQ) and Queue Pair (QP), exchange connection info,
    // and perform RDMA operations (e.g., RDMA write) from the remote peer.
    // This section is highly system- and application-specific. On Perlmutter,
    // you may integrate this with MPI (using MVAPICH2-GDR or OpenMPI with CUDA support)
    // or use direct sockets for exchanging QP info.
    //
    // Pseudocode for connection setup:
    // - Create CQ: ibv_create_cq(context, ...);
    // - Create QP: ibv_create_qp(pd, ...);
    // - Transition QP to INIT, RTR, and RTS states.
    // - Exchange QP attributes with the peer.
    //
    // For example, on the sender side:
    //   ibv_post_send(qp, &wr, &bad_wr);
    //
    // And on the receiver side, poll for completions:
    //   ibv_poll_cq(cq, 1, &wc);
    //
    // Once the RDMA write completes, the receiver can launch a CUDA kernel or use cudaMemcpy
    // to copy the data from GPU memory to host memory for verification.

    printf("GPU Direct RDMA test setup complete. MR key: 0x%x, GPU buffer address: %p\n",
           mr->rkey, d_buf);

    // Cleanup (in production code, cleanup after your RDMA transactions)
    ibv_dereg_mr(mr);
    ibv_dealloc_pd(pd);
    ibv_close_device(context);
    ibv_free_device_list(dev_list);
    cudaFree(d_buf);

    return EXIT_SUCCESS;
}
