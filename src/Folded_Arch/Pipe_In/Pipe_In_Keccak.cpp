#include <iostream>
#include <cstdint>
#include <array>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// SYCL implementation of Keccak-f[1600] for FPGA using pipes.
// Uses a pipelined architecture with data feeder, processor, and consumer kernels.
// Processes blocks through SYCL pipes for high-throughput streaming.

const uint64_t RC[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};

inline uint64_t rotl64(uint64_t a, int n) {
    return (n == 0) ? a : ((a << n) | (a >> (64 - n)));
}

// Define SYCL pipes 
using PipeIn = ext::intel::pipe<class KeccakInPipe, std::array<uint64_t, 25>>;
using PipeOut = ext::intel::pipe<class KeccakOutPipe, std::array<uint64_t, 25>>;

// Changed to accept std::array by reference to match the unrolled version
void keccak_f1600_optimized(std::array<uint64_t, 25>& state) {
    uint64_t C[5], D[5];
    uint64_t t0, t1, t2, t3, t4;

    #pragma unroll 1
    for (int i = 0; i < 24; ++i) {
        // THETA 
        C[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        C[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        C[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        C[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        C[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        D[0] = C[4] ^ rotl64(C[1], 1);
        D[1] = C[0] ^ rotl64(C[2], 1);
        D[2] = C[1] ^ rotl64(C[3], 1);
        D[3] = C[2] ^ rotl64(C[4], 1);
        D[4] = C[3] ^ rotl64(C[0], 1);

        state[0] ^= D[0]; state[5] ^= D[0]; state[10] ^= D[0]; state[15] ^= D[0]; state[20] ^= D[0];
        state[1] ^= D[1]; state[6] ^= D[1]; state[11] ^= D[1]; state[16] ^= D[1]; state[21] ^= D[1];
        state[2] ^= D[2]; state[7] ^= D[2]; state[12] ^= D[2]; state[17] ^= D[2]; state[22] ^= D[2];
        state[3] ^= D[3]; state[8] ^= D[3]; state[13] ^= D[3]; state[18] ^= D[3]; state[23] ^= D[3];
        state[4] ^= D[4]; state[9] ^= D[4]; state[14] ^= D[4]; state[19] ^= D[4]; state[24] ^= D[4];

        // RHO & PI 
        t1 = state[1];
        state[1]  = rotl64(state[6], 44);
        state[6]  = rotl64(state[9], 20);
        state[9]  = rotl64(state[22], 61);
        state[22] = rotl64(state[14], 39);
        state[14] = rotl64(state[20], 18);
        state[20] = rotl64(state[2], 62);
        state[2]  = rotl64(state[12], 43);
        state[12] = rotl64(state[13], 25);
        state[13] = rotl64(state[19], 8);
        state[19] = rotl64(state[23], 56);
        state[23] = rotl64(state[15], 41);
        state[15] = rotl64(state[4], 27);
        state[4]  = rotl64(state[24], 14);
        state[24] = rotl64(state[21], 2);
        state[21] = rotl64(state[8], 55);
        state[8]  = rotl64(state[16], 45);
        state[16] = rotl64(state[5], 36);
        state[5]  = rotl64(state[3], 28);
        state[3]  = rotl64(state[18], 21);
        state[18] = rotl64(state[17], 15);
        state[17] = rotl64(state[11], 10);
        state[11] = rotl64(state[7], 6);
        state[7]  = rotl64(state[10], 3);
        state[10] = rotl64(t1, 1);

        // CHI 
        #pragma unroll
        for (int y = 0; y < 25; y += 5) {
            t0 = state[y + 0];
            t1 = state[y + 1];
            t2 = state[y + 2];
            t3 = state[y + 3];
            t4 = state[y + 4];

            state[y + 0] = t0 ^ ((~t1) & t2);
            state[y + 1] = t1 ^ ((~t2) & t3);
            state[y + 2] = t2 ^ ((~t3) & t4);
            state[y + 3] = t3 ^ ((~t4) & t0);
            state[y + 4] = t4 ^ ((~t0) & t1);
        }

        // IOTA 
        state[0] ^= RC[i];
    }
}

int main() {
    try {
        
        #if defined(FPGA_EMULATOR)
            queue q{ext::intel::fpga_emulator_selector_v};
        #else
            queue q{ext::intel::fpga_selector_v};
        #endif

        std::cout << "Target Device: " << q.get_device().get_info<info::device::name>() << "\n";

        int num_blocks = 50000000;

        std::cout << "Synthesising Folded Keccak-f[1600] hardware pipeline with Pipes...\n";

    ]
        q.submit([&](handler& h) {
            h.single_task<class DataFeeder>([=]() {
                for (int i = 0; i < num_blocks; ++i) {
                    std::array<uint64_t, 25> input_block = {0}; 
                    PipeIn::write(input_block);
                }
            });
        });

  
        q.submit([&](handler& h) {
            h.single_task<class KeccakPipeline>([=]() [[intel::kernel_args_restrict]] {
                for (int i = 0; i < num_blocks; ++i) {
                    std::array<uint64_t, 25> local_state = PipeIn::read();
                    
                    keccak_f1600_optimized(local_state);
                    
                    PipeOut::write(local_state);
                }
            });
        });

  
        q.submit([&](handler& h) {
            h.single_task<class DataConsumer>([=]() {
                for (int i = 0; i < num_blocks; ++i) {
                    std::array<uint64_t, 25> output_block = PipeOut::read();
                }
            });
        }).wait(); 

        std::cout << "Kernel execution complete!\n";

    } catch (exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return 1;
    }

    return 0;
}