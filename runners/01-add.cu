#include "KernelAdd.cuh"

int main() {
  // Good source:
  //   https://cc.dvfu.ru/ru/lesson-1/
  // Also a good explaination of those
  //   blocks, that are not related to
  //   architecture, but rather to
  //   cuda language and computing.
  //   https://stackoverflow.com/a/2392271

  static const int NUM_ELEMENTS_MAX    = 1 << 26;             // 256MB, if ints
  static const int NUM_STEPS           = 1 << 10;             // 1024.
  static const int NUM_ELEMENTS_STEP   = NUM_ELEMENTS_MAX / NUM_STEPS;

  // Cores in the same block have a common register file.
  // Those are specialized cores, each of them has a type and can do only operations of it's type.
  static const int TURING_CORES_PER_BLOCK   = 32;
  // Blocks in the same SM share L1 cache, so if we use one SM fully may have better performance, less memory reads.
  static const int TURING_ARCH_BLOCKS_IN_SM = 4;
  static const int TURING_MIN_NUM_SM        = 5; // Picked out of nowhere, not based on some docs and etc.

  for (int num_elements = NUM_ELEMENTS_STEP; num_elements <= NUM_ELEMENTS_MAX; num_elements += NUM_ELEMENTS_STEP) {
    for (int num_blocks = 1; num_blocks <= TURING_MIN_NUM_SM * TURING_ARCH_BLOCKS_IN_SM; ++num_blocks) {

      KernelAdd<<num_blocks, >>
    }
  }


  return 0;
}
