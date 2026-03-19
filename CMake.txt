cmake_minimum_required(VERSION 3.20)
project(Keccak_FPGA CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# source file
set(SRC src/Optimised_cpp.cpp)

set(FPGA_FLAGS "-fsycl -fintelfpga")

# half compilation
# add_executable(keccak_report ${SRC})
# target_compile_options(keccak_report PRIVATE ${FPGA_FLAGS} "-fsycl-link=early")
# target_link_options(keccak_report PRIVATE ${FPGA_FLAGS} "-fsycl-link=early")

## Full Comp - full Quartus
add_executable(keccak_hardware ${SRC})
target_compile_options(keccak_hardware PRIVATE ${FPGA_FLAGS} "-Xshardware")
target_link_options(keccak_hardware PRIVATE ${FPGA_FLAGS} "-Xshardware")