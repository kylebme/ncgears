cmake_minimum_required(VERSION 3.24)

project(ncgear VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CGAL REQUIRED)
find_package(Threads REQUIRED)

add_library(ncgear
  src/gear.cpp
)
target_include_directories(ncgear PUBLIC include)
target_link_libraries(ncgear PUBLIC CGAL::CGAL Threads::Threads)

add_executable(ncgear_generate src/main.cpp)
target_link_libraries(ncgear_generate PRIVATE ncgear)

# Python wheels invoke the generator as a private, bundled executable. Keeping
# the geometry engine out of the Python process also gives callers useful error
# isolation for unusually demanding designs.
install(
  TARGETS ncgear_generate
  RUNTIME DESTINATION ncgear/bin
  COMPONENT python
)

enable_testing()
add_executable(ncgear_tests tests/test_gears.cpp)
target_link_libraries(ncgear_tests PRIVATE ncgear)
add_test(NAME ncgear_tests COMMAND ncgear_tests)

if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|GNU")
  target_compile_options(ncgear PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(ncgear_generate PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
  target_compile_options(ncgear_tests PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()
