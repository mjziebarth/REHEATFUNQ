# Wrapper around the eigen3 headers.
cmake_minimum_required(VERSION 3.4.0)
project(eigen)

#set(THREADS_PREFER_PTHREAD_FLAG ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#add_compile_options(-DSQLITE_ENABLE_RTREE=1)
#find_package(Threads REQUIRED)

add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE Eigen)
add_executable(sqlite3_exe sqlite3.h sqlite3ext.h shell.c sqlite3.c)
#add_link_options(sqlite3 Threads)
install(DIRECTORY Eigen/ DESTINATION include FILES_MATCHING PATTERN
target_link_libraries(sqlite3_exe PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(sqlite3_exe PROPERTIES OUTPUT_NAME sqlite3)
install(FILES sqlite3ext.h sqlite3.h DESTINATION include)
install(TARGETS sqlite3 DESTINATION /usr/lib)
install(TARGETS sqlite3_exe RUNTIME DESTINATION /usr/bin)
