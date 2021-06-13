# Extractor library
add_library(llvm_extractor
        llvm_extractor.cc
        llvm_graph_pass.cc
        llvm_graph_funcinfo.cc
        llvm_seq_pass.cc
	llvm_msf_pass.cc
	llvm_names_pass.cc
  llvm_ir2vec_pass.cc
	llvm_insts_pass.cc
        llvm_wl_cost_pass.cc
	llvm_wl_cost_funcinfo.cc
        )
target_link_libraries(llvm_extractor
	wl_static_profiler
        extractors_common
        libIR2Vec.a
        )
target_compile_options(llvm_extractor PRIVATE
        ${compile_options_common}
        )

# Extractor tests
add_executable(llvm_extractor_tests
        llvm_pass_test.cc
        llvm_extractor_test.cc
        )
target_link_libraries(llvm_extractor_tests
        llvm_extractor
        gmock
        gtest
        gtest_main
        )
target_compile_options(llvm_extractor_tests PRIVATE
        -fno-rtti -fPIC
        )

add_subdirectory(wl_static_profiler)
