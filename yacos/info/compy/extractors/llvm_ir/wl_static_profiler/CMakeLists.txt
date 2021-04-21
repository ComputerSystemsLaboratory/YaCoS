# Extractor library
add_library(wl_static_profiler
	block_edge_frequency_pass.cc
	branch_heuristics_info.cc
	branch_prediction_info.cc
	branch_prediction_pass.cc
        )
target_link_libraries(wl_static_profiler
        extractors_common
        )
target_compile_options(wl_static_profiler PRIVATE
        ${compile_options_common}
        )
